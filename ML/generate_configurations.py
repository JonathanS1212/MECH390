

import numpy as np
import torch
import torch.nn as nn
import pandas as pd

# ── 0.  Configuration ────────────────────────────────────────────────────────
CHECKPOINT = r"/Users/Jonathan/School/Python/MECH390/ML/saved_model/Sohoite_predictor.pt"
N_CANDIDATES = 50_000     # random pool to sample from before filtering
N_RESULTS    = 5          # how many configurations to return
SEED         = 42

# Fixed inputs
RPM_FIXED     = 30.0
FBOX_FIXED    = 4.5
DENSITY_FIXED = 2700.0

# Output constraints
DX_TARGET  = 0.250        # desired predicted dx
DX_TOL     = 0.05         # accept dx within ± this tolerance
QRR_MIN    = 1.5
QRR_MAX    = 2.5

# Free input ranges (from dataset statistics)
FREE_RANGES = {
    #  feature   : (min,    max)
    'r'          : (0.036,  0.851),
    'e'          : (0.002,  0.661),
    'l'          : (0.100,  0.999),
    'Ls'         : (0.150,  1.899),
    'Height'     : (0.015,  0.035),
    'Width'      : (0.005,  0.012),
    'Pin dia'    : (0.004,  0.008),
}

# ── 1.  Sohoite architecture ─────────────────────────────────────────────────
class Sohoite(nn.Module):
    def __init__(self, input_dim, n_output, dropout=0.1):
        super().__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(input_dim, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(dropout))
        self.hidden2 = nn.Sequential(
            nn.Linear(64, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(dropout))
        self.drop  = nn.Dropout(dropout)
        self.skip  = nn.Linear(input_dim, 64)
        self.heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(n_output)])
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        h1  = self.hidden1(x)
        h2  = self.drop(self.hidden2(h1) + self.skip(x))
        return torch.cat([head(h2) for head in self.heads], dim=1)

# ── 2.  Load checkpoint ──────────────────────────────────────────────────────
ckpt       = torch.load(CHECKPOINT, map_location='cpu')
input_col  = ckpt['input_col']    # ['r','e','l','Ls','Height','Width','Density','Pin dia','RPM','Fbox']
out_col = ckpt['out_col']   # ['|P1| Max','|B0| Max','FOS','Torque','QRR','dx','Bearing']
x_mean     = ckpt['x_mean']
x_std      = ckpt['x_std']
y_mean     = ckpt['y_mean']
y_std      = ckpt['y_std']
n_input    = len(input_col)
n_output   = len(out_col)

model = Sohoite(input_dim=n_input, n_output=n_output)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Sohoite loaded — {n_input} inputs, {n_output} outputs\n")

# ── 3.  Build candidate pool ─────────────────────────────────────────────────
rng = np.random.default_rng(SEED)

# Sample free parameters randomly within their observed ranges
free_cols  = list(FREE_RANGES.keys())
candidates = np.zeros((N_CANDIDATES, n_input), dtype=np.float32)

for fi, feat in enumerate(input_col):
    if feat in FREE_RANGES:
        lo, hi = FREE_RANGES[feat]
        candidates[:, fi] = rng.uniform(lo, hi, size=N_CANDIDATES)
    elif feat == 'RPM':
        candidates[:, fi] = RPM_FIXED
    elif feat == 'Fbox':
        candidates[:, fi] = FBOX_FIXED
    elif feat == 'Density':
        candidates[:, fi] = DENSITY_FIXED

# ── 4.  Run Sohoite on all candidates ────────────────────────────────────────
X_t    = torch.tensor(candidates)
X_norm = (X_t - x_mean) / x_std

with torch.no_grad():
    y_norm = model(X_norm)

y_real = (y_norm * y_std + y_mean).numpy()   # (N_CANDIDATES, n_output)

# ── 5.  Filter by constraints ─────────────────────────────────────────────────
# Output column indices
idx_qrr = out_col.index('QRR')
idx_dx  = out_col.index('dx')

pred_qrr = y_real[:, idx_qrr]
pred_dx  = y_real[:, idx_dx]

qrr_ok  = (pred_qrr >= QRR_MIN) & (pred_qrr <= QRR_MAX)
dx_ok   = np.abs(pred_dx - DX_TARGET) <= DX_TOL
valid   = qrr_ok & dx_ok

n_valid = valid.sum()
print(f"Candidates satisfying constraints: {n_valid} / {N_CANDIDATES}")

if n_valid < N_RESULTS:
    print(f"\nOnly {n_valid} candidates met the constraints.")
    print("Consider relaxing DX_TOL or the QRR range.")
    valid_idx = np.where(valid)[0]
else:
    # Among valid candidates, pick N_RESULTS spread across the QRR range
    # (sort by QRR so the 5 configs cover different operating points)
    valid_idx = np.where(valid)[0]
    qrr_sorted = np.argsort(pred_qrr[valid_idx])
    step = max(1, len(qrr_sorted) // N_RESULTS)
    chosen = [qrr_sorted[i * step] for i in range(N_RESULTS)]
    valid_idx = valid_idx[chosen]

# ── 6.  Build results table ───────────────────────────────────────────────────
rows = []
for rank, ci in enumerate(valid_idx, start=1):
    row = {'Config': rank}
    # Inputs
    for fi, feat in enumerate(input_col):
        row[feat] = round(float(candidates[ci, fi]), 6)
    # Predicted outputs
    for oi, out in enumerate(out_col):
        row[f'Pred {out}'] = round(float(y_real[ci, oi]), 4)
    rows.append(row)

df_results = pd.DataFrame(rows).set_index('Config')

# ── 7.  Display ───────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  INPUTS")
print("="*70)
print(df_results[input_col].to_string())

print("\n" + "="*70)
print("  PREDICTED OUTPUTS")
print("="*70)
pred_cols = [f'Pred {c}' for c in out_col]
print(df_results[pred_cols].to_string())

# ── 8.  Save to Excel ─────────────────────────────────────────────────────────
out_file = "configurations_5.xlsx"
with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
    df_results[input_col].to_excel(writer, sheet_name='Inputs')
    df_results[pred_cols].to_excel(writer, sheet_name='Predicted Outputs')

print(f"\nResults saved to  '{out_file}'")
