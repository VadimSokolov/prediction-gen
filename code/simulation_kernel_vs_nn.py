"""
Predictive Quantile Regression: Local Kernel vs Global Neural Network
Comparing efficiency bounds for p(X_{n+1} | X_1, ..., X_n)
Based on Schmidt-Hieber & Zamolodtchikov (2024) and Polson-Sokolov (2023)

The AR(2) DGP has intrinsic dimension 2 (true quantile depends on X_{t-1},
X_{t-2} only).  The kernel operates on the full d=20 lag vector; the neural
network receives k=3 summary statistics.  Proposition 1 predicts rates
m^{-0.50} (NN, k=3) vs m^{-0.17} (kernel, d=20), a large gap.
"""

import numpy as np
import torch
import torch.nn as nn
import os, time

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cpu")

###############################################################################
# 1. DATA GENERATING PROCESS  (AR(2), heteroscedastic, beta=2)
###############################################################################

def generate_time_series(n_sequences, sequence_length=50, seed=None):
    rng = np.random.RandomState(seed)
    data = []
    for _ in range(n_sequences):
        X = np.zeros(sequence_length)
        X[0], X[1] = rng.randn(), rng.randn()
        for t in range(2, sequence_length):
            mu_t = 0.5 * X[t-1] - 0.3 * X[t-2]
            sigma_t = np.sqrt(0.5 + 0.3 * X[t-1]**2 + 0.2 * X[t-2]**2)
            X[t] = mu_t + rng.randn() * sigma_t
        data.append(X)
    return data

###############################################################################
# 2. PREPARE DATA
###############################################################################

N_LAGS = 20
QUANTILE_LEVELS = np.array([0.1, 0.25, 0.5, 0.75, 0.9])

def create_lagged_data(sequences, n_lags=N_LAGS):
    histories, targets, summaries = [], [], []
    for X in sequences:
        for t in range(n_lags, len(X) - 1):
            history = X[t - n_lags + 1 : t + 1]
            histories.append(history)
            targets.append(X[t + 1])
            summaries.append([np.mean(history), np.std(history, ddof=1), X[t]])
    return (np.array(histories, dtype=np.float32),
            np.array(targets, dtype=np.float32),
            np.array(summaries, dtype=np.float32))

###############################################################################
# 3. KERNEL QUANTILE REGRESSION (Nadaraya-Watson, full d-dim input)
###############################################################################

def kernel_quantile_predict(X_train, y_train, X_test, tau_levels, chunk=500):
    mu, sigma = X_train.mean(0), X_train.std(0)
    sigma[sigma == 0] = 1.0
    X_tr = (X_train - mu) / sigma
    X_te = (X_test  - mu) / sigma

    d, n = X_tr.shape[1], len(X_tr)
    h = (4.0 / (d + 2)) ** (1.0 / (d + 4)) * n ** (-1.0 / (d + 4))
    inv_2h2 = 1.0 / (2 * h ** 2)

    sort_idx = np.argsort(y_train)
    y_sorted = y_train[sort_idx]
    X_tr_sq = np.sum(X_tr ** 2, axis=1)

    preds = np.empty((len(X_te), len(tau_levels)))
    for s in range(0, len(X_te), chunk):
        e = min(s + chunk, len(X_te))
        c = X_te[s:e]
        d2 = np.sum(c ** 2, axis=1, keepdims=True) + X_tr_sq[None, :] - 2.0 * c @ X_tr.T
        np.maximum(d2, 0, out=d2)
        w = np.exp(-d2 * inv_2h2)
        ws = w.sum(axis=1, keepdims=True); ws[ws == 0] = 1.0; w /= ws
        cumw = np.cumsum(w[:, sort_idx], axis=1)
        for q, tau in enumerate(tau_levels):
            preds[s:e, q] = y_sorted[np.argmax(cumw >= tau, axis=1)]
    return preds

###############################################################################
# 4. SINGLE MULTI-QUANTILE NEURAL NETWORK  (tau as input)
###############################################################################

class MultiQuantileNet(nn.Module):
    """Takes (summary, tau) as input; outputs scalar quantile prediction."""
    def __init__(self, summary_dim, hidden=(128, 128, 64)):
        super().__init__()
        layers = []
        prev = summary_dim + 1          # +1 for tau
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

def pinball_loss(y_pred, y_true, tau):
    err = y_true - y_pred
    return torch.mean(torch.maximum(tau * err, (tau - 1) * err))

def train_multi_qnn(X_summ, y, tau_levels, epochs=50, batch_size=256, lr=1e-3):
    """Train a single network on all tau levels jointly."""
    n = len(X_summ)
    K = len(tau_levels)

    # Replicate data for each tau: (n*K, summary_dim+1)
    X_rep = np.repeat(X_summ, K, axis=0)
    tau_col = np.tile(tau_levels, n).reshape(-1, 1).astype(np.float32)
    X_aug = np.hstack([X_rep, tau_col])
    y_rep = np.repeat(y, K)
    tau_rep = np.tile(tau_levels, n).astype(np.float32)

    X_t = torch.tensor(X_aug, device=device)
    y_t = torch.tensor(y_rep, device=device)
    tau_t = torch.tensor(tau_rep, device=device)

    # Train / val split
    n_total = len(X_t)
    n_val = int(0.2 * n_total)
    perm = torch.randperm(n_total)
    tr_idx, val_idx = perm[n_val:], perm[:n_val]

    model = MultiQuantileNet(X_summ.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, best_state, patience = float('inf'), None, 0

    for _ in range(epochs):
        model.train()
        for s in range(0, len(tr_idx), batch_size):
            b = tr_idx[torch.randperm(len(tr_idx))[s:s+batch_size]]
            optimizer.zero_grad()
            loss = pinball_loss(model(X_t[b]), y_t[b], tau_t[b])
            loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            vl = pinball_loss(model(X_t[val_idx]), y_t[val_idx], tau_t[val_idx]).item()
        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 10: break

    model.load_state_dict(best_state)
    return model

def predict_multi_qnn(model, X_summ, tau_levels):
    model.eval()
    preds = np.empty((len(X_summ), len(tau_levels)))
    X_t = torch.tensor(X_summ, device=device)
    with torch.no_grad():
        for q, tau in enumerate(tau_levels):
            tau_col = torch.full((len(X_t), 1), tau, device=device)
            preds[:, q] = model(torch.cat([X_t, tau_col], dim=1)).cpu().numpy()
    return preds

###############################################################################
# 5. EVALUATION
###############################################################################

def evaluate(y_test, pred_matrix, tau_levels):
    K = len(tau_levels)
    losses = np.empty(K)
    coverage = np.empty(K)
    for q in range(K):
        err = y_test - pred_matrix[:, q]
        losses[q] = np.mean(np.where(err >= 0, tau_levels[q]*err, (tau_levels[q]-1)*err))
        coverage[q] = np.mean(y_test <= pred_matrix[:, q])
    return {'avg_loss': np.mean(losses), 'coverage_mae': np.mean(np.abs(coverage - tau_levels)),
            'losses': losses, 'coverage': coverage}

###############################################################################
# 6. MAIN EXPERIMENT
###############################################################################

N_REPS  = 3
N_TRAIN = 1000
N_TEST  = 100

results = []
t0 = time.time()

for rep in range(1, N_REPS + 1):
    print(f"\n=== Replication {rep}/{N_REPS} ===", flush=True)
    train_seq = generate_time_series(N_TRAIN, seed=42 + rep)
    test_seq  = generate_time_series(N_TEST,  seed=1000 + rep)

    X_train_full, y_train, X_train_summ = create_lagged_data(train_seq)
    X_test_full,  y_test,  X_test_summ  = create_lagged_data(test_seq)

    print(f"  Data: {len(y_train)} train, {len(y_test)} test  (d={X_train_full.shape[1]}, k={X_train_summ.shape[1]})")

    # Kernel on full d-dim input
    print("  Kernel quantile regression...", flush=True)
    kern_pred = kernel_quantile_predict(X_train_full, y_train, X_test_full, QUANTILE_LEVELS)
    kern_eval = evaluate(y_test, kern_pred, QUANTILE_LEVELS)

    # Single NN on k=3 summaries
    print("  Neural network quantile regression...", flush=True)
    torch.manual_seed(42 + rep * 100)
    model = train_multi_qnn(X_train_summ, y_train, QUANTILE_LEVELS, epochs=50)
    nn_pred = predict_multi_qnn(model, X_test_summ, QUANTILE_LEVELS)
    nn_eval = evaluate(y_test, nn_pred, QUANTILE_LEVELS)

    results.append({
        'kern_loss': kern_eval['avg_loss'], 'kern_cov': kern_eval['coverage_mae'],
        'nn_loss': nn_eval['avg_loss'],     'nn_cov': nn_eval['coverage_mae'],
        'kern_losses_per_tau': kern_eval['losses'], 'kern_cov_per_tau': kern_eval['coverage'],
        'nn_losses_per_tau': nn_eval['losses'],     'nn_cov_per_tau': nn_eval['coverage'],
    })
    elapsed = time.time() - t0
    print(f"  Kernel: loss={kern_eval['avg_loss']:.4f}  cov_mae={kern_eval['coverage_mae']:.4f}")
    print(f"  NN:     loss={nn_eval['avg_loss']:.4f}  cov_mae={nn_eval['coverage_mae']:.4f}")
    print(f"  [{elapsed:.0f}s elapsed]", flush=True)

###############################################################################
# 7. SUMMARY
###############################################################################

print("\n=== SUMMARY ===\n")
kern_mean_loss = np.mean([r['kern_loss'] for r in results])
kern_se_loss   = np.std([r['kern_loss'] for r in results], ddof=1) / np.sqrt(N_REPS)
kern_mean_cov  = np.mean([r['kern_cov'] for r in results])
kern_se_cov    = np.std([r['kern_cov'] for r in results], ddof=1) / np.sqrt(N_REPS)
nn_mean_loss   = np.mean([r['nn_loss'] for r in results])
nn_se_loss     = np.std([r['nn_loss'] for r in results], ddof=1) / np.sqrt(N_REPS)
nn_mean_cov    = np.mean([r['nn_cov'] for r in results])
nn_se_cov      = np.std([r['nn_cov'] for r in results], ddof=1) / np.sqrt(N_REPS)

print(f"  Kernel (d={N_LAGS}):  loss = {kern_mean_loss:.4f} (SE {kern_se_loss:.4f}), "
      f"cov MAE = {kern_mean_cov:.4f} (SE {kern_se_cov:.4f})")
print(f"  NN     (k=3):    loss = {nn_mean_loss:.4f} (SE {nn_se_loss:.4f}), "
      f"cov MAE = {nn_mean_cov:.4f} (SE {nn_se_cov:.4f})")
pct = 100 * (kern_mean_loss - nn_mean_loss) / kern_mean_loss
print(f"\n  Relative improvement: {pct:.1f}%")

beta = 2
rate_nn = 2*beta / (2*beta + 3 + 1)
rate_kern = 2*beta / (2*beta + N_LAGS)
print(f"\n  Theoretical rates:  NN m^(-{rate_nn:.2f})  vs  Kernel m^(-{rate_kern:.2f})")

###############################################################################
# 8. WRITE LEDGER
###############################################################################

script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, '..', 'results')
os.makedirs(results_dir, exist_ok=True)
ledger_path = os.path.join(results_dir, 'numbers.txt')

from datetime import datetime
with open(ledger_path, 'w') as f:
    f.write("=== Number Provenance Ledger ===\n")
    f.write(f"Script: simulation_kernel_vs_nn.py\n")
    f.write(f"Date: {datetime.now()}\n")
    f.write(f"Config: N_LAGS={N_LAGS}, N_TRAIN={N_TRAIN}, N_TEST={N_TEST}, N_REPS={N_REPS}\n")
    f.write(f"Quantile levels: {list(QUANTILE_LEVELS)}\n\n")
    f.write("Table 1:\n")
    f.write(f"  Kernel avg quantile loss: {kern_mean_loss:.4f} (SE: {kern_se_loss:.4f})\n")
    f.write(f"  NN avg quantile loss:     {nn_mean_loss:.4f} (SE: {nn_se_loss:.4f})\n")
    f.write(f"  Kernel coverage MAE:      {kern_mean_cov:.4f} (SE: {kern_se_cov:.4f})\n")
    f.write(f"  NN coverage MAE:          {nn_mean_cov:.4f} (SE: {nn_se_cov:.4f})\n")
    f.write(f"  Relative improvement:     {pct:.1f}%\n")
    f.write(f"\nTheoretical rates: NN m^(-{rate_nn:.2f}), Kernel m^(-{rate_kern:.2f})\n")
print(f"\nResults written to {ledger_path}")

###############################################################################
# 9. FIGURE
###############################################################################

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

kern_losses_tau = np.mean([r['kern_losses_per_tau'] for r in results], axis=0)
nn_losses_tau   = np.mean([r['nn_losses_per_tau']   for r in results], axis=0)
kern_cov_tau    = np.mean([r['kern_cov_per_tau']    for r in results], axis=0)
nn_cov_tau      = np.mean([r['nn_cov_per_tau']      for r in results], axis=0)

kern_losses_se = np.std([r['kern_losses_per_tau'] for r in results], axis=0, ddof=1) / np.sqrt(N_REPS)
nn_losses_se   = np.std([r['nn_losses_per_tau']   for r in results], axis=0, ddof=1) / np.sqrt(N_REPS)
kern_cov_se    = np.std([r['kern_cov_per_tau']    for r in results], axis=0, ddof=1) / np.sqrt(N_REPS)
nn_cov_se      = np.std([r['nn_cov_per_tau']      for r in results], axis=0, ddof=1) / np.sqrt(N_REPS)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.errorbar(QUANTILE_LEVELS, kern_losses_tau, yerr=kern_losses_se,
             marker='o', label=f'Local kernel ($d={N_LAGS}$)', capsize=3, lw=1.5)
ax1.errorbar(QUANTILE_LEVELS, nn_losses_tau, yerr=nn_losses_se,
             marker='s', label='ReLU network ($k=3$)', capsize=3, lw=1.5)
ax1.set_xlabel(r'Quantile level $\tau$')
ax1.set_ylabel('Pinball loss')
ax1.legend(frameon=False)
ax1.set_xticks(QUANTILE_LEVELS)

ax2.errorbar(QUANTILE_LEVELS, kern_cov_tau, yerr=kern_cov_se,
             marker='o', label=f'Local kernel ($d={N_LAGS}$)', capsize=3, lw=1.5)
ax2.errorbar(QUANTILE_LEVELS, nn_cov_tau, yerr=nn_cov_se,
             marker='s', label='ReLU network ($k=3$)', capsize=3, lw=1.5)
ax2.plot([0, 1], [0, 1], 'k--', lw=0.8, label='Ideal')
ax2.set_xlabel(r'Nominal level $\tau$')
ax2.set_ylabel('Empirical coverage')
ax2.legend(frameon=False)
ax2.set_xticks(QUANTILE_LEVELS)
ax2.set_xlim(0.05, 0.95)
ax2.set_ylim(0.05, 0.95)

plt.tight_layout()
fig_path = os.path.join(script_dir, '..', 'fig', 'simulation.png')
fig.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to {fig_path}")
plt.close()

print(f"\n=== COMPLETE ({time.time()-t0:.0f}s) ===")
