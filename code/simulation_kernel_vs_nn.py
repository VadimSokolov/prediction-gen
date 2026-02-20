"""
Predictive Quantile Regression: Local Kernel vs Global Neural Network
Comparing efficiency bounds for p(X_{n+1} | X_1, ..., X_n)
Based on Schmidt-Hieber & Zamolodtchikov (2024) and Polson-Sokolov (2023)

Python/PyTorch rewrite of simulation_kernel_vs_nn.R
"""

import numpy as np
import torch
import torch.nn as nn
import os, sys, time

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cpu")

###############################################################################
# 1. DATA GENERATING PROCESS
###############################################################################

# AR(2) with smooth heteroscedastic errors (beta = 2 Holder smooth)
# X_t = 0.5*X_{t-1} - 0.3*X_{t-2} + epsilon_t
# epsilon_t ~ N(0, sigma_t^2), sigma_t = sqrt(0.5 + 0.3*X_{t-1}^2 + 0.2*X_{t-2}^2)
# All functions are C^2, so beta = 2.

def generate_time_series(n_sequences, sequence_length=50, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    data = []
    for _ in range(n_sequences):
        X = np.zeros(sequence_length)
        X[0] = rng.randn()
        X[1] = rng.randn()
        for t in range(2, sequence_length):
            mu_t = 0.5 * X[t-1] - 0.3 * X[t-2]
            sigma_t = np.sqrt(0.5 + 0.3 * X[t-1]**2 + 0.2 * X[t-2]**2)
            X[t] = mu_t + rng.randn() * sigma_t
        data.append(X)
    return data

###############################################################################
# 2. PREPARE DATA
###############################################################################

def create_lagged_data(sequences, n_lags=5):
    histories = []
    targets = []
    summaries = []
    for X in sequences:
        n = len(X)
        for t in range(n_lags, n - 1):
            history = X[t - n_lags + 1 : t + 1]
            histories.append(history)
            targets.append(X[t + 1])
            summaries.append([np.mean(history), np.std(history, ddof=1), X[t]])
    return (np.array(histories, dtype=np.float32),
            np.array(targets, dtype=np.float32),
            np.array(summaries, dtype=np.float32))

N_LAGS = 5
QUANTILE_LEVELS = np.arange(0.1, 1.0, 0.1)  # 0.1, 0.2, ..., 0.9

###############################################################################
# 3. KERNEL QUANTILE REGRESSION (Nadaraya-Watson)
###############################################################################

def kernel_quantile_predict_all(X_train, y_train, X_test, tau_levels,
                                chunk_size=500):
    """Vectorized Nadaraya-Watson kernel quantile on full d-dimensional input.

    Uses Silverman's multivariate bandwidth rule: h = n^{-1/(d+4)} * sigma_avg.
    Predicts all quantile levels simultaneously by precomputing the sort
    order of y_train and using chunked distance computation.
    Returns: pred_matrix of shape (n_test, len(tau_levels))
    """
    # Standardize features for isotropic kernel
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1.0
    X_tr = (X_train - mu) / sigma
    X_te = (X_test - mu) / sigma

    d = X_tr.shape[1]
    n = len(X_tr)
    # Silverman's rule for d dimensions: h = (4/(d+2))^{1/(d+4)} * n^{-1/(d+4)}
    h = (4.0 / (d + 2)) ** (1.0 / (d + 4)) * n ** (-1.0 / (d + 4))
    inv_2h2 = 1.0 / (2 * h ** 2)

    # Precompute sort order of y_train (same for all test points)
    sort_idx = np.argsort(y_train)
    y_sorted = y_train[sort_idx]

    n_test = len(X_te)
    n_tau = len(tau_levels)
    preds = np.empty((n_test, n_tau))

    # Precompute ||X_tr||^2
    X_tr_sq = np.sum(X_tr ** 2, axis=1)  # (n_train,)

    for start in range(0, n_test, chunk_size):
        end = min(start + chunk_size, n_test)
        chunk = X_te[start:end]  # (chunk_size, 2)

        # Squared Euclidean distances via ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        chunk_sq = np.sum(chunk ** 2, axis=1, keepdims=True)  # (cs, 1)
        d2 = chunk_sq + X_tr_sq[None, :] - 2.0 * chunk @ X_tr.T  # (cs, n_train)
        np.maximum(d2, 0, out=d2)  # numerical safety
        w = np.exp(-d2 * inv_2h2)
        w_sum = w.sum(axis=1, keepdims=True)
        w_sum[w_sum == 0] = 1.0
        w /= w_sum

        # Reorder weights by y sort order and cumsum
        w_sorted = w[:, sort_idx]  # (chunk_size, n_train)
        cumw = np.cumsum(w_sorted, axis=1)  # (chunk_size, n_train)

        for q, tau in enumerate(tau_levels):
            # First index where cumulative weight >= tau
            idx = np.argmax(cumw >= tau, axis=1)
            preds[start:end, q] = y_sorted[idx]

    return preds

###############################################################################
# 4. NEURAL NETWORK QUANTILE REGRESSION (separate model per tau)
###############################################################################

class QuantileNet(nn.Module):
    def __init__(self, input_dim, hidden=(128, 128, 64)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

def pinball_loss(y_pred, y_true, tau):
    err = y_true - y_pred
    return torch.mean(torch.maximum(tau * err, (tau - 1) * err))

def train_qnn(X_train, y_train, X_test, tau, epochs=50, batch_size=256, lr=1e-3):
    """Train a single quantile network for a given tau level."""
    X_tr = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_te = torch.tensor(X_test, dtype=torch.float32, device=device)
    tau_t = torch.tensor(tau, dtype=torch.float32, device=device)

    n = len(X_tr)
    n_val = int(0.2 * n)
    perm = torch.randperm(n)
    train_idx, val_idx = perm[n_val:], perm[:n_val]

    model = QuantileNet(X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        idx_perm = torch.randperm(len(train_idx))
        for start in range(0, len(train_idx), batch_size):
            batch = train_idx[idx_perm[start:start+batch_size]]
            optimizer.zero_grad()
            pred = model(X_tr[batch])
            loss = pinball_loss(pred, y_tr[batch], tau_t)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_tr[val_idx])
            val_loss = pinball_loss(val_pred, y_tr[val_idx], tau_t).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(X_te).cpu().numpy()
    return preds

def train_qnn_per_tau(X_train, y_train, X_test, tau_levels, epochs=50, batch_size=256):
    """Train separate quantile networks for each tau level."""
    preds = np.empty((len(X_test), len(tau_levels)))
    for q, tau in enumerate(tau_levels):
        preds[:, q] = train_qnn(X_train, y_train, X_test, tau,
                                epochs=epochs, batch_size=batch_size)
    return preds

###############################################################################
# 5. EVALUATION HELPERS
###############################################################################

def evaluate_predictions(y_test, pred_matrix, tau_levels):
    K = len(tau_levels)
    losses = np.empty(K)
    coverage = np.empty(K)
    for q in range(K):
        err = y_test - pred_matrix[:, q]
        losses[q] = np.mean(np.where(err >= 0, tau_levels[q] * err,
                                     (tau_levels[q] - 1) * err))
        coverage[q] = np.mean(y_test <= pred_matrix[:, q])
    return {
        'avg_loss': np.mean(losses),
        'coverage_mae': np.mean(np.abs(coverage - tau_levels)),
        'losses': losses,
        'coverage': coverage
    }

###############################################################################
# 6. MAIN EXPERIMENT WITH REPLICATIONS
###############################################################################

N_REPS = 5
N_TRAIN = 5000
N_TEST = 200   # 200 sequences × 44 = 8,800 test points (sufficient for evaluation)

results = []

t0 = time.time()
for rep in range(1, N_REPS + 1):
    print(f"\n=== Replication {rep}/{N_REPS} ===", flush=True)
    seed_train = 42 + rep
    seed_test = 1000 + rep

    train_seq = generate_time_series(N_TRAIN, seed=seed_train)
    test_seq = generate_time_series(N_TEST, seed=seed_test)

    X_train_full, y_train, X_train_summ = create_lagged_data(train_seq, N_LAGS)
    X_test_full, y_test, X_test_summ = create_lagged_data(test_seq, N_LAGS)

    # Kernel predictions (vectorized over all quantile levels)
    print("  Kernel quantile regression...", flush=True)
    kern_pred = kernel_quantile_predict_all(X_train_full, y_train,
                                            X_test_full, QUANTILE_LEVELS)
    kern_eval = evaluate_predictions(y_test, kern_pred, QUANTILE_LEVELS)

    # Neural network predictions (separate model per tau)
    print("  Neural network quantile regression...", flush=True)
    torch.manual_seed(42 + rep * 100)
    nn_pred = train_qnn_per_tau(X_train_summ, y_train, X_test_summ,
                                QUANTILE_LEVELS, epochs=50)
    nn_eval = evaluate_predictions(y_test, nn_pred, QUANTILE_LEVELS)

    results.append({
        'rep': rep,
        'kern_loss': kern_eval['avg_loss'],
        'kern_cov': kern_eval['coverage_mae'],
        'nn_loss': nn_eval['avg_loss'],
        'nn_cov': nn_eval['coverage_mae']
    })

    elapsed = time.time() - t0
    print(f"  Kernel: loss={kern_eval['avg_loss']:.4f}  cov_mae={kern_eval['coverage_mae']:.4f}")
    print(f"  NN:     loss={nn_eval['avg_loss']:.4f}  cov_mae={nn_eval['coverage_mae']:.4f}")
    print(f"  [{elapsed:.0f}s elapsed]", flush=True)

###############################################################################
# 7. SUMMARY WITH STANDARD ERRORS
###############################################################################

print("\n=== SUMMARY ACROSS REPLICATIONS ===\n")

kern_losses = [r['kern_loss'] for r in results]
kern_covs = [r['kern_cov'] for r in results]
nn_losses = [r['nn_loss'] for r in results]
nn_covs = [r['nn_cov'] for r in results]

kern_mean_loss = np.mean(kern_losses)
kern_se_loss = np.std(kern_losses, ddof=1) / np.sqrt(N_REPS)
kern_mean_cov = np.mean(kern_covs)
kern_se_cov = np.std(kern_covs, ddof=1) / np.sqrt(N_REPS)

nn_mean_loss = np.mean(nn_losses)
nn_se_loss = np.std(nn_losses, ddof=1) / np.sqrt(N_REPS)
nn_mean_cov = np.mean(nn_covs)
nn_se_cov = np.std(nn_covs, ddof=1) / np.sqrt(N_REPS)

print(f"  Kernel:  loss = {kern_mean_loss:.4f} (SE {kern_se_loss:.4f}), "
      f"cov MAE = {kern_mean_cov:.4f} (SE {kern_se_cov:.4f})")
print(f"  NN:      loss = {nn_mean_loss:.4f} (SE {nn_se_loss:.4f}), "
      f"cov MAE = {nn_mean_cov:.4f} (SE {nn_se_cov:.4f})")
print(f"\n  Relative improvement: {100*(kern_mean_loss - nn_mean_loss)/kern_mean_loss:.1f}%")

###############################################################################
# 8. CONVERGENCE RATE STUDY
###############################################################################

print("\n=== CONVERGENCE RATE STUDY ===")

sample_sizes = [500, 1000, 2000, 5000, 10000]
conv_losses = []

test_seq_conv = generate_time_series(200, seed=1001)
X_te_conv, y_te_conv, X_te_summ_conv = create_lagged_data(test_seq_conv, N_LAGS)

for m_size in sample_sizes:
    print(f"  m = {m_size}")
    small_seq = generate_time_series(m_size, seed=42)
    _, y_sm, X_sm_summ = create_lagged_data(small_seq, N_LAGS)

    torch.manual_seed(42 + m_size)
    nn_pred_conv = train_qnn_per_tau(X_sm_summ, y_sm, X_te_summ_conv,
                                     QUANTILE_LEVELS, epochs=30, batch_size=128)
    eval_conv = evaluate_predictions(y_te_conv, nn_pred_conv, QUANTILE_LEVELS)
    conv_losses.append(eval_conv['avg_loss'])
    print(f"    loss = {eval_conv['avg_loss']:.4f}")

# Empirical rate via log-log regression
log_m = np.log(np.array(sample_sizes, dtype=float))
log_loss = np.log(np.array(conv_losses))
slope, intercept = np.polyfit(log_m, log_loss, 1)
empirical_rate = -slope

beta = 2; k = 3
theoretical_rate = 2 * beta / (2 * beta + k + 1)

print(f"\n  Empirical convergence rate:    m^(-{empirical_rate:.3f})")
print(f"  Theoretical rate (b=2, k=3):  m^(-{theoretical_rate:.3f})")

###############################################################################
# 9. WRITE RESULTS TO LEDGER
###############################################################################

script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, '..', 'results')
os.makedirs(results_dir, exist_ok=True)
ledger_path = os.path.join(results_dir, 'numbers.txt')

with open(ledger_path, 'w') as f:
    f.write("=== Number Provenance Ledger ===\n")
    f.write(f"Script: simulation_kernel_vs_nn.py\n")
    from datetime import datetime
    f.write(f"Date: {datetime.now()}\n\n")
    f.write("Table 1:\n")
    f.write(f"  Kernel avg quantile loss: {kern_mean_loss:.4f} (SE: {kern_se_loss:.4f})\n")
    f.write(f"  NN avg quantile loss:     {nn_mean_loss:.4f} (SE: {nn_se_loss:.4f})\n")
    f.write(f"  Kernel coverage MAE:      {kern_mean_cov:.4f} (SE: {kern_se_cov:.4f})\n")
    f.write(f"  NN coverage MAE:          {nn_mean_cov:.4f} (SE: {nn_se_cov:.4f})\n")
    f.write(f"  Relative improvement:     {100*(kern_mean_loss - nn_mean_loss)/kern_mean_loss:.1f}%\n")
    f.write(f"\nConvergence rates:\n")
    f.write(f"  Empirical NN rate:       m^(-{empirical_rate:.3f})\n")
    f.write(f"  Theoretical (b=2,k=3):   m^(-{theoretical_rate:.3f})\n")
    for i, m_size in enumerate(sample_sizes):
        f.write(f"  m={m_size}: loss={conv_losses[i]:.4f}\n")

print(f"\n=== COMPLETE ===")
print(f"Results written to {ledger_path}")
