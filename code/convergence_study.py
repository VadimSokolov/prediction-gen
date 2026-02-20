"""
Convergence rate study for quantile neural networks.
Reports excess loss above oracle (true conditional quantile) to reveal rates.
"""
import numpy as np
import torch
import torch.nn as nn
import time

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cpu")

# ---------- DGP ----------
def generate_time_series(n_sequences, sequence_length=50, seed=None):
    rng = np.random.RandomState(seed)
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

def create_lagged_data(sequences, n_lags=5):
    histories, targets, summaries = [], [], []
    for X in sequences:
        n = len(X)
        for t in range(n_lags, n - 1):
            h = X[t - n_lags + 1 : t + 1]
            histories.append(h)
            targets.append(X[t + 1])
            summaries.append([np.mean(h), np.std(h, ddof=1), X[t]])
    return (np.array(histories, dtype=np.float32),
            np.array(targets, dtype=np.float32),
            np.array(summaries, dtype=np.float32))

# ---------- Oracle ----------
def oracle_quantile_predictions(X_full, tau_levels):
    """Compute true conditional quantiles from known DGP."""
    from scipy.stats import norm
    n = len(X_full)
    preds = np.empty((n, len(tau_levels)))
    for i in range(n):
        x_t = X_full[i, -1]      # last lag
        x_tm1 = X_full[i, -2]    # second-to-last lag
        mu = 0.5 * x_t - 0.3 * x_tm1
        sigma = np.sqrt(0.5 + 0.3 * x_t**2 + 0.2 * x_tm1**2)
        for q, tau in enumerate(tau_levels):
            preds[i, q] = mu + sigma * norm.ppf(tau)
    return preds

# ---------- Neural network ----------
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

def pinball_loss_torch(y_pred, y_true, tau):
    err = y_true - y_pred
    return torch.mean(torch.maximum(tau * err, (tau - 1) * err))

def train_qnn(X_train, y_train, X_test, tau, epochs=50, batch_size=256):
    X_tr = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_te = torch.tensor(X_test, dtype=torch.float32, device=device)
    tau_t = torch.tensor(tau, dtype=torch.float32, device=device)
    n = len(X_tr)
    n_val = int(0.2 * n)
    perm = torch.randperm(n)
    train_idx, val_idx = perm[n_val:], perm[:n_val]
    model = QuantileNet(X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val_loss = float('inf')
    patience = 0
    best_state = None
    for epoch in range(epochs):
        model.train()
        idx_perm = torch.randperm(len(train_idx))
        for start in range(0, len(train_idx), batch_size):
            batch = train_idx[idx_perm[start:start+batch_size]]
            optimizer.zero_grad()
            loss = pinball_loss_torch(model(X_tr[batch]), y_tr[batch], tau_t)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            vl = pinball_loss_torch(model(X_tr[val_idx]), y_tr[val_idx], tau_t).item()
        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(X_te).cpu().numpy()

def train_qnn_per_tau(X_train, y_train, X_test, tau_levels, epochs=50, batch_size=256):
    preds = np.empty((len(X_test), len(tau_levels)))
    for q, tau in enumerate(tau_levels):
        preds[:, q] = train_qnn(X_train, y_train, X_test, tau, epochs, batch_size)
    return preds

# ---------- Evaluation ----------
def avg_pinball_loss(y_test, pred_matrix, tau_levels):
    K = len(tau_levels)
    losses = np.empty(K)
    for q in range(K):
        err = y_test - pred_matrix[:, q]
        losses[q] = np.mean(np.where(err >= 0, tau_levels[q] * err,
                                     (tau_levels[q] - 1) * err))
    return np.mean(losses)

# ---------- Main ----------
N_LAGS = 5
QUANTILE_LEVELS = np.arange(0.1, 1.0, 0.1)

# Fixed test data
test_seq = generate_time_series(200, seed=1001)
X_te_full, y_te, X_te_summ = create_lagged_data(test_seq, N_LAGS)

# Oracle loss
oracle_pred = oracle_quantile_predictions(X_te_full, QUANTILE_LEVELS)
oracle_loss = avg_pinball_loss(y_te, oracle_pred, QUANTILE_LEVELS)
print(f"Oracle loss: {oracle_loss:.6f}", flush=True)

# Convergence study
sample_sizes = [50, 100, 200, 500, 1000, 2000, 5000]
n_reps_conv = 3

print("\n=== CONVERGENCE RATE STUDY (excess loss) ===", flush=True)

all_excess = {m: [] for m in sample_sizes}

for rep in range(1, n_reps_conv + 1):
    print(f"\n--- Rep {rep}/{n_reps_conv} ---", flush=True)
    for m_size in sample_sizes:
        t0 = time.time()
        small_seq = generate_time_series(m_size, seed=42 + rep * 1000 + m_size)
        _, y_sm, X_sm_summ = create_lagged_data(small_seq, N_LAGS)

        torch.manual_seed(42 + rep * 100 + m_size)
        nn_pred = train_qnn_per_tau(X_sm_summ, y_sm, X_te_summ,
                                    QUANTILE_LEVELS, epochs=50, batch_size=128)
        nn_loss = avg_pinball_loss(y_te, nn_pred, QUANTILE_LEVELS)
        excess = nn_loss - oracle_loss
        all_excess[m_size].append(excess)
        elapsed = time.time() - t0
        print(f"  m={m_size:5d}: loss={nn_loss:.6f}, excess={excess:.6f} ({elapsed:.0f}s)",
              flush=True)

# Summary
print("\n=== EXCESS LOSS SUMMARY ===")
mean_excess = []
for m_size in sample_sizes:
    me = np.mean(all_excess[m_size])
    se = np.std(all_excess[m_size], ddof=1) / np.sqrt(n_reps_conv)
    mean_excess.append(me)
    print(f"  m={m_size:5d}: excess={me:.6f} (SE {se:.6f})")

# Log-log regression for rate
log_m = np.log(np.array(sample_sizes, dtype=float))
log_excess = np.log(np.array(mean_excess))
# Filter out any negative or zero excess
valid = np.array(mean_excess) > 0
if valid.sum() >= 2:
    slope, intercept = np.polyfit(log_m[valid], log_excess[valid], 1)
    empirical_rate = -slope
    print(f"\nEmpirical convergence rate: m^(-{empirical_rate:.3f})")
else:
    empirical_rate = float('nan')
    print("\nCould not estimate rate (excess losses not positive)")

beta = 2; k = 3
theoretical_rate = 2 * beta / (2 * beta + k + 1)
print(f"Theoretical rate (b=2, k=3): m^(-{theoretical_rate:.3f})")

# Write to ledger
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, '..', 'results')
os.makedirs(results_dir, exist_ok=True)
ledger_path = os.path.join(results_dir, 'convergence.txt')
with open(ledger_path, 'w') as f:
    f.write("=== Convergence Rate Ledger ===\n")
    from datetime import datetime
    f.write(f"Date: {datetime.now()}\n")
    f.write(f"Oracle loss: {oracle_loss:.6f}\n\n")
    for m_size in sample_sizes:
        me = np.mean(all_excess[m_size])
        f.write(f"m={m_size}: excess={me:.6f}\n")
    if not np.isnan(empirical_rate):
        f.write(f"\nEmpirical rate: m^(-{empirical_rate:.3f})\n")
    f.write(f"Theoretical rate: m^(-{theoretical_rate:.3f})\n")

print(f"\nResults written to {ledger_path}")
