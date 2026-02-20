################################################################################
# Predictive Quantile Regression: Local Kernel vs Global Neural Network
# Comparing efficiency bounds for p(X_{n+1} | X_1, ..., X_n)
# Based on Schmidt-Hieber & Zamolodtchikov (2024) and Polson-Sokolov (2023)
################################################################################

library(keras)
library(ggplot2)
library(dplyr)

set.seed(42)

################################################################################
# 1. DATA GENERATING PROCESS
################################################################################

# AR(2) with smooth heteroscedastic errors (beta = 2 Holder smooth)
# X_t = 0.5*X_{t-1} - 0.3*X_{t-2} + epsilon_t
# epsilon_t ~ N(0, sigma_t^2), sigma_t = 0.5 + 0.3*X_{t-1}^2 + 0.2*X_{t-2}^2
# Note: sigma_t is the standard deviation; all functions are C^2, so beta = 2.

generate_time_series <- function(n_sequences, sequence_length = 50, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  data_list <- vector("list", n_sequences)
  for (i in seq_len(n_sequences)) {
    X <- numeric(sequence_length)
    X[1] <- rnorm(1)
    X[2] <- rnorm(1)
    for (t in 3:sequence_length) {
      mu_t <- 0.5 * X[t-1] - 0.3 * X[t-2]
      sigma_t <- sqrt(0.5 + 0.3 * X[t-1]^2 + 0.2 * X[t-2]^2)
      X[t] <- mu_t + rnorm(1, sd = sigma_t)
    }
    data_list[[i]] <- X
  }
  return(data_list)
}

################################################################################
# 2. PREPARE DATA
################################################################################

create_lagged_data <- function(sequences, n_lags = 5) {
  records <- list()
  for (i in seq_along(sequences)) {
    X <- sequences[[i]]
    n <- length(X)
    for (t in (n_lags + 1):(n - 1)) {
      history <- X[(t - n_lags + 1):t]
      records[[length(records) + 1]] <- list(
        history = history,
        target  = X[t + 1],
        summary = c(mean(history), sd(history), X[t])
      )
    }
  }
  return(records)
}

n_lags <- 5
quantile_levels <- seq(0.1, 0.9, by = 0.1)

################################################################################
# 3. KERNEL QUANTILE REGRESSION (Nadaraya-Watson)
################################################################################

weighted_quantile <- function(x, w, tau) {
  ord <- order(x)
  x_sorted <- x[ord]
  w_sorted <- w[ord]
  idx <- which(cumsum(w_sorted) >= tau)[1]
  return(x_sorted[idx])
}

kernel_quantile_predict <- function(X_train, y_train, X_test, tau) {
  # Kernel operates on full d-dimensional input with Silverman's rule
  X_tr <- scale(X_train)
  X_te <- scale(X_test, center = attr(X_tr, "scaled:center"),
                scale = attr(X_tr, "scaled:scale"))

  d <- ncol(X_tr)
  n <- nrow(X_tr)
  h <- (4 / (d + 2))^(1 / (d + 4)) * n^(-1 / (d + 4))
  preds <- numeric(nrow(X_te))
  for (i in seq_len(nrow(X_te))) {
    d2 <- rowSums((X_tr - matrix(X_te[i,], nrow(X_tr), ncol(X_tr), byrow = TRUE))^2)
    w <- exp(-d2 / (2 * h^2))
    w <- w / sum(w)
    preds[i] <- weighted_quantile(y_train, w, tau)
  }
  return(preds)
}

################################################################################
# 4. NEURAL NETWORK QUANTILE REGRESSION (separate model per tau)
################################################################################

quantile_loss <- function(tau) {
  function(y_true, y_pred) {
    err <- y_true - y_pred
    k_mean(k_maximum(tau * err, (tau - 1) * err))
  }
}

build_qnn <- function(input_dim, hidden = c(128, 128, 64)) {
  keras_model_sequential() %>%
    layer_dense(hidden[1], activation = "relu", input_shape = input_dim) %>%
    layer_dropout(0.2) %>%
    layer_dense(hidden[2], activation = "relu") %>%
    layer_dropout(0.2) %>%
    layer_dense(hidden[3], activation = "relu") %>%
    layer_dense(1)
}

train_qnn_per_tau <- function(X_train, y_train, X_test, tau_levels,
                              epochs = 50, batch_size = 256) {
  preds <- matrix(NA, nrow(X_test), length(tau_levels))
  for (q in seq_along(tau_levels)) {
    tau <- tau_levels[q]
    model <- build_qnn(input_dim = ncol(X_train))
    model %>% compile(
      optimizer = optimizer_adam(learning_rate = 0.001),
      loss = quantile_loss(tau)
    )
    model %>% fit(X_train, y_train,
                  epochs = epochs, batch_size = batch_size,
                  validation_split = 0.2, verbose = 0)
    preds[, q] <- predict(model, X_test, verbose = 0)
  }
  return(preds)
}

################################################################################
# 5. EVALUATION HELPERS
################################################################################

pinball_loss <- function(y, yhat, tau) {
  err <- y - yhat
  mean(ifelse(err >= 0, tau * err, (tau - 1) * err))
}

evaluate_predictions <- function(y_test, pred_matrix, tau_levels) {
  K <- length(tau_levels)
  losses <- numeric(K)
  coverage <- numeric(K)
  for (q in seq_len(K)) {
    losses[q]   <- pinball_loss(y_test, pred_matrix[, q], tau_levels[q])
    coverage[q] <- mean(y_test <= pred_matrix[, q])
  }
  list(avg_loss     = mean(losses),
       coverage_mae = mean(abs(coverage - tau_levels)),
       losses       = losses,
       coverage     = coverage)
}

################################################################################
# 6. MAIN EXPERIMENT WITH REPLICATIONS
################################################################################

n_reps   <- 5
n_train  <- 5000
n_test   <- 1000

results <- data.frame()

for (rep in 1:n_reps) {
  cat(sprintf("\n=== Replication %d/%d ===\n", rep, n_reps))
  seed_train <- 42 + rep
  seed_test  <- 1000 + rep

  train_seq <- generate_time_series(n_train, seed = seed_train)
  test_seq  <- generate_time_series(n_test,  seed = seed_test)

  train_data <- create_lagged_data(train_seq, n_lags)
  test_data  <- create_lagged_data(test_seq,  n_lags)

  X_train_full <- t(sapply(train_data, function(x) x$history))
  y_train      <- sapply(train_data, function(x) x$target)
  X_test_full  <- t(sapply(test_data,  function(x) x$history))
  y_test       <- sapply(test_data,  function(x) x$target)
  X_train_summ <- t(sapply(train_data, function(x) x$summary))
  X_test_summ  <- t(sapply(test_data,  function(x) x$summary))

  # Kernel predictions
  cat("  Kernel quantile regression...\n")
  kern_pred <- matrix(NA, nrow(X_test_full), length(quantile_levels))
  for (q in seq_along(quantile_levels))
    kern_pred[, q] <- kernel_quantile_predict(
      X_train_full, y_train, X_test_full, quantile_levels[q])
  kern_eval <- evaluate_predictions(y_test, kern_pred, quantile_levels)

  # Neural network predictions (separate model per tau)
  cat("  Neural network quantile regression...\n")
  nn_pred <- train_qnn_per_tau(X_train_summ, y_train, X_test_summ,
                               quantile_levels, epochs = 50)
  nn_eval <- evaluate_predictions(y_test, nn_pred, quantile_levels)

  results <- rbind(results, data.frame(
    rep           = rep,
    method        = c("Kernel", "Neural Network"),
    avg_loss      = c(kern_eval$avg_loss, nn_eval$avg_loss),
    coverage_mae  = c(kern_eval$coverage_mae, nn_eval$coverage_mae)
  ))

  cat(sprintf("  Kernel: loss=%.4f  cov_mae=%.4f\n",
              kern_eval$avg_loss, kern_eval$coverage_mae))
  cat(sprintf("  NN:     loss=%.4f  cov_mae=%.4f\n",
              nn_eval$avg_loss, nn_eval$coverage_mae))
}

################################################################################
# 7. SUMMARY WITH STANDARD ERRORS
################################################################################

cat("\n=== SUMMARY ACROSS REPLICATIONS ===\n\n")

summary_df <- results %>%
  group_by(method) %>%
  summarise(
    mean_loss     = mean(avg_loss),
    se_loss       = sd(avg_loss) / sqrt(n()),
    mean_cov_mae  = mean(coverage_mae),
    se_cov_mae    = sd(coverage_mae) / sqrt(n()),
    .groups = "drop"
  )

print(summary_df)

kern_mean <- summary_df$mean_loss[summary_df$method == "Kernel"]
nn_mean   <- summary_df$mean_loss[summary_df$method == "Neural Network"]
cat(sprintf("\nRelative improvement: %.1f%%\n",
            100 * (kern_mean - nn_mean) / kern_mean))

################################################################################
# 8. CONVERGENCE RATE STUDY
################################################################################

cat("\n=== CONVERGENCE RATE STUDY ===\n")

sample_sizes <- c(500, 1000, 2000, 5000, 10000)
conv_results <- data.frame()

for (m_size in sample_sizes) {
  cat(sprintf("  m = %d\n", m_size))

  small_seq  <- generate_time_series(m_size, seed = 42)
  small_data <- create_lagged_data(small_seq, n_lags)

  X_sm <- t(sapply(small_data, function(x) x$summary))
  y_sm <- sapply(small_data, function(x) x$target)

  # Use first replication's test data
  test_seq_conv  <- generate_time_series(n_test, seed = 1001)
  test_data_conv <- create_lagged_data(test_seq_conv, n_lags)
  X_te <- t(sapply(test_data_conv, function(x) x$summary))
  y_te <- sapply(test_data_conv, function(x) x$target)

  nn_pred_conv <- train_qnn_per_tau(X_sm, y_sm, X_te,
                                    quantile_levels, epochs = 30, batch_size = 128)
  nn_eval_conv <- evaluate_predictions(y_te, nn_pred_conv, quantile_levels)

  conv_results <- rbind(conv_results,
    data.frame(m = m_size, avg_loss = nn_eval_conv$avg_loss))
}

# Estimate empirical rate via log-log regression
fit <- lm(log(avg_loss) ~ log(m), data = conv_results)
empirical_rate <- -coef(fit)[2]
cat(sprintf("\nEmpirical convergence rate: m^(-%.3f)\n", empirical_rate))

beta <- 2; k <- 3
theoretical_rate <- 2 * beta / (2 * beta + k + 1)
cat(sprintf("Theoretical rate (beta=%d, k=%d): m^(-%.3f)\n",
            beta, k, theoretical_rate))

################################################################################
# 9. WRITE RESULTS TO LEDGER
################################################################################

sink("results/numbers.txt")
cat("=== Number Provenance Ledger ===\n")
cat(sprintf("Script: simulation_kernel_vs_nn.R\n"))
cat(sprintf("Date: %s\n\n", Sys.time()))
cat(sprintf("Table 1:\n"))
cat(sprintf("  Kernel avg quantile loss: %.4f (SE: %.4f)\n",
            summary_df$mean_loss[1], summary_df$se_loss[1]))
cat(sprintf("  NN avg quantile loss:     %.4f (SE: %.4f)\n",
            summary_df$mean_loss[2], summary_df$se_loss[2]))
cat(sprintf("  Kernel coverage MAE:      %.4f (SE: %.4f)\n",
            summary_df$mean_cov_mae[1], summary_df$se_cov_mae[1]))
cat(sprintf("  NN coverage MAE:          %.4f (SE: %.4f)\n",
            summary_df$mean_cov_mae[2], summary_df$se_cov_mae[2]))
cat(sprintf("  Relative improvement:     %.1f%%\n",
            100 * (kern_mean - nn_mean) / kern_mean))
cat(sprintf("\nConvergence rates:\n"))
cat(sprintf("  Empirical NN rate:   m^(-%.3f)\n", empirical_rate))
cat(sprintf("  Theoretical (b=2,k=3): m^(-%.3f)\n", theoretical_rate))
sink()

cat("\n=== COMPLETE ===\n")
cat("Results written to results/numbers.txt\n")
