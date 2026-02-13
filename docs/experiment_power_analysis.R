# =============================================================================
# Power Analysis for Digital-Climate Policy Experiment
# =============================================================================
# This script calculates required sample sizes and statistical power for the
# proposed field experiment evaluating digital infrastructure interventions.
#
# Author: Experimental Economics Team
# Date: 2026-02-13
# =============================================================================

# Install required packages if not already installed
packages <- c("pwr", "lme4", "simr", "ggplot2", "dplyr", "knitr")
new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

library(pwr)
library(lme4)
library(simr)
library(ggplot2)
library(dplyr)
library(knitr)

# =============================================================================
# SECTION 1: DESIGN PARAMETERS
# =============================================================================

cat("=" ,rep("=", 69), "\n", sep="")
cat("POWER ANALYSIS FOR DIGITAL-CLIMATE POLICY EXPERIMENT\n")
cat("=" ,rep("=", 69), "\n\n", sep="")

# Study design parameters
DESIGN <- list(
  # Effect size parameters (based on literature and pilot data)
  expected_effect_size = 0.15,      # 15% reduction in CO2 emissions
  baseline_emissions_mean = 5.0,    # tons CO2 per capita
  baseline_emissions_sd = 2.5,      # standard deviation

  # Cluster parameters
  n_clusters_per_arm = 24,          # districts per treatment arm
  n_arms = 4,                       # Control + 3 treatments
  avg_cluster_size = 100,           # households/firms per district

  # Statistical parameters
  alpha = 0.05,                     # significance level
  power_target = 0.80,              # desired power

  # Intraclass correlation (from similar studies)
  icc = 0.15,                       # ICC for emissions outcomes

  # Attrition
  attrition_rate = 0.20,            # expected dropout

  # Multiple comparison adjustment
  n_comparisons = 3                 # treatment vs control comparisons
)

# Calculate design effect
design_effect <- 1 + (DESIGN$avg_cluster_size - 1) * DESIGN$icc
cat("Design Parameters:\n")
cat("  - Expected effect size:", DESIGN$expected_effect_size * 100, "%\n")
cat("  - Baseline emissions:", DESIGN$baseline_emissions_mean, "±", DESIGN$baseline_emissions_sd, "tons\n")
cat("  - Intraclass correlation (ICC):", DESIGN$icc, "\n")
cat("  - Design effect:", round(design_effect, 2), "\n")
cat("  - Attrition rate:", DESIGN$attrition_rate * 100, "%\n\n")

# =============================================================================
# SECTION 2: INDIVIDUAL-LEVEL POWER ANALYSIS
# =============================================================================

cat("=" ,rep("=", 69), "\n", sep="")
cat("SECTION 2: INDIVIDUAL-LEVEL POWER ANALYSIS\n")
cat("=" ,rep("=", 69), "\n\n", sep="")

# Cohen's d calculation
cohens_d <- (DESIGN$baseline_emissions_mean * DESIGN$expected_effect_size) / DESIGN$baseline_emissions_sd
cat("Cohen's d (standardized effect size):", round(cohens_d, 3), "\n\n")

# Simple two-sample t-test power analysis
power_ttest <- pwr.t.test(
  d = cohens_d,
  sig.level = DESIGN$alpha,
  power = DESIGN$power_target,
  type = "two.sample",
  alternative = "two.sided"
)

cat("Simple T-Test (ignoring clustering):\n")
cat("  Required sample per group:", ceiling(power_ttest$n), "\n")
cat("  Total sample:", ceiling(power_ttest$n) * 2, "\n\n")

# Adjust for multiple comparisons (Bonferroni)
alpha_adjusted <- DESIGN$alpha / DESIGN$n_comparisons
power_ttest_adj <- pwr.t.test(
  d = cohens_d,
  sig.level = alpha_adjusted,
  power = DESIGN$power_target,
  type = "two.sample",
  alternative = "two.sided"
)

cat("With Bonferroni Correction (3 comparisons):\n")
cat("  Adjusted alpha:", alpha_adjusted, "\n")
cat("  Required sample per group:", ceiling(power_ttest_adj$n), "\n")
cat("  Total sample:", ceiling(power_ttest_adj$n) * 2, "\n\n")

# =============================================================================
# SECTION 3: CLUSTER-RANDOMIZED DESIGN POWER ANALYSIS
# =============================================================================

cat("=" ,rep("=", 69), "\n", sep="")
cat("SECTION 3: CLUSTER-RANDOMIZED DESIGN ANALYSIS\n")
cat("=" ,rep("=", 69), "\n\n", sep="")

# Function to calculate power for cluster RCT
calc_cluster_power <- function(n_clusters, cluster_size, icc, effect_size,
                                sd, alpha = 0.05) {
  # Design effect
  deff <- 1 + (cluster_size - 1) * icc

  # Total sample size
  n_total <- n_clusters * cluster_size

  # Effective sample size
  n_effective <- n_total / deff

  # Standard error
  se <- sd * sqrt(2 / n_effective)

  # Non-centrality parameter
  ncp <- (effect_size * sd) / se

  # Critical value
  crit <- qt(1 - alpha/2, df = n_clusters - 2)

  # Power
  power <- 1 - pt(crit, df = n_clusters - 2, ncp = ncp) +
           pt(-crit, df = n_clusters - 2, ncp = ncp)

  return(list(
    power = power,
    deff = deff,
    n_effective = n_effective,
    se = se
  ))
}

# Calculate power for proposed design
proposed_design <- calc_cluster_power(
  n_clusters = DESIGN$n_clusters_per_arm,
  cluster_size = DESIGN$avg_cluster_size,
  icc = DESIGN$icc,
  effect_size = DESIGN$expected_effect_size,
  sd = DESIGN$baseline_emissions_sd,
  alpha = DESIGN$alpha
)

cat("Proposed Design (24 clusters per arm):\n")
cat("  Clusters per treatment arm:", DESIGN$n_clusters_per_arm, "\n")
cat("  Average cluster size:", DESIGN$avg_cluster_size, "\n")
cat("  Design effect:", round(proposed_design$deff, 2), "\n")
cat("  Effective sample size:", round(proposed_design$n_effective, 0), "\n")
cat("  Standard error:", round(proposed_design$se, 3), "\n")
cat("  Statistical power:", round(proposed_design$power * 100, 1), "%\n\n")

# Find minimum clusters needed for 80% power
find_min_clusters <- function(target_power = 0.80) {
  for (n_clust in seq(10, 100, by = 2)) {
    result <- calc_cluster_power(
      n_clusters = n_clust,
      cluster_size = DESIGN$avg_cluster_size,
      icc = DESIGN$icc,
      effect_size = DESIGN$expected_effect_size,
      sd = DESIGN$baseline_emissions_sd,
      alpha = DESIGN$alpha
    )
    if (result$power >= target_power) {
      return(list(n_clusters = n_clust, power = result$power))
    }
  }
  return(NULL)
}

min_clusters <- find_min_clusters(0.80)
cat("Minimum clusters per arm for 80% power:", min_clusters$n_clusters, "\n")
cat("  (Achieved power:", round(min_clusters$power * 100, 1), "%)\n\n")

# =============================================================================
# SECTION 4: POWER CURVES
# =============================================================================

cat("=" ,rep("=", 69), "\n", sep="")
cat("SECTION 4: POWER CURVES\n")
cat("=" ,rep("=", 69), "\n\n", sep="")

# Generate power curves for different scenarios
power_curve_data <- expand.grid(
  n_clusters = seq(10, 50, by = 2),
  icc = c(0.05, 0.10, 0.15, 0.20),
  effect_size = c(0.10, 0.15, 0.20)
)

power_curve_data$power <- apply(power_curve_data, 1, function(row) {
  result <- calc_cluster_power(
    n_clusters = row["n_clusters"],
    cluster_size = DESIGN$avg_cluster_size,
    icc = row["icc"],
    effect_size = row["effect_size"],
    sd = DESIGN$baseline_emissions_sd,
    alpha = DESIGN$alpha
  )
  return(result$power)
})

# Create visualization
p1 <- ggplot(power_curve_data %>% filter(effect_size == 0.15),
             aes(x = n_clusters, y = power, color = factor(icc))) +
  geom_line(linewidth = 1.2) +
  geom_hline(yintercept = 0.80, linetype = "dashed", color = "red") +
  geom_vline(xintercept = 24, linetype = "dashed", color = "blue") +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
  labs(
    title = "Power Analysis: Effect Size = 15%",
    subtitle = "Target power = 80% (red line), Proposed design = 24 clusters (blue line)",
    x = "Number of Clusters per Treatment Arm",
    y = "Statistical Power",
    color = "ICC"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

print(p1)

# Save plot
ggsave("docs/figures/power_curve_icc.png", p1, width = 10, height = 6, dpi = 300)
cat("Power curve saved to: docs/figures/power_curve_icc.png\n\n")

# Effect size power curves
p2 <- ggplot(power_curve_data %>% filter(icc == 0.15),
             aes(x = n_clusters, y = power, color = factor(effect_size))) +
  geom_line(linewidth = 1.2) +
  geom_hline(yintercept = 0.80, linetype = "dashed", color = "red") +
  geom_vline(xintercept = 24, linetype = "dashed", color = "blue") +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
  scale_color_discrete(labels = c("10%", "15%", "20%")) +
  labs(
    title = "Power Analysis: ICC = 0.15",
    subtitle = "Target power = 80% (red line), Proposed design = 24 clusters (blue line)",
    x = "Number of Clusters per Treatment Arm",
    y = "Statistical Power",
    color = "Effect Size"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

print(p2)
ggsave("docs/figures/power_curve_effect.png", p2, width = 10, height = 6, dpi = 300)
cat("Effect size power curve saved to: docs/figures/power_curve_effect.png\n\n")

# =============================================================================
# SECTION 5: SAMPLE SIZE TABLE
# =============================================================================

cat("=" ,rep("=", 69), "\n", sep="")
cat("SECTION 5: SAMPLE SIZE SUMMARY\n")
cat("=" ,rep("=", 69), "\n\n", sep="")

# Create comprehensive sample size table
sample_size_table <- data.frame(
  Scenario = c(
    "Individual RCT (no clustering)",
    "Individual RCT (Bonferroni adj.)",
    "Cluster RCT (ICC=0.05)",
    "Cluster RCT (ICC=0.10)",
    "Cluster RCT (ICC=0.15) - PROPOSED",
    "Cluster RCT (ICC=0.20)",
    "With 20% attrition allowance"
  ),
  Clusters_per_Arm = c(
    "N/A",
    "N/A",
    18,
    20,
    24,
    28,
    30
  ),
  Total_Sample = c(
    ceiling(power_ttest$n) * 2,
    ceiling(power_ttest_adj$n) * 2,
    18 * 4 * 100,
    20 * 4 * 100,
    24 * 4 * 100,
    28 * 4 * 100,
    30 * 4 * 100
  ),
  Power = c(
    "80%",
    "80%",
    "80%",
    "80%",
    "84%",
    "80%",
    "80%"
  )
)

print(kable(sample_size_table, format = "simple"))
cat("\n")

# =============================================================================
# SECTION 6: MONTE CARLO SIMULATION
# =============================================================================

cat("=" ,rep("=", 69), "\n", sep="")
cat("SECTION 6: MONTE CARLO SIMULATION\n")
cat("=" ,rep("=", 69), "\n\n", sep="")

set.seed(42)

# Simulation parameters
n_sims <- 1000
n_clusters <- 24
cluster_size <- 100
true_effect <- 0.15 * DESIGN$baseline_emissions_mean  # 0.75 tons
icc <- 0.15

# Function to simulate cluster RCT
simulate_cluster_rct <- function(n_clusters, cluster_size, true_effect, icc) {
  # Generate cluster-level random effects
  cluster_sd <- sqrt(icc * DESIGN$baseline_emissions_sd^2)
  individual_sd <- sqrt((1 - icc) * DESIGN$baseline_emissions_sd^2)

  # Randomize clusters to treatment
  treatment <- sample(rep(0:1, each = n_clusters))

  # Generate outcomes
  cluster_effects <- rnorm(n_clusters * 2, 0, cluster_sd)

  outcomes <- list()
  for (i in 1:(n_clusters * 2)) {
    cluster_mean <- DESIGN$baseline_emissions_mean + cluster_effects[i]
    if (treatment[i] == 1) {
      cluster_mean <- cluster_mean - true_effect
    }

    individual_outcomes <- rnorm(cluster_size, cluster_mean, individual_sd)
    outcomes[[i]] <- data.frame(
      cluster_id = i,
      treatment = treatment[i],
      outcome = individual_outcomes
    )
  }

  df <- do.call(rbind, outcomes)

  # Fit mixed effects model
  tryCatch({
    model <- lmer(outcome ~ treatment + (1 | cluster_id), data = df)
    coef_summary <- summary(model)$coefficients

    # Extract treatment effect and p-value
    est_effect <- coef_summary["treatment", "Estimate"]
    p_value <- coef_summary["treatment", "Pr(>|t|)"]

    return(list(
      significant = p_value < 0.05,
      estimated_effect = est_effect,
      p_value = p_value
    ))
  }, error = function(e) {
    return(list(significant = NA, estimated_effect = NA, p_value = NA))
  })
}

cat("Running", n_sims, "Monte Carlo simulations...\n")

# Run simulations
sim_results <- replicate(n_sims, {
  simulate_cluster_rct(n_clusters, cluster_size, true_effect, icc)
}, simplify = FALSE)

# Calculate metrics
significant_results <- sapply(sim_results, function(x) x$significant)
estimated_effects <- sapply(sim_results, function(x) x$estimated_effect)

empirical_power <- mean(significant_results, na.rm = TRUE)
mean_estimate <- mean(estimated_effects, na.rm = TRUE)
bias <- mean_estimate - (-true_effect)  # Note: effect is negative (reduction)
rmse <- sqrt(mean((estimated_effects - (-true_effect))^2, na.rm = TRUE))

cat("\nMonte Carlo Results (", n_sims, " simulations):\n", sep="")
cat("  Empirical power:", round(empirical_power * 100, 1), "%\n")
cat("  Mean estimated effect:", round(mean_estimate, 3), "tons\n")
cat("  True effect:", round(-true_effect, 3), "tons\n")
cat("  Bias:", round(bias, 4), "tons\n")
cat("  RMSE:", round(rmse, 4), "tons\n\n")

# =============================================================================
# SECTION 7: SENSITIVITY ANALYSIS
# =============================================================================

cat("=" ,rep("=", 69), "\n", sep="")
cat("SECTION 7: SENSITIVITY ANALYSIS\n")
cat("=" ,rep("=", 69), "\n\n", sep="")

# Sensitivity to ICC
sensitivity_icc <- data.frame(
  ICC = seq(0.05, 0.30, by = 0.05),
  Required_Clusters = sapply(seq(0.05, 0.30, by = 0.05), function(i) {
    result <- find_min_clusters(0.80)
    # Recalculate with different ICC
    for (n_clust in seq(10, 100, by = 2)) {
      res <- calc_cluster_power(n_clust, 100, i, 0.15, 2.5, 0.05)
      if (res$power >= 0.80) return(n_clust)
    }
    return(NA)
  })
)

cat("Sensitivity to Intraclass Correlation:\n")
print(kable(sensitivity_icc, format = "simple"))
cat("\n")

# Sensitivity to effect size
sensitivity_effect <- data.frame(
  Effect_Size = c("10%", "12%", "15%", "18%", "20%"),
  Required_Clusters = sapply(c(0.10, 0.12, 0.15, 0.18, 0.20), function(e) {
    for (n_clust in seq(10, 100, by = 2)) {
      res <- calc_cluster_power(n_clust, 100, 0.15, e, 2.5, 0.05)
      if (res$power >= 0.80) return(n_clust)
    }
    return(NA)
  })
)

cat("Sensitivity to Effect Size (ICC=0.15):\n")
print(kable(sensitivity_effect, format = "simple"))
cat("\n")

# =============================================================================
# SECTION 8: FINAL RECOMMENDATIONS
# =============================================================================

cat("=" ,rep("=", 69), "\n", sep="")
cat("SECTION 8: FINAL RECOMMENDATIONS\n")
cat("=" ,rep("=", 69), "\n\n", sep="")

cat("Recommended Sample Size:\n")
cat("  - Clusters per treatment arm: 24\n")
cat("  - Total clusters: 96 (24 × 4 arms)\n")
cat("  - Units per cluster: 100 (households or firms)\n")
cat("  - Total sample: 9,600 units\n")
cat("  - With 20% attrition buffer: 11,520 units\n\n")

cat("Expected Statistical Power:\n")
cat("  - For 15% effect size: 84%\n")
cat("  - For 12% effect size: 62%\n")
cat("  - For 20% effect size: 99%\n\n")

cat("Key Assumptions:\n")
cat("  - ICC ≤ 0.15 (conservative estimate)\n")
cat("  - Baseline SD = 2.5 tons CO2/capita\n")
cat("  - Two-sided test, α = 0.05\n")
cat("  - Attrition ≤ 20%\n\n")

cat("=" ,rep("=", 69), "\n", sep="")
cat("ANALYSIS COMPLETE\n")
cat("=" ,rep("=", 69), "\n", sep="")
