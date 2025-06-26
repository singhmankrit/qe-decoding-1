from simulation_code import problem_1, problem_2, problem_3

# ==================================================================
# Problem 1
# ==================================================================

# Majority Voting
threshold_mv = problem_1.simulate_threshold_mv(n_runs=10**6)
print(f"Estimated threshold (Majority Voting): {threshold_mv}")

# Minimum Weight Perfect Matching
threshold_mwpm = problem_1.simulate_threshold_mwpm(n_runs=10**6)
print(f"Estimated threshold (MWPM): {threshold_mwpm}")

# ==================================================================
# Problem 2
# ==================================================================

# Minimum Weight Perfect Matching for Multiple Ancilla Measurements
threshold_w_ancilla = problem_2.simulate_threshold(n_runs=10**6)
print(f"Estimated threshold (with Ancillas): {threshold_w_ancilla}")

# ==================================================================
# Problem 3
# ==================================================================

# Minimum Weight Perfect Matching for Biased Noise and Constant Weight Graph
threshold_w_bias = problem_3.simulate_threshold_bias(n_runs=10**6)
print(f"Estimated threshold (with biased noise, constant weight): {threshold_w_bias}")

# Minimum Weight Perfect Matching for Biased Noise and Correct Graph
threshold_w_correct_graph = problem_3.simulate_threshold_bias_correct_graph(
    n_runs=10**6
)
print(
    f"Estimated threshold (with biased noise, correct graph): {threshold_w_correct_graph}"
)
