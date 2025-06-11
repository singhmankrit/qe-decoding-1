from simulation_code import problem_1, problem_2
import matplotlib.pyplot as plt

# Problem 1

# # Majority Voting
# threshold_mv, _, _ = problem_1.simulate_threshold_mv(n_runs=10**4)
# print(f"Estimated threshold (Majority Voting): {threshold_mv}")

# # Minimum Weight Perfect Matching
# threshold_mwpm = problem_1.simulate_threshold_mwpm(n_runs=10**4)
# print(f"Estimated threshold (MWPM): {threshold_mwpm}")

# # Problem 2

# graph = problem_2.build_decoding_graph(3, 0.1, 0.1)
# plt.figure()
# graph.draw()
# plt.savefig("graph.png")

# Minimum Weight Perfect Matching for Multiple Ancilla Measurements
threshold_w_ancilla = problem_2.simulate_threshold_mwpm(n_runs=10**4)
print(f"Estimated threshold (with Ancillas): {threshold_w_ancilla}")

# Problem 3

# Minimum Weight Perfect Matching for Multiple Ancilla Measurements, Biased Noise
# threshold_w_ancilla = problem_2.simulate_threshold_mwpm_bias(n_runs=10**5)
# print(f"Estimated threshold (with Ancillas, biased noise): {threshold_w_ancilla}")
