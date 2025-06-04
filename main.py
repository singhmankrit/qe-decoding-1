from simulation_code import problem_1, problem_2

# Problem 1

# # Majority Voting
# threshold_mv, _, _ = problem_1.simulate_threshold_mv(n_runs=10**4)
# print(f"Estimated threshold (Majority Voting): {threshold_mv}")

# # Minimum Weight Perfect Matching
# threshold_mwpm, _, _ = problem_1.simulate_threshold_mwpm(n_runs=10**4)
# print(f"Estimated threshold (MWPM): {threshold_mwpm}")

# Problem 2

circuit = problem_2.generate_repetition_code_circuit(3, 0.1, 0.2)
print(circuit.diagram())

sampler = problem_2.measurement_sampler(circuit, n_runs=3)
# print(sampler)

all_syndromes, all_defects = problem_2.process_measurements(sampler, 3)
# print(all_syndromes)
# print(all_defects)
