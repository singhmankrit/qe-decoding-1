from simulation_code import problem_1, problem_2

# Problem 1

# # Majority Voting
# threshold_mv, _, _ = problem_1.simulate_threshold_mv(n_runs=10**4)
# print(f"Estimated threshold (Majority Voting): {threshold_mv}")

# # Minimum Weight Perfect Matching
# threshold_mwpm, _, _ = problem_1.simulate_threshold_mwpm(n_runs=10**4)
# print(f"Estimated threshold (MWPM): {threshold_mwpm}")

# Problem 2

d = 3
p = 0.1
n_runs = 3
# circuit = problem_2.generate_repetition_code_circuit(d, p, p)
# print(circuit.diagram())
# samples = problem_2.measurement_sampler(circuit, n_runs)
# print(samples)
graph = problem_2.build_decoding_graph(d, p, p)
print(graph.num_fault_ids)
# defects = problem_2.process_measurements(samples, d)
# print(defects)
# corrections = graph.decode_batch(defects)
# print("Corrections:", corrections)

# problem_2.simulate_threshold_mwpm(n_runs=10**4)
