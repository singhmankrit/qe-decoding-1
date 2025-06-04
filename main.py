from simulator import (
    generate_repetition_code_circuit,
    measurement_sampler,
    majority_vote,
    simulate_threshold,
    extract_syndromes,
)


circuit = generate_repetition_code_circuit(7, 0.1)
sampled_runs = measurement_sampler(circuit, 10)
# print(majority_vote(sampled_runs, 7))
# threshold, _, _ = simulate_threshold()
# print(f"Estimated threshold: {threshold}")


print(sampled_runs)
print("=========================")
print(extract_syndromes(sampled_runs))
