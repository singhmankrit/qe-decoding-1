from simulator import (
    generate_repetition_code_circuit,
    measurement_sampler,
    majority_vote,
    simulate_threshold,
)


circuit = generate_repetition_code_circuit(7, 0.1)

print(circuit.diagram())

sampler = measurement_sampler(circuit, 10)
print(sampler)

print(majority_vote(sampler, 7))

simulate_threshold(10**5)
