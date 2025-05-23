from simulator import generate_repetition_code_circuit, measurement_sampler


circuit = generate_repetition_code_circuit(7, 0.1)

print(circuit.diagram())

print(measurement_sampler(circuit, 10))
