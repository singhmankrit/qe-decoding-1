import stim


# Problem 2A
def generate_repetition_code_circuit(d, p, q):
    circuit = stim.Circuit()
    total_qubits = 2 * d - 1

    # First initalise d data qubits and d-1 ancilla qubits
    for qubit in range(total_qubits):
        circuit.append("R", [qubit])

    for _ in range(d):
        # Errors for data qubits
        for qubit in range(d):
            circuit.append("X_ERROR", [qubit], p)

        # Errors for ancilla qubits
        for qubit in range(d, total_qubits):
            circuit.append("X_ERROR", [qubit], q)

        # Hadamard for ancilla qubits
        for qubit in range(d, total_qubits):
            circuit.append("H", [qubit])

        # 1st set of CZ
        for qubit in range(d, total_qubits):
            circuit.append("CZ", [qubit, qubit - d + 1])

        # 2nd set of CZ
        for qubit in range(d, total_qubits):
            circuit.append("CZ", [qubit, qubit - d])

        # Hadamard for ancilla qubits
        for qubit in range(d, total_qubits):
            circuit.append("H", [qubit])

        # Measure ancilla qubits
        for qubit in range(d, total_qubits):
            circuit.append("MZ", [qubit])

        # Reset ancilla qubits
        for qubit in range(d, total_qubits):
            circuit.append("R", [qubit])

    # Measure ancilla qubits
    for qubit in range(d):
        circuit.append("MZ", [qubit])

    return circuit
