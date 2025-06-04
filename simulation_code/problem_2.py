import stim
import numpy as np


# Problem 2A
def generate_repetition_code_circuit(d, p, q):
    """
    Generate a Stim circuit for a distance-d repetition code with separate
    phenomenological bit-flip noise on data and ancilla qubits.

    Args:
        d (int): Code distance (generally odd).
        p (float): Probability of a bit-flip (X error) on each data qubit.
        q (float): Probability of a bit-flip (X error) on each ancilla qubit.

    Returns:
        stim.Circuit: The corresponding Stim circuit with initialization,
                      error application, Hadamard gates, controlled-Z gates,
                      measurements, and resets for the repetition code process.
    """

    circuit = stim.Circuit()
    total_qubits = 2 * d - 1

    # First initalise d data qubits and d-1 ancilla qubits
    for qubit in range(total_qubits):
        circuit.append("R", [qubit])

    for _ in range(d - 1):
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

    # Measure data qubits
    for qubit in range(d):
        circuit.append("MZ", [qubit])

    return circuit


# Problem 2B
def measurement_sampler(circuit, n_runs, seed=42):
    """
    Samples measurement outcomes after compiling a measurement sampler.

    Args:
        circuit (stim.Circuit): The quantum circuit to be measured.
        n_runs (int): The number of times to sample the circuit.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        np.ndarray: An array of measurement results, with each row corresponding
                    to a run and each column to a measured qubit.
    """

    sampler = stim.CompiledMeasurementSampler(circuit, seed=seed)
    return np.array(sampler.sample(n_runs)).astype(int)


# Problem 2C
def process_measurements(sampled_runs, d):
    syndromes = []
    defects = []
    for run in sampled_runs:
        n_rounds = d - 1
        n_ancillas = d - 1

        # Syndromes
        measured_syndromes = np.array(run[: n_rounds * n_ancillas]).reshape(
            n_rounds, n_ancillas
        )
        final_data = np.array(run[-d:])
        projected_syndrome = np.array(
            [final_data[i] ^ final_data[i + 1] for i in range(d - 1)]
        )
        syndrome_in_this_run = np.vstack([measured_syndromes, projected_syndrome])

        # Compute defects = syndrome flip in time
        defects_in_this_run = []
        for t in range(n_rounds):
            for i in range(n_ancillas):
                if syndrome_in_this_run[t, i] != syndrome_in_this_run[t + 1, i]:
                    defects_in_this_run.append((t, i))  # (time, location)

        syndromes.append(syndrome_in_this_run)
        defects.append(defects_in_this_run)
    return syndromes, defects
