import stim
import pymatching
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


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
    defects = []
    for run in sampled_runs:
        n_rounds = d
        n_ancillas = d - 1

        # Syndromes
        measured_syndromes = np.array(run[: n_ancillas * n_ancillas]).reshape(
            n_ancillas, n_ancillas
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
                if t == 0:
                    defects_in_this_run.append(syndrome_in_this_run[0][i])
                else:
                    defect = syndrome_in_this_run[t][i] ^ syndrome_in_this_run[t - 1][i]
                    defects_in_this_run.append(defect)

        defects.append(defects_in_this_run)
    return defects


# Problem 2D
def build_decoding_graph(d, p, q):
    graph = pymatching.Matching()
    n_rounds = d
    n_ancillas = d - 1

    def detector_id(i, t):
        """Map stabilizer i at time t to a unique node ID."""
        return i + t * n_ancillas

    total_detectors = n_rounds * n_ancillas
    boundary_node = total_detectors

    # Spatial edges (within same round)
    for t in range(n_rounds):
        for i in range(n_ancillas - 1):
            a = detector_id(i, t)
            b = detector_id(i + 1, t)
            graph.add_edge(a, b, weight=-np.log(p))

    # Temporal edges (same stabilizer across rounds)
    for t in range(n_rounds):
        for i in range(n_ancillas):
            a = detector_id(i, t)
            b = detector_id(i, t + 1)
            graph.add_edge(a, b, weight=-np.log(q))

    # Boundary edges
    for t in range(n_rounds):
        for i in range(n_ancillas):
            node = detector_id(i, t)
            graph.add_boundary_edge(node, boundary_node, weight=-np.log(q))

    return graph


# Problem 2E
def simulate_threshold_mwpm(n_runs=10**6):
    distances = [3, 5, 7, 9]
    probabilities = np.linspace(0.05, 0.15, 20)
    results = {}

    for d in distances:
        pL_list = []
        print(f"\nSimulating for d = {d}")
        for p in tqdm(probabilities):
            circuit = generate_repetition_code_circuit(d, p, p)
            samples = measurement_sampler(circuit, n_runs=n_runs)
            defects = process_measurements(samples, d)
            graph = build_decoding_graph(d, p, p)
            corrections = graph.decode_batch(defects)
            logical_outcomes = np.sum((samples + corrections) % 2, axis=1) > (d - 1) / 2
            pL = sum(logical_outcomes.astype(int)) / n_runs
            pL_list.append(pL)
        results[d] = pL_list

    # Estimate threshold
    threshold_p = None
    for i in range(len(probabilities) - 1):
        pL_prev_dist = -1
        for d in distances:
            if i > 0 and pL_prev_dist > 0 and pL_prev_dist < results[d][i]:
                threshold_p = (probabilities[i - 1] + probabilities[i]) / 2
                break
            pL_prev_dist = results[d][i]
        if threshold_p is not None:
            break

    # Plotting
    plt.figure(figsize=(10, 6))
    for d in distances:
        plt.plot(probabilities, results[d], label=f"d = {d}")

    # Plot threshold marker
    plt.axvline(
        x=threshold_p,
        color="red",
        linestyle="--",
        label=f"Estimated threshold ≈ {threshold_p:.2f}",
    )
    plt.xlabel("Physical error rate p")
    plt.ylabel("Logical error rate pL")
    plt.title("MWPM: Repetition Code Logical vs Physical Error Rate")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")
    plt.savefig("images/problem_2/mwpm.png")

    return threshold_p, probabilities, results
