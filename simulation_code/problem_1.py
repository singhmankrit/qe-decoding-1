import stim
import pymatching
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# Problem 1A
def generate_repetition_code_circuit(d, p):
    """
    Generate a Stim circuit for a distance-d repetition code with incoming bit-flip noise.

    Args:
        d (int): Code distance (generally odd).
        p (float): Probability of a bit-flip (X error) on each qubit.

    Returns:
        stim.Circuit: The corresponding Stim circuit.
    """
    circuit = stim.Circuit()

    for q in range(d):
        circuit.append("R", [q])

    for q in range(d):
        circuit.append("X_ERROR", [q], p)

    for q in range(d):
        circuit.append("MZ", [q])

    return circuit


# Problem 1B
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


# Problem 1C
def majority_vote(sampled_runs, d):
    """
    Performs majority voting on measurement results to error-correct them.

    Args:
        sampled_runs (np.ndarray): 2D array where each row is a run and each
                                   column is a measured qubit outcome.
        d (int): Code distance used for error correction.

    Returns:
        list: A list of error-corrected outcomes, where each element is the
              result of applying majority voting to a corresponding run.
    """

    error_corrected = []
    for run in sampled_runs:
        if sum(run) <= (d - 1) / 2:
            error_corrected.append(0)
        else:
            error_corrected.append(1)
    return error_corrected


# Problem 1D
def simulate_threshold_mv(n_runs=10**6):
    """
    Simulates the logical error rate of the repetition code (using Majority Voting) for a variety of physical error rates and code distances and plots them.

    Args:
        n_runs (int): The number of runs to perform at each physical error rate.

    Returns:
        tuple: A tuple containing the estimated threshold error rate, an array of physical error rates, and a dictionary of logical error rates for each code distance.
    """

    distances = [3, 5, 7, 9]
    probabilities = np.linspace(0.01, 0.9, 20)
    results = {}

    for d in distances:
        pL_list = []
        print(f"\nSimulating for d = {d}")
        for p in tqdm(probabilities):
            circuit = generate_repetition_code_circuit(d, p)
            samples = measurement_sampler(circuit, n_runs=n_runs)
            logical_errors = majority_vote(samples, d)
            pL = sum(logical_errors) / n_runs
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
    plt.title("Majority Voting: Repetition Code Logical vs Physical Error Rate")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")
    plt.savefig("images/problem_1/majority_voting.png")

    return threshold_p, probabilities, results


# Problem 1E
def extract_syndromes(samples):
    """
    Convert data qubit measurement samples to syndrome defects.
    Each sample produces d-1 syndrome bits: s_i = m_i XOR m_{i+1}

    Args:
        samples (np.ndarray): shape (n_runs, d), with Z-basis measurements (0/1)

    Returns:
        np.ndarray: shape (n_runs, d - 1), with syndrome bits
    """
    return np.logical_xor(samples[:, :-1], samples[:, 1:]).astype(int)


# Problem 1F
def decoding_graph_mwpm(d, p, syndromes):
    """
    Creates a decoding graph for the repetition code using the minimum weight
    perfect matching (MWPM) algorithm.

    Args:
        d (int): The code distance.
        p (float): The physical error rate.
        syndromes (np.ndarray): Syndromes to be decoded.

    Returns:
        np.ndarray: Error corrections, with each element corresponding to a run
                    and each column to a qubit.
    """
    graph = pymatching.Matching()
    weight = -np.log(p)

    # Boundary edges: qubit 0 and qubit d-1
    graph.add_boundary_edge(0, weight=weight, fault_ids={0}, error_probability=p)
    graph.add_boundary_edge(
        d - 2, weight=weight, fault_ids={d - 1}, error_probability=p
    )

    # Internal edges: qubit 1 to d-2
    for i in range(1, d - 1):
        graph.add_edge(i - 1, i, weight=weight, fault_ids={i}, error_probability=p)

    return graph.decode_batch(syndromes)


# Problem 1G
def simulate_threshold_mwpm(n_runs=10**6):
    """
    Simulates the logical error rate of the repetition code using the minimum
    weight perfect matching (MWPM) algorithm for various physical error rates
    and code distances, and plots the results.

    Args:
        n_runs (int): The number of runs to perform at each physical error rate.

    Returns:
        threshold: The estimated threshold error rate.
    """

    distances = [3, 5, 7, 9]
    probabilities = np.linspace(0.01, 0.9, 20)
    results = {}

    for d in distances:
        pL_list = []
        print(f"\nSimulating for d = {d}")
        for p in tqdm(probabilities):
            circuit = generate_repetition_code_circuit(d, p)
            samples = measurement_sampler(circuit, n_runs=n_runs)
            syndromes = extract_syndromes(samples)
            corrections = decoding_graph_mwpm(d, p, syndromes)
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
    plt.savefig("images/problem_1/mwpm.png")

    return threshold_p
