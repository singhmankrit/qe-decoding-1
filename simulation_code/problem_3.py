import pymatching
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from . import problem_2


# Problem 3A
def build_decoding_graph_const_weight(d, p):
    """
    Builds a decoding graph for a distance-d surface code with separate
    phenomenological bit-flip noise on data and ancilla qubits. Here, the weights are set to -log(p).

    Args:
        d (int): The code distance.
        p (float): The probability of a bit-flip (X error) on each data qubit.
        q (float): The probability of a bit-flip (X error) on each ancilla qubit.

    Returns:
        pymatching.Matching: The decoding graph.
    """

    graph = pymatching.Matching()

    def get_index(d, t, i):
        return (d - 1) * t + i

    # adding space-like edges
    for t in range(d):  # loop over time
        for i in range(1, d - 1):  # loop over space
            a = get_index(d, t, i - 1)
            b = get_index(d, t, i)
            graph.add_edge(a, b, error_probability=p, fault_ids={i})

    # adding time-like edges
    for t in range(1, d):  # loop over time
        for i in range(d - 1):  # loop over space
            a = get_index(d, t - 1, i)
            b = get_index(d, t, i)
            graph.add_edge(a, b, error_probability=p, fault_ids=set())

    # Boundary edges:
    for t in range(d):
        for i in range(d - 1):
            idx = get_index(d, t, i)
            is_edge = i in [0, d - 2]

            if i == 0:
                fault_id = 0
            else:
                fault_id = d - 1

            if is_edge:
                graph.add_boundary_edge(idx, error_probability=p, fault_ids={fault_id})

    return graph


# Problem 3A
def simulate_threshold_bias(n_runs=10**6):
    """
    Simulates the logical error rate of the repetition code using the minimum
    weight perfect matching (MWPM) algorithm for various physical error rates
    and code distances, and plots the results. This time the noise is biased and graph weights are constant.

    Args:
        n_runs (int): The number of runs to perform at each physical error rate.

    Returns:
        threshold: The estimated threshold error rate.
    """

    distances = [3, 5, 7, 9]
    probabilities = np.linspace(0.05, 0.15, 20)
    results = {}

    for d in distances:
        pL_list = []
        print(f"\nSimulating for d = {d}")
        for p in tqdm(probabilities):
            circuit = problem_2.generate_repetition_code_circuit(d, p, 2 * p)  # q = 2*p
            samples = problem_2.measurement_sampler(circuit, n_runs=n_runs)
            defects = problem_2.process_measurements(samples, d)
            graph = build_decoding_graph_const_weight(d, p)
            corrections = graph.decode_batch(defects)
            final_data = samples[:, -d:]
            logical_outcomes = np.sum((final_data ^ corrections), axis=1) % 2
            pL = sum(logical_outcomes) / n_runs
            pL_list.append(pL)
        results[d] = pL_list

    # Estimate threshold
    threshold_p = None
    for i in range(len(probabilities) - 1):
        pL_prev_dist = -1
        all_d = True
        for d in distances:
            if i > 0 and pL_prev_dist > 0:
                # Only when the pL of distances is in increasing order do we mark the threshold
                if pL_prev_dist < results[d][i]:
                    if d == distances[-1] and all_d:
                        threshold_p = (probabilities[i - 1] + probabilities[i]) / 2
                        break
                else:
                    all_d = False
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
        label=f"Estimated threshold ≈ {threshold_p:.3f}",
    )
    plt.xlabel("Physical error rate p")
    plt.ylabel("Logical error rate pL")
    plt.title("Minimum Weight Perfect Matching with Ancillas")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")
    plt.savefig("images/problem_3/threshold_w_bias_noise.png")

    return threshold_p


# Problem 3B
def simulate_threshold_bias_correct_graph(n_runs=10**6):
    """
    Simulates the logical error rate of the repetition code using the minimum
    weight perfect matching (MWPM) algorithm for various physical error rates
    and code distances, and plots the results. This time the noise is biased and graph has correct weights.

    Args:
        n_runs (int): The number of runs to perform at each physical error rate.

    Returns:
        threshold: The estimated threshold error rate.
    """

    distances = [3, 5, 7, 9]
    probabilities = np.linspace(0.05, 0.15, 20)
    results = {}

    for d in distances:
        pL_list = []
        print(f"\nSimulating for d = {d}")
        for p in tqdm(probabilities):
            circuit = problem_2.generate_repetition_code_circuit(d, p, 2 * p)  # q = 2*p
            samples = problem_2.measurement_sampler(circuit, n_runs=n_runs)
            defects = problem_2.process_measurements(samples, d)
            graph = problem_2.build_decoding_graph(d, p, 2 * p)
            corrections = graph.decode_batch(defects)
            final_data = samples[:, -d:]
            logical_outcomes = np.sum((final_data ^ corrections), axis=1) % 2
            pL = sum(logical_outcomes) / n_runs
            pL_list.append(pL)
        results[d] = pL_list

    # Estimate threshold
    threshold_p = None
    for i in range(len(probabilities) - 1):
        pL_prev_dist = -1
        all_d = True
        for d in distances:
            if i > 0 and pL_prev_dist > 0:
                # Only when the pL of distances is in increasing order do we mark the threshold
                if pL_prev_dist < results[d][i]:
                    if d == distances[-1] and all_d:
                        threshold_p = (probabilities[i - 1] + probabilities[i]) / 2
                        break
                else:
                    all_d = False
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
        label=f"Estimated threshold ≈ {threshold_p:.3f}",
    )
    plt.xlabel("Physical error rate p")
    plt.ylabel("Logical error rate pL")
    plt.title("Minimum Weight Perfect Matching with Ancillas")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")
    plt.savefig("images/problem_3/threshold_w_correct_graph.png")

    return threshold_p


# Bonus
def simulate_threshold_bias_bonus(n_runs=10**6):
    """
    Simulates the logical error rate of the repetition code using the minimum
    weight perfect matching (MWPM) algorithm for various physical error rates
    and code distances, and plots the results. This time ancilla probability is almost 0.

    Args:
        n_runs (int): The number of runs to perform at each physical error rate.

    Returns:
        threshold: The estimated threshold error rate.
    """

    distances = [3, 5, 7, 9]
    probabilities = np.linspace(0.05, 0.15, 20)
    results = {}

    for d in distances:
        pL_list = []
        print(f"\nSimulating for d = {d}")
        for p in tqdm(probabilities):
            circuit = problem_2.generate_repetition_code_circuit(d, p, 1e-5)
            samples = problem_2.measurement_sampler(circuit, n_runs=n_runs)
            defects = problem_2.process_measurements(samples, d)
            graph = problem_2.build_decoding_graph(d, p, 1e-5)
            corrections = graph.decode_batch(defects)
            final_data = samples[:, -d:]
            logical_outcomes = np.sum((final_data ^ corrections), axis=1) % 2
            pL = sum(logical_outcomes) / n_runs
            pL_list.append(pL)
        results[d] = pL_list

    # Estimate threshold
    threshold_p = None
    for i in range(len(probabilities) - 1):
        pL_prev_dist = -1
        all_d = True
        for d in distances:
            if i > 0 and pL_prev_dist > 0:
                # Only when the pL of distances is in increasing order do we mark the threshold
                if pL_prev_dist < results[d][i]:
                    if d == distances[-1] and all_d:
                        threshold_p = (probabilities[i - 1] + probabilities[i]) / 2
                        break
                else:
                    all_d = False
            pL_prev_dist = results[d][i]
        if threshold_p is not None:
            break

    if threshold_p is None:
        threshold_p = 0

    # Plotting
    plt.figure(figsize=(10, 6))
    for d in distances:
        plt.plot(probabilities, results[d], label=f"d = {d}")

    # Plot threshold marker
    plt.axvline(
        x=threshold_p,
        color="red",
        linestyle="--",
        label=f"Estimated threshold ≈ {threshold_p:.3f}",
    )
    plt.xlabel("Physical error rate p")
    plt.ylabel("Logical error rate pL")
    plt.title("Minimum Weight Perfect Matching with Ancillas")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")
    plt.savefig("images/bonus/threshold_no_ancilla_noise.png")

    return threshold_p
