import stim
import numpy as np

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
        if sum(run) < (d-1)/2:
            error_corrected.append(0)
        else:
            error_corrected.append(1)
    return error_corrected

