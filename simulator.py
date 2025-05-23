import stim


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
        circuit.append("M", [q])

    return circuit
