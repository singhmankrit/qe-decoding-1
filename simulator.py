import stim

def generate_repetition_code_circuit(d: int, p: float) -> stim.Circuit:
    """
    Generate a Stim circuit for a distance-d repetition code with incoming bit-flip noise.

    Args:
        d (int): Code distance (must be odd).
        p (float): Probability of a bit-flip (X error) on each qubit.

    Returns:
        stim.Circuit: The corresponding Stim circuit.
    """
    circuit = stim.Circuit()

    # Step 1: Initialize all data qubits to |0>
    for q in range(d):
        circuit.append("R", [q])         # Reset to |0⟩
    
    # Step 2: Apply bit-flip noise (X with probability p)
    for q in range(d):
        circuit.append("X_ERROR", [q], p)
    
    # Step 3: Measure all data qubits in Z-basis
    for q in range(d):
        circuit.append("M", [q])
    
    return circuit
