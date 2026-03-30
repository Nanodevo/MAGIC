"""
Measurement Module
==================

THE PHYSICS OF MEASUREMENT IN TRAPPED IONS:

    ┌───────────────────────────────────────────────────────────────────┐
    │              STATE-DEPENDENT FLUORESCENCE                        │
    │                                                                  │
    │  Measurement in trapped ions uses the "electron shelving"        │
    │  technique — one of the highest-fidelity qubit readouts in any   │
    │  quantum computing platform.                                     │
    │                                                                  │
    │  HOW IT WORKS:                                                   │
    │                                                                  │
    │  1. Shine a laser resonant with a strong cycling transition      │
    │     (e.g., S₁/₂ ↔ P₁/₂ in Ca⁺ at 397 nm).                     │
    │                                                                  │
    │  2. If the ion is in |0⟩ = |S₁/₂⟩:                             │
    │     → The laser drives rapid cycling: S₁/₂ → P₁/₂ → S₁/₂      │
    │     → Each cycle emits a fluorescence photon                     │
    │     → We detect ~10-30 photons in ~100 μs                        │
    │     → ION GLOWS BRIGHTLY ☀                                       │
    │                                                                  │
    │  3. If the ion is in |1⟩ = |D₅/₂⟩ (shelved state):             │
    │     → The laser is far off-resonance with D₅/₂ ↔ P₁/₂          │
    │     → No cycling occurs, no photons emitted                      │
    │     → ION IS DARK ●                                              │
    │                                                                  │
    │                                                                  │
    │     Energy levels:                                               │
    │                                                                  │
    │         P₁/₂  ─────  (short-lived, ~7 ns)                      │
    │          ↑  ↓ cycling                                            │
    │     |0⟩ S₁/₂  ─────  ← detection laser (397 nm)                │
    │                                                                  │
    │     |1⟩ D₅/₂  ─────  (metastable, ~1.17 s) "shelved"           │
    │                                                                  │
    │                                                                  │
    │  4. A threshold on photon count discriminates |0⟩ vs |1⟩:       │
    │     photons > threshold → measured as |0⟩                        │
    │     photons ≤ threshold → measured as |1⟩                        │
    │                                                                  │
    │  FIDELITY: >99.99% single-shot readout demonstrated!             │
    │  (limited by off-resonant scattering and dark counts)            │
    │                                                                  │
    │  POST-MEASUREMENT STATE:                                         │
    │  After measurement, the qubit collapses to the measured state    │
    │  (projective measurement / Born rule).                           │
    │  If we measured |0⟩, the state is projected: |ψ⟩ → |0⟩          │
    │  This is a fundamental aspect of quantum mechanics.              │
    └───────────────────────────────────────────────────────────────────┘
"""

import numpy as np
from typing import List, Tuple, Optional
from .quantum_state import QuantumState


def measure(state: QuantumState, qubits: Optional[List[int]] = None,
            shots: int = 1, readout_error: float = 0.0) -> dict:
    """
    Perform projective measurement on the quantum state.

    PHYSICS:
        Born rule: P(outcome k) = |⟨k|ψ⟩|² = Tr(Π_k ρ)
        
        Post-measurement state: ρ → Π_k ρ Π_k / P(k)
        where Π_k = |k⟩⟨k| is the projection operator.

        In trapped ions, this is implemented by state-dependent fluorescence:
        each ion is illuminated and we count photons to determine |0⟩ or |1⟩.

    Args:
        state: The quantum state to measure.
        qubits: Which qubits to measure (None = all). Non-measured qubits
                remain in a post-measurement state.
        shots: Number of measurement repetitions (for statistics).
        readout_error: Probability of bit-flip error in readout (0.0 = perfect).
                      PHYSICS: accounts for off-resonant scattering, dark counts,
                      and imperfect photon detection.

    Returns:
        Dictionary mapping bitstring outcomes to counts.
        E.g., {'00': 512, '11': 488} for a Bell state measured 1000 times.
    """
    if qubits is None:
        qubits = list(range(state.num_qubits))

    probabilities = state.get_probabilities()
    num_qubits = state.num_qubits

    # Build mapping from full basis states to measured outcomes
    outcomes = {}
    for k in range(state.dim):
        full_bits = format(k, f'0{num_qubits}b')
        measured_bits = ''.join(full_bits[q] for q in qubits)

        if measured_bits not in outcomes:
            outcomes[measured_bits] = 0.0
        outcomes[measured_bits] += probabilities[k]

    # Sample from the distribution
    outcome_list = list(outcomes.keys())
    prob_list = np.array([outcomes[o] for o in outcome_list])
    prob_list = prob_list / prob_list.sum()  # Normalize (numerical safety)

    rng = np.random.default_rng()
    samples = rng.choice(len(outcome_list), size=shots, p=prob_list)

    # Apply readout error (bit-flip with probability readout_error)
    result_counts = {}
    for idx in samples:
        outcome = outcome_list[idx]

        if readout_error > 0:
            flipped = ""
            for bit in outcome:
                if rng.random() < readout_error:
                    flipped += '1' if bit == '0' else '0'
                else:
                    flipped += bit
            outcome = flipped

        result_counts[outcome] = result_counts.get(outcome, 0) + 1

    return result_counts


def measure_and_collapse(state: QuantumState, qubit: int,
                          readout_error: float = 0.0) -> Tuple[int, QuantumState]:
    """
    Measure a single qubit and return the post-measurement state.

    PHYSICS:
        Projective measurement collapses the state:

        If outcome is |0⟩:
            |ψ⟩ → (Π_0 ⊗ I)|ψ⟩ / ||...||
            where Π_0 = |0⟩⟨0|

        If outcome is |1⟩:
            |ψ⟩ → (Π_1 ⊗ I)|ψ⟩ / ||...||
            where Π_1 = |1⟩⟨1|

        This is the "wavefunction collapse" postulate.
        After measuring one ion, the remaining ions are left in a
        (potentially entangled) conditional state.

    Returns:
        (outcome, new_state): The measurement result (0 or 1) and
        the post-measurement quantum state.
    """
    n = state.num_qubits
    dim = state.dim
    probabilities = state.get_probabilities()

    # Compute probability of measuring 0 on this qubit
    p0 = 0.0
    for k in range(dim):
        bit = (k >> (n - 1 - qubit)) & 1
        if bit == 0:
            p0 += probabilities[k]
    p1 = 1.0 - p0

    # Sample outcome
    rng = np.random.default_rng()
    outcome = 0 if rng.random() < p0 else 1

    # Apply readout error
    if readout_error > 0 and rng.random() < readout_error:
        outcome = 1 - outcome

    # True outcome (for state collapse, use the pre-error outcome)
    true_outcome = outcome

    # Build projection operator
    projector = np.zeros((dim, dim), dtype=complex)
    for k in range(dim):
        bit = (k >> (n - 1 - qubit)) & 1
        if bit == true_outcome:
            projector[k, k] = 1.0

    # Apply projection
    if state.is_pure:
        new_sv = projector @ state.statevector
        norm = np.linalg.norm(new_sv)
        if norm > 0:
            new_sv /= norm
        new_state = QuantumState(n, new_sv)
    else:
        new_rho = projector @ state.density_matrix @ projector
        trace = np.real(np.trace(new_rho))
        if trace > 0:
            new_rho /= trace
        new_state = QuantumState(n, new_rho)

    return outcome, new_state


def expectation_value(state: QuantumState, observable: np.ndarray) -> complex:
    """
    Compute ⟨O⟩ = Tr(ρ·O) for an observable O.

    PHYSICS:
        The expectation value is the average result of measuring O
        on many copies of the state. For Pauli operators:
            ⟨σ_z⟩ = P(|0⟩) - P(|1⟩)  (population difference)
            ⟨σ_x⟩ = 2·Re(ρ₀₁)         (coherence, real part)
            ⟨σ_y⟩ = 2·Im(ρ₀₁)         (coherence, imaginary part)
    """
    rho = state.density_matrix
    return np.trace(rho @ observable)
