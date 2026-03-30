"""
Decoherence Module
==================

THE PHYSICS OF DECOHERENCE IN TRAPPED IONS:

    ┌────────────────────────────────────────────────────────────────┐
    │  Decoherence = loss of quantum information to the environment │
    │                                                               │
    │  A qubit in superposition α|0⟩ + β|1⟩ gradually loses its    │
    │  "quantumness" due to unwanted interactions with the          │
    │  environment. This is described by two timescales:            │
    │                                                               │
    │  T₁ (RELAXATION / AMPLITUDE DAMPING):                        │
    │    |1⟩ → |0⟩ spontaneously (like radioactive decay)          │
    │    Time for excited state population to decay to 1/e          │
    │    Physical cause: spontaneous photon emission                │
    │    Ca⁺ optical qubit: T₁ ~ 1.17 s (metastable D₅/₂)        │
    │    Yb⁺ hyperfine qubit: T₁ ~ 10¹⁰ s (essentially ∞)        │
    │                                                               │
    │  T₂ (DEPHASING / PHASE DAMPING):                             │
    │    The relative phase between |0⟩ and |1⟩ randomizes          │
    │    Time for coherence (off-diagonal ρ₀₁) to decay to 1/e    │
    │    Physical causes:                                           │
    │      - Fluctuating magnetic fields (Zeeman shift noise)      │
    │      - Laser phase noise (for optical qubits)                │
    │      - AC-Stark shifts from stray light                      │
    │    Always T₂ ≤ 2T₁ (dephasing is always faster)             │
    │                                                               │
    │  T₂* (INHOMOGENEOUS DEPHASING):                              │
    │    Additional dephasing from shot-to-shot fluctuations        │
    │    Can be partially reversed by spin echo (refocusing) pulses │
    │                                                               │
    │  MOTIONAL HEATING:                                            │
    │    The ion's motion in the trap heats up due to electric      │
    │    field noise from the electrodes:                           │
    │      dn̄/dt = heating rate (phonons/second)                   │
    │    This affects entangling gates that use phonon modes.       │
    │    Cryogenic traps reduce this by ~100× compared to RT.      │
    │                                                               │
    │  MATHEMATICAL FRAMEWORK:                                      │
    │    Decoherence is modeled via the Lindblad master equation:   │
    │                                                               │
    │    dρ/dt = -i[H,ρ]/ℏ + Σ_k γ_k (L_k ρ L_k† - ½{L_k†L_k,ρ})│
    │             ↑ unitary      ↑ dissipative (Lindblad terms)     │
    │                                                               │
    │    Or equivalently via Kraus operators for discrete steps:    │
    │    ρ → Σ_i K_i ρ K_i†                                       │
    └────────────────────────────────────────────────────────────────┘
"""

import numpy as np
from typing import List, Optional
from .quantum_state import QuantumState
from .gates import I2, SIGMA_X, SIGMA_Y, SIGMA_Z, SIGMA_PLUS, SIGMA_MINUS


# ═══════════════════════════════════════════════════════════════════
# Decoherence Channels (Kraus Operator Representations)
# ═══════════════════════════════════════════════════════════════════

def amplitude_damping_kraus(gamma: float) -> List[np.ndarray]:
    """
    Amplitude damping channel (T₁ process).

    PHYSICS:
        Models spontaneous emission: |1⟩ → |0⟩ with probability γ.

        For a waiting time t:
            γ = 1 - exp(-t/T₁)

        Kraus operators:
            K₀ = [[1, 0], [0, √(1-γ)]]   (no decay occurred)
            K₁ = [[0, √γ], [0, 0]]         (decay occurred: photon emitted)

        Effect on density matrix:
            ρ₀₀ → ρ₀₀ + γ·ρ₁₁    (ground state population increases)
            ρ₁₁ → (1-γ)·ρ₁₁      (excited state decays)
            ρ₀₁ → √(1-γ)·ρ₀₁    (coherence partially lost)
    """
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    return [K0, K1]


def phase_damping_kraus(gamma: float) -> List[np.ndarray]:
    """
    Phase damping (pure dephasing) channel (T₂ process, beyond T₁ contribution).

    PHYSICS:
        Models random phase kicks from environmental noise (magnetic field
        fluctuations, laser phase noise, etc.) without energy exchange.

        For a waiting time t (pure dephasing part only):
            γ = 1 - exp(-t·(1/T₂ - 1/(2T₁)))

        Kraus operators:
            K₀ = [[1, 0], [0, √(1-γ)]]   (no phase kick)
            K₁ = [[0, 0], [0, √γ]]         (phase randomized)

        Effect:
            ρ₀₀ → ρ₀₀  (populations unchanged!)
            ρ₁₁ → ρ₁₁
            ρ₀₁ → (1-γ)·ρ₀₁  (coherence decays)

        Pure dephasing kills superpositions but doesn't change populations.
        This is the dominant error in hyperfine qubits (Yb⁺, Be⁺).
    """
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    K1 = np.array([[0, 0], [0, np.sqrt(gamma)]], dtype=complex)
    return [K0, K1]


def depolarizing_kraus(p: float) -> List[np.ndarray]:
    """
    Depolarizing channel.

    PHYSICS:
        With probability p, the qubit is replaced by the maximally mixed
        state I/2 (complete loss of information). With probability 1-p,
        it's unchanged.

        ρ → (1-p)ρ + p·I/2

        Kraus operators:
            K₀ = √(1-3p/4) · I    (nothing happens)
            K₁ = √(p/4) · σ_x     (bit flip)
            K₂ = √(p/4) · σ_y     (bit+phase flip)
            K₃ = √(p/4) · σ_z     (phase flip)

        This is a "worst-case" noise model where all types of errors
        occur with equal probability.
    """
    K0 = np.sqrt(1 - 3*p/4) * I2
    K1 = np.sqrt(p/4) * SIGMA_X
    K2 = np.sqrt(p/4) * SIGMA_Y
    K3 = np.sqrt(p/4) * SIGMA_Z
    return [K0, K1, K2, K3]


# ═══════════════════════════════════════════════════════════════════
# Decoherence Model for Trapped Ions
# ═══════════════════════════════════════════════════════════════════

class TrappedIonDecoherence:
    """
    Realistic decoherence model for trapped-ion qubits.

    Combines:
        1. Amplitude damping (T₁ relaxation)
        2. Phase damping (T₂ dephasing)
        3. Gate infidelity (depolarizing noise per gate)
        4. Crosstalk (off-target light on neighboring ions)

    PHYSICS of each noise source:

    AMPLITUDE DAMPING (T₁):
        For optical qubits (Ca⁺): spontaneous emission from D₅/₂
        Rate: Γ₁ = 1/T₁ ≈ 0.86 s⁻¹
        For hyperfine qubits (Yb⁺): effectively zero

    PHASE DAMPING (T₂):
        Magnetic field fluctuations → random Zeeman shift → phase noise
        For "clock" transitions (m_F=0 ↔ m_F=0): first-order insensitive
        Residual dephasing from second-order Zeeman, AC Stark shifts
        Can be improved with dynamical decoupling (spin echo sequences)

    GATE ERRORS:
        Single-qubit: ~10⁻⁴ (intensity noise, beam pointing, pulse calibration)
        Two-qubit MS: ~10⁻³ to 10⁻² (motional heating, mode frequency drift,
                       off-resonant coupling to spectator modes)

    CROSSTALK:
        Laser beams on one ion partially illuminate neighbors.
        For tightly focused beams: crosstalk ~10⁻⁴ to 10⁻²
        Depends on ion spacing (~5 μm) vs beam waist (~1-2 μm)
    """

    def __init__(self, t1: float = 1.0, t2: float = 0.01,
                 single_qubit_error: float = 1e-4,
                 two_qubit_error: float = 1e-3,
                 readout_error: float = 1e-3,
                 crosstalk: float = 1e-3,
                 heating_rate: float = 100.0):
        """
        Args:
            t1: T₁ relaxation time in seconds.
            t2: T₂ coherence time in seconds.
            single_qubit_error: Depolarizing error per single-qubit gate.
            two_qubit_error: Depolarizing error per two-qubit (MS) gate.
            readout_error: Bit-flip probability per measurement.
            crosstalk: Fraction of gate applied to neighboring ions.
            heating_rate: Motional heating rate in phonons/second.
        """
        self.t1 = t1
        self.t2 = t2
        self.single_qubit_error = single_qubit_error
        self.two_qubit_error = two_qubit_error
        self.readout_error = readout_error
        self.crosstalk = crosstalk
        self.heating_rate = heating_rate

    def idle_decoherence(self, duration_seconds: float, qubit: int,
                          num_qubits: int) -> List[np.ndarray]:
        """
        Compute Kraus operators for idle decoherence during a wait time.

        PHYSICS:
            When an ion is idle (no gate being applied), it still experiences:
            1. Amplitude damping: γ₁ = 1 - exp(-t/T₁)
            2. Phase damping: γ₂ = 1 - exp(-t/T₂) (includes T₁ contribution)

            The total coherence decay is:
                ρ₀₁(t) = ρ₀₁(0) · exp(-t/T₂)
                        = ρ₀₁(0) · exp(-t/(2T₁)) · exp(-t/T_φ)
            where T_φ is the pure dephasing time: 1/T₂ = 1/(2T₁) + 1/T_φ
        """
        gamma_1 = 1 - np.exp(-duration_seconds / self.t1) if self.t1 > 0 else 0
        gamma_phi = 0
        if self.t2 > 0 and self.t1 > 0:
            # Pure dephasing rate
            rate_phi = max(0, 1/self.t2 - 1/(2*self.t1))
            gamma_phi = 1 - np.exp(-duration_seconds * rate_phi)

        # Combine amplitude damping and pure dephasing
        kraus_1q = _combine_channels(
            amplitude_damping_kraus(gamma_1),
            phase_damping_kraus(gamma_phi)
        )

        # Embed in full Hilbert space
        return _embed_kraus(kraus_1q, qubit, num_qubits)

    def gate_error_kraus(self, gate_type: str, target_qubits: List[int],
                          num_qubits: int) -> List[np.ndarray]:
        """
        Compute Kraus operators for gate imperfections.

        Models gate error as a depolarizing channel applied after the
        ideal gate operation.
        """
        if gate_type == 'single':
            p = self.single_qubit_error
            kraus_1q = depolarizing_kraus(p)
            return _embed_kraus(kraus_1q, target_qubits[0], num_qubits)
        elif gate_type == 'two_qubit':
            p = self.two_qubit_error
            kraus_2q = _two_qubit_depolarizing_kraus(p)
            return _embed_kraus_2q(kraus_2q, target_qubits[0],
                                    target_qubits[1], num_qubits)
        return [np.eye(2**num_qubits, dtype=complex)]

    def print_error_budget(self, circuit_depth: int = 10,
                            num_two_qubit_gates: int = 5):
        """Print estimated error contributions for a circuit."""
        print("\n  ── Error Budget Estimate ──")
        print(f"  Circuit: {circuit_depth} layers, {num_two_qubit_gates} MS gates")

        # Single-qubit gate errors
        sq_errors = circuit_depth * self.single_qubit_error
        print(f"  Single-qubit gate errors: {circuit_depth} × {self.single_qubit_error:.1e} "
              f"= {sq_errors:.4f}")

        # Two-qubit gate errors (dominant!)
        tq_errors = num_two_qubit_gates * self.two_qubit_error
        print(f"  Two-qubit gate errors:    {num_two_qubit_gates} × {self.two_qubit_error:.1e} "
              f"= {tq_errors:.4f} ← usually dominant")

        # Decoherence during circuit (rough estimate)
        gate_time = 5e-6 * circuit_depth + 100e-6 * num_two_qubit_gates
        decoh_error = gate_time / self.t2 if self.t2 > 0 else 0
        print(f"  Decoherence (T₂):         t_circuit ≈ {gate_time*1e6:.0f} μs, "
              f"t/T₂ = {decoh_error:.4f}")

        # Readout errors
        print(f"  Readout error:            {self.readout_error:.4f}")

        total = sq_errors + tq_errors + decoh_error + self.readout_error
        print(f"  ─────────────────────────────────────")
        print(f"  Total estimated error:    ~{total:.4f} ({total*100:.2f}%)")


# ═══════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════

def _combine_channels(kraus_a: List[np.ndarray],
                       kraus_b: List[np.ndarray]) -> List[np.ndarray]:
    """Compose two channels: (B ∘ A)(ρ) = B(A(ρ))."""
    combined = []
    for Ka in kraus_a:
        for Kb in kraus_b:
            combined.append(Kb @ Ka)
    return combined


def _embed_kraus(kraus_1q: List[np.ndarray], qubit: int,
                  num_qubits: int) -> List[np.ndarray]:
    """Embed single-qubit Kraus operators into full Hilbert space."""
    result = []
    for K in kraus_1q:
        matrices = []
        for i in range(num_qubits):
            if i == qubit:
                matrices.append(K)
            else:
                matrices.append(I2)
        full_K = matrices[0]
        for m in matrices[1:]:
            full_K = np.kron(full_K, m)
        result.append(full_K)
    return result


def _two_qubit_depolarizing_kraus(p: float) -> List[np.ndarray]:
    """
    Two-qubit depolarizing channel.

    ρ → (1-p)ρ + p·I/4

    Uses 16 Kraus operators: tensor products of {I, σ_x, σ_y, σ_z}.
    """
    paulis = [I2, SIGMA_X, SIGMA_Y, SIGMA_Z]
    kraus = []
    # Identity contribution
    kraus.append(np.sqrt(1 - 15*p/16) * np.kron(I2, I2))
    # Error contributions
    for i, Pi in enumerate(paulis):
        for j, Pj in enumerate(paulis):
            if i == 0 and j == 0:
                continue  # Already added
            kraus.append(np.sqrt(p/16) * np.kron(Pi, Pj))
    return kraus


def _embed_kraus_2q(kraus_2q: List[np.ndarray], q0: int, q1: int,
                     num_qubits: int) -> List[np.ndarray]:
    """Embed two-qubit Kraus operators into full Hilbert space."""
    dim = 2 ** num_qubits
    result = []

    for K_2q in kraus_2q:
        full_K = np.zeros((dim, dim), dtype=complex)

        for i in range(dim):
            for j in range(dim):
                # Check if i,j differ only on qubits q0, q1
                mask = ~((1 << (num_qubits - 1 - q0)) | (1 << (num_qubits - 1 - q1)))
                if (i & mask) != (j & mask):
                    continue

                i_q0 = (i >> (num_qubits - 1 - q0)) & 1
                i_q1 = (i >> (num_qubits - 1 - q1)) & 1
                j_q0 = (j >> (num_qubits - 1 - q0)) & 1
                j_q1 = (j >> (num_qubits - 1 - q1)) & 1

                sub_i = i_q0 * 2 + i_q1
                sub_j = j_q0 * 2 + j_q1

                full_K[i, j] = K_2q[sub_i, sub_j]

        result.append(full_K)
    return result
