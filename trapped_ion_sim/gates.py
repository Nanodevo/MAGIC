"""
Quantum Gate Operations Module
===============================

THE PHYSICS OF GATES IN TRAPPED-ION SYSTEMS:

    ┌─────────────────────────────────────────────────────────────────────┐
    │                    LASER-ION INTERACTION                           │
    │                                                                    │
    │  Gates are implemented by shining precisely tuned laser beams      │
    │  on individual ions. The laser-ion Hamiltonian (in the rotating    │
    │  frame and Lamb-Dicke approximation) is:                          │
    │                                                                    │
    │    H = (ℏΩ/2)(σ₊ e^{iφ} + σ₋ e^{-iφ})                          │
    │      + ℏΩη(σ₊ e^{iφ}(â e^{-iδt} + â†e^{iδt}) + h.c.)           │
    │                                                                    │
    │  where:                                                            │
    │    Ω   = Rabi frequency (laser power → gate speed)                │
    │    φ   = laser phase (controls rotation axis)                     │
    │    σ±  = qubit raising/lowering operators                         │
    │    â,â†= motional mode ladder operators                           │
    │    η   = Lamb-Dicke parameter (ion-motion coupling)               │
    │    δ   = laser detuning from qubit transition                     │
    │                                                                    │
    │  Three types of transitions:                                       │
    │                                                                    │
    │    CARRIER (δ=0):      |↓,n⟩ ↔ |↑,n⟩       (qubit flip only)    │
    │    RED SIDEBAND (δ=-ω): |↓,n⟩ ↔ |↑,n-1⟩    (flip + cool)       │
    │    BLUE SIDEBAND (δ=+ω):|↓,n⟩ ↔ |↑,n+1⟩    (flip + heat)       │
    │                                                                    │
    │  Single-qubit gates use CARRIER transitions.                       │
    │  Two-qubit gates use SIDEBAND transitions to create entanglement  │
    │  mediated by shared phonon modes.                                  │
    └─────────────────────────────────────────────────────────────────────┘

    SINGLE-QUBIT GATES:
        A carrier pulse for time t produces Rabi oscillations:
            |↓⟩ → cos(Ωt/2)|↓⟩ + ie^{iφ} sin(Ωt/2)|↑⟩

        This implements the rotation:
            R(θ,φ) = exp(-iθ(σ_x cos φ + σ_y sin φ)/2)

        where θ = Ωt is the pulse area.

        Common gates:
            π/2 pulse (θ=π/2): creates superposition
            π pulse   (θ=π):   flips the qubit (NOT gate)
            R_z       :        implemented by adjusting laser phase (virtual Z gate)

        Gate times: ~1-10 μs for single-qubit gates
        Fidelities: >99.99% demonstrated

    TWO-QUBIT GATES:
        The Mølmer-Sørensen (MS) gate is the workhorse entangling gate.
        See gates module for detailed physics.
"""

import numpy as np
from typing import Optional, Tuple
from scipy.linalg import expm

# ═══════════════════════════════════════════════════════════════════
# Pauli Matrices (fundamental building blocks)
# ═══════════════════════════════════════════════════════════════════

# The Pauli matrices form a basis for all single-qubit operations.
# Any 2×2 Hermitian matrix = a·I + b·σ_x + c·σ_y + d·σ_z
I2 = np.eye(2, dtype=complex)
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)       # Bit flip
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)    # Bit+phase flip
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)      # Phase flip
SIGMA_PLUS = np.array([[0, 1], [0, 0]], dtype=complex)    # |0⟩→|1⟩
SIGMA_MINUS = np.array([[0, 0], [1, 0]], dtype=complex)   # |1⟩→|0⟩


# ═══════════════════════════════════════════════════════════════════
# Single-Qubit Gate
# ═══════════════════════════════════════════════════════════════════

class SingleQubitGate:
    """
    Single-qubit gate implemented via carrier Rabi pulse on one ion.

    PHYSICS:
        A resonant laser pulse on ion i produces the Hamiltonian:
            H = (ℏΩ/2)(σ₊ e^{iφ} + σ₋ e^{-iφ})

        Time evolution for duration t gives:
            R(θ, φ) = cos(θ/2)·I - i·sin(θ/2)·(cos(φ)·σ_x + sin(φ)·σ_y)

        where θ = Ω·t is the pulse area (rotation angle).

        This is physically a RABI OSCILLATION: the qubit coherently
        oscillates between |0⟩ and |1⟩ at the Rabi frequency Ω.
    """

    def __init__(self, theta: float, phi: float, target_qubit: int):
        """
        Args:
            theta: Rotation angle (pulse area Ω·t), in radians
            phi: Rotation axis angle in X-Y plane (laser phase), in radians
            target_qubit: Which ion (0-indexed) to apply gate to
        """
        self.theta = theta
        self.phi = phi
        self.target_qubit = target_qubit
        self._matrix = self._compute_matrix()

    def _compute_matrix(self) -> np.ndarray:
        """
        Compute the 2×2 unitary for this rotation.

        R(θ,φ) = [[cos(θ/2),        -i·e^{-iφ}·sin(θ/2)],
                   [-i·e^{iφ}·sin(θ/2),  cos(θ/2)         ]]
        """
        c = np.cos(self.theta / 2)
        s = np.sin(self.theta / 2)
        return np.array([
            [c, -1j * np.exp(-1j * self.phi) * s],
            [-1j * np.exp(1j * self.phi) * s, c]
        ], dtype=complex)

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix.copy()

    def full_matrix(self, num_qubits: int) -> np.ndarray:
        """
        Embed 2×2 gate into full 2^N × 2^N Hilbert space.

        PHYSICS:
            Since the laser only addresses ion i, the full operator is:
            U = I ⊗ I ⊗ ... ⊗ R(θ,φ) ⊗ ... ⊗ I
                                ↑ position i

            This tensor product structure reflects that the gate acts
            only on the target ion's internal state.
        """
        q = self.target_qubit
        matrices = []
        for i in range(num_qubits):
            if i == q:
                matrices.append(self._matrix)
            else:
                matrices.append(I2)

        result = matrices[0]
        for m in matrices[1:]:
            result = np.kron(result, m)
        return result


# ═══════════════════════════════════════════════════════════════════
# Common Single-Qubit Gate Constructors
# ═══════════════════════════════════════════════════════════════════

def Rx(theta: float, target: int) -> SingleQubitGate:
    """
    Rotation about X-axis: R_x(θ) = exp(-iθσ_x/2)

    PHYSICS: Corresponds to a carrier pulse with laser phase φ = 0.
    """
    return SingleQubitGate(theta=theta, phi=0.0, target_qubit=target)


def Ry(theta: float, target: int) -> SingleQubitGate:
    """
    Rotation about Y-axis: R_y(θ) = exp(-iθσ_y/2)

    PHYSICS: Carrier pulse with laser phase φ = π/2.
    """
    return SingleQubitGate(theta=theta, phi=np.pi/2, target_qubit=target)


def Rz(theta: float, target: int) -> 'VirtualZGate':
    """
    Rotation about Z-axis: R_z(θ) = exp(-iθσ_z/2)

    PHYSICS: In trapped ions, Z rotations are "virtual" — they don't
    require a physical laser pulse! Instead, we simply update the
    reference frame (phase) of all subsequent pulses on that ion.

    R_z(θ) = [[e^{-iθ/2}, 0], [0, e^{iθ/2}]]

    This is a key advantage: Rz gates are PERFECT (zero error, zero time).
    """
    return VirtualZGate(theta=theta, target_qubit=target)


class VirtualZGate:
    """
    Virtual Z gate — implemented by phase tracking, not a physical pulse.

    PHYSICS:
        Instead of physically rotating the qubit about Z, we shift the
        phase reference frame for all subsequent gates on this qubit.
        This is equivalent because:
            R_z(θ) · R(α, φ) = R(α, φ-θ) · R_z(θ)

        So we can commute all Z rotations to the end of the circuit
        and absorb them into the measurement basis.

        This "virtual Z gate" trick was introduced by McKay et al.
        and is universally used in trapped-ion systems.
    """

    def __init__(self, theta: float, target_qubit: int):
        self.theta = theta
        self.target_qubit = target_qubit
        self._matrix = np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)]
        ], dtype=complex)

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix.copy()

    def full_matrix(self, num_qubits: int) -> np.ndarray:
        matrices = []
        for i in range(num_qubits):
            if i == self.target_qubit:
                matrices.append(self._matrix)
            else:
                matrices.append(I2)
        result = matrices[0]
        for m in matrices[1:]:
            result = np.kron(result, m)
        return result


def Hadamard(target: int) -> SingleQubitGate:
    """
    Hadamard gate: H = R_z(π) · R_y(π/2)

    PHYSICS: Creates equal superposition: H|0⟩ = (|0⟩+|1⟩)/√2
    In trapped ions, decomposed as virtual-Z(π) then R(π/2, π/2).
    We construct the correct matrix directly:
    H = (1/√2)[[1, 1], [1, -1]]
    """
    gate = SingleQubitGate(theta=0, phi=0, target_qubit=target)
    gate._matrix = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    return gate


def PauliX(target: int) -> SingleQubitGate:
    """NOT gate / σ_x: π-pulse about X. Flips |0⟩↔|1⟩."""
    return Rx(np.pi, target)


def PauliY(target: int) -> SingleQubitGate:
    """σ_y gate: π-pulse about Y."""
    return Ry(np.pi, target)


def S_gate(target: int) -> 'VirtualZGate':
    """S gate = R_z(π/2): Quarter-turn phase gate. Virtual in trapped ions."""
    return Rz(np.pi/2, target)


def T_gate(target: int) -> 'VirtualZGate':
    """T gate = R_z(π/4): Eighth-turn phase gate. Virtual in trapped ions."""
    return Rz(np.pi/4, target)


# ═══════════════════════════════════════════════════════════════════
# Mølmer-Sørensen (MS) Gate — THE Entangling Gate
# ═══════════════════════════════════════════════════════════════════

class MolmerSorensenGate:
    """
    The Mølmer-Sørensen (MS) entangling gate.

    ╔═══════════════════════════════════════════════════════════════╗
    ║  THE PHYSICS OF THE MØLMER-SØRENSEN GATE                    ║
    ║                                                              ║
    ║  This is the most important gate in trapped-ion QC.          ║
    ║  It creates entanglement between two ions using their        ║
    ║  shared motional mode as a quantum bus.                      ║
    ║                                                              ║
    ║  HOW IT WORKS (step by step):                                ║
    ║                                                              ║
    ║  1. Two laser beams illuminate both ions simultaneously,     ║
    ║     with frequencies ω_L ± δ (slightly detuned from the     ║
    ║     red and blue motional sidebands):                        ║
    ║                                                              ║
    ║       ω_qubit + ω_mode - δ  (near blue sideband)            ║
    ║       ω_qubit - ω_mode + δ  (near red sideband)             ║
    ║                                                              ║
    ║  2. These beams drive simultaneous spin-dependent forces     ║
    ║     on the ions. The effective Hamiltonian is:               ║
    ║                                                              ║
    ║       H_MS = ℏΩη/2 · Σ_i σ_φ,i · (â·e^{-iδt} + â†·e^{iδt})║
    ║                                                              ║
    ║     where σ_φ = cos(φ)σ_x + sin(φ)σ_y.                     ║
    ║                                                              ║
    ║  3. The motional mode gets displaced in phase space by an    ║
    ║     amount that depends on the SPIN STATE:                   ║
    ║                                                              ║
    ║     |↑↑⟩: displaced in one direction      ↗ in phase space  ║
    ║     |↓↓⟩: displaced opposite              ↙                 ║
    ║     |↑↓⟩,|↓↑⟩: no net displacement        ●                ║
    ║                                                              ║
    ║  4. After a complete loop (time τ = 2π/δ), the motional     ║
    ║     mode returns to its original state (disentangles from    ║
    ║     the spins), but the SPINS acquire a geometric phase      ║
    ║     proportional to the enclosed phase-space area:           ║
    ║                                                              ║
    ║       φ_geom = π · (Ωη)²/δ                                  ║
    ║                                                              ║
    ║  5. When φ_geom = π/4 (by tuning Ω, η, δ), we get the      ║
    ║     maximally entangling MS gate:                            ║
    ║                                                              ║
    ║       MS = exp(-i(π/4) σ_x⊗σ_x)                             ║
    ║                                                              ║
    ║     |↓↓⟩ → (|↓↓⟩ + i|↑↑⟩)/√2  (a Bell state!)             ║
    ║                                                              ║
    ║  KEY INSIGHT: The motional mode acts as a "quantum bus"      ║
    ║  that mediates the interaction, but is not left entangled    ║
    ║  with the qubits at the end. This is the "geometric phase   ║
    ║  gate" mechanism.                                            ║
    ║                                                              ║
    ║  Gate time: ~50-200 μs                                       ║
    ║  Fidelity: >99.9% demonstrated (Ballance et al., 2016)      ║
    ╚═══════════════════════════════════════════════════════════════╝
    """

    def __init__(self, target_qubits: Tuple[int, int], chi: float = np.pi/4,
                 phi: float = 0.0):
        """
        Args:
            target_qubits: Pair of ion indices to entangle.
            chi: Entangling angle. χ = π/4 gives maximally entangling gate.
                 PHYSICS: chi = (Ωη)²τ/(4δ), controlled by laser power and detuning.
            phi: Axis of the spin-spin interaction in the X-Y plane.
                 φ = 0 → XX interaction, φ = π/2 → YY interaction.
        """
        self.target_qubits = target_qubits
        self.chi = chi
        self.phi = phi
        self._matrix = self._compute_matrix()

    def _compute_matrix(self) -> np.ndarray:
        """
        Compute the 4×4 unitary for the MS gate on the two-qubit subspace.

        U_MS = exp(-i·χ·σ_φ ⊗ σ_φ)

        where σ_φ = cos(φ)σ_x + sin(φ)σ_y.

        For φ=0 (XX gate):
            U_XX = exp(-iχ·σ_x⊗σ_x)
                 = [[cos(χ),  0,        0,      -i·sin(χ)],
                    [0,       cos(χ),  -i·sin(χ), 0       ],
                    [0,      -i·sin(χ), cos(χ),   0       ],
                    [-i·sin(χ), 0,      0,        cos(χ)  ]]
        """
        sigma_phi = np.cos(self.phi) * SIGMA_X + np.sin(self.phi) * SIGMA_Y
        H_eff = self.chi * np.kron(sigma_phi, sigma_phi)
        return expm(-1j * H_eff)

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix.copy()

    def full_matrix(self, num_qubits: int) -> np.ndarray:
        """
        Embed 4×4 MS gate into full 2^N Hilbert space.

        Only acts on the two target qubits; identity on all others.
        """
        q0, q1 = self.target_qubits
        dim = 2 ** num_qubits

        full_U = np.eye(dim, dtype=complex)

        # Iterate over all basis states and apply the 4×4 on target subspace
        for i in range(dim):
            for j in range(dim):
                # Check if i and j differ only on qubits q0 and q1
                mask = ~((1 << (num_qubits - 1 - q0)) | (1 << (num_qubits - 1 - q1)))
                if (i & mask) != (j & mask):
                    continue

                # Extract bits for the two target qubits
                i_q0 = (i >> (num_qubits - 1 - q0)) & 1
                i_q1 = (i >> (num_qubits - 1 - q1)) & 1
                j_q0 = (j >> (num_qubits - 1 - q0)) & 1
                j_q1 = (j >> (num_qubits - 1 - q1)) & 1

                # 2-qubit subspace indices
                sub_i = i_q0 * 2 + i_q1
                sub_j = j_q0 * 2 + j_q1

                full_U[i, j] = self._matrix[sub_i, sub_j]

        return full_U


# ═══════════════════════════════════════════════════════════════════
# CNOT Gate (decomposed into native trapped-ion gates)
# ═══════════════════════════════════════════════════════════════════

class CNOTGate:
    """
    CNOT gate decomposed into native trapped-ion operations.

    PHYSICS:
        CNOT is NOT a native gate in trapped-ion systems!
        It must be decomposed into native gates (MS + single-qubit rotations):

        CNOT = (R_y(π/2) ⊗ I) · MS(π/4) · (R_x(-π/2) ⊗ R_x(-π/2)) · (R_y(-π/2) ⊗ I)

        This decomposition uses 1 MS gate + 3 single-qubit gates.
        Since single-qubit gates have ~10× higher fidelity and ~10× shorter
        duration than MS gates, the CNOT fidelity is dominated by the MS gate.
    """

    def __init__(self, control: int, target: int):
        self.control = control
        self.target = target

    def decompose(self):
        """
        Return the sequence of native gates that implement CNOT.

        CNOT(c,t) = Ry(π/2)_t · MS(π/4) · Rx(-π/2)_c · Rx(-π/2)_t · Ry(-π/2)_t

        Returns list of gate objects in execution order.
        """
        return [
            Ry(-np.pi/2, self.target),                            # R_y(-π/2) on target
            MolmerSorensenGate((self.control, self.target)),      # MS(π/4) gate
            Rx(-np.pi/2, self.control),                           # R_x(-π/2) on control
            Rx(-np.pi/2, self.target),                            # R_x(-π/2) on target
            Ry(np.pi/2, self.target),                             # R_y(π/2) on target
        ]

    def full_matrix(self, num_qubits: int) -> np.ndarray:
        """Compute the CNOT unitary directly: |0><0|⊗I + |1><1|⊗X."""
        dim = 2 ** num_qubits
        U = np.zeros((dim, dim), dtype=complex)
        for i in range(dim):
            bits = list(format(i, f'0{num_qubits}b'))
            if bits[self.control] == '0':
                U[i, i] = 1.0
            else:
                bits[self.target] = '1' if bits[self.target] == '0' else '0'
                j = int(''.join(bits), 2)
                U[j, i] = 1.0
        return U


# ═══════════════════════════════════════════════════════════════════
# Gate Timing Model
# ═══════════════════════════════════════════════════════════════════

class GateTimingModel:
    """
    Models realistic gate durations in a trapped-ion system.

    PHYSICS:
        Gate times depend on the Rabi frequency Ω and the pulse area θ:
            t_gate = θ/Ω  for single-qubit gates
            t_MS ≈ 2π/δ   for Mølmer-Sørensen gates (one phonon loop)

        Typical values:
            Single-qubit: 1-10 μs
            MS gate: 50-200 μs
            Measurement: 100-500 μs
    """

    def __init__(self, single_qubit_us: float = 5.0,
                 ms_gate_us: float = 100.0,
                 measurement_us: float = 200.0):
        self.single_qubit_us = single_qubit_us
        self.ms_gate_us = ms_gate_us
        self.measurement_us = measurement_us

    def gate_duration(self, gate) -> float:
        """Return gate duration in microseconds."""
        if isinstance(gate, (SingleQubitGate, VirtualZGate)):
            if isinstance(gate, VirtualZGate):
                return 0.0  # Virtual Z gates take zero time!
            return self.single_qubit_us
        elif isinstance(gate, MolmerSorensenGate):
            return self.ms_gate_us
        elif isinstance(gate, CNOTGate):
            return (self.single_qubit_us * 3 + self.ms_gate_us)
        return 0.0
