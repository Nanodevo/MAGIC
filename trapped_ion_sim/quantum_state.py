"""
Quantum State Module
====================

THE PHYSICS:
    In a trapped-ion quantum computer, each ion encodes one qubit using two
    internal electronic states:

        |0⟩ = |↓⟩ = ground state   (e.g., S₁/₂ in Ca⁺ or ²S₁/₂ F=0 in ¹⁷¹Yb⁺)
        |1⟩ = |↑⟩ = excited state  (e.g., D₅/₂ in Ca⁺ or ²S₁/₂ F=1 in ¹⁷¹Yb⁺)

    For N ions, the combined qubit Hilbert space has dimension 2^N.
    A general state is:

        |ψ⟩ = Σ_{k=0}^{2^N - 1} α_k |k⟩

    where |k⟩ is the computational basis state and α_k are complex amplitudes
    satisfying Σ|α_k|² = 1.

    DENSITY MATRIX:
    For mixed states (open quantum systems with decoherence), we use the
    density matrix ρ = Σ_i p_i |ψ_i⟩⟨ψ_i|, which naturally describes
    classical mixtures of quantum states.

IMPLEMENTATION:
    We support both pure state (statevector) and mixed state (density matrix)
    representations. Pure states are promoted to density matrices when
    decoherence channels are applied.
"""

import numpy as np
from typing import Optional, List, Tuple


class QuantumState:
    """
    Represents the quantum state of N qubits (ions).

    Can be a pure state (statevector) or mixed state (density matrix).
    Internally stores the density matrix for generality; pure states are
    stored as |ψ⟩⟨ψ| but flagged for efficient pure-state operations.
    """

    def __init__(self, num_qubits: int, initial_state: Optional[np.ndarray] = None):
        """
        Initialize quantum state.

        Args:
            num_qubits: Number of qubits (ions).
            initial_state: Optional initial statevector or density matrix.
                           Defaults to |00...0⟩.
        """
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits  # Hilbert space dimension

        if initial_state is None:
            # Default: all ions in ground state |00...0⟩
            self._statevector = np.zeros(self.dim, dtype=complex)
            self._statevector[0] = 1.0
            self._is_pure = True
            self._density_matrix = None
        elif initial_state.ndim == 1:
            # Pure state given as statevector
            assert len(initial_state) == self.dim, \
                f"Statevector length {len(initial_state)} != Hilbert space dim {self.dim}"
            self._statevector = initial_state.astype(complex).copy()
            self._normalize_statevector()
            self._is_pure = True
            self._density_matrix = None
        elif initial_state.ndim == 2:
            # Mixed state given as density matrix
            assert initial_state.shape == (self.dim, self.dim), \
                f"Density matrix shape {initial_state.shape} != ({self.dim}, {self.dim})"
            self._density_matrix = initial_state.astype(complex).copy()
            self._statevector = None
            self._is_pure = False
        else:
            raise ValueError("initial_state must be a 1D (statevector) or 2D (density matrix) array")

    @property
    def is_pure(self) -> bool:
        return self._is_pure

    @property
    def statevector(self) -> Optional[np.ndarray]:
        """Return statevector if pure, else None."""
        return self._statevector.copy() if self._is_pure else None

    @property
    def density_matrix(self) -> np.ndarray:
        """Return density matrix (computed from statevector if pure)."""
        if self._is_pure:
            return np.outer(self._statevector, self._statevector.conj())
        return self._density_matrix.copy()

    def _normalize_statevector(self):
        """Ensure ⟨ψ|ψ⟩ = 1."""
        norm = np.linalg.norm(self._statevector)
        if norm > 0:
            self._statevector /= norm

    def apply_unitary(self, U: np.ndarray):
        """
        Apply a unitary operator U to the state.

        PHYSICS: Unitary evolution |ψ⟩ → U|ψ⟩  or  ρ → UρU†
        This is the Schrödinger picture evolution for a gate operation.
        """
        if self._is_pure:
            self._statevector = U @ self._statevector
        else:
            self._density_matrix = U @ self._density_matrix @ U.conj().T

    def apply_kraus(self, kraus_operators: List[np.ndarray]):
        """
        Apply a quantum channel described by Kraus operators {K_i}.

        PHYSICS: ρ → Σ_i K_i ρ K_i†

        Kraus operators satisfy the completeness relation Σ_i K_i† K_i = I,
        ensuring trace preservation (probability conservation).

        This is how we model decoherence:
        - Dephasing: random phase kicks from fluctuating magnetic fields
        - Amplitude damping: spontaneous emission from excited state
        - Depolarizing: isotropic noise
        """
        # Promote to density matrix if currently pure
        if self._is_pure:
            self._density_matrix = np.outer(self._statevector, self._statevector.conj())
            self._statevector = None
            self._is_pure = False

        new_rho = np.zeros_like(self._density_matrix)
        for K in kraus_operators:
            new_rho += K @ self._density_matrix @ K.conj().T
        self._density_matrix = new_rho

    def get_probabilities(self) -> np.ndarray:
        """
        Get measurement probabilities for each computational basis state.

        PHYSICS: P(k) = |⟨k|ψ⟩|² for pure states, or P(k) = ⟨k|ρ|k⟩ = ρ_kk
        (Born rule).
        """
        if self._is_pure:
            return np.abs(self._statevector) ** 2
        else:
            return np.real(np.diag(self._density_matrix))

    def partial_trace(self, keep_qubits: List[int]) -> np.ndarray:
        """
        Compute partial trace over all qubits NOT in keep_qubits.

        PHYSICS: ρ_A = Tr_B(ρ_{AB})
        This gives the reduced density matrix for a subsystem, which tells
        us about entanglement — if Tr(ρ_A²) < 1, the qubit is entangled
        with others.
        """
        rho = self.density_matrix
        n = self.num_qubits

        # Reshape into tensor with 2 indices per qubit
        rho_tensor = rho.reshape([2] * (2 * n))

        # Determine which qubits to trace over
        trace_qubits = sorted(set(range(n)) - set(keep_qubits))

        # Trace over qubits from highest index to lowest to avoid index shifting
        for q in reversed(trace_qubits):
            # Contract indices q and q+n (bra and ket for this qubit)
            rho_tensor = np.trace(rho_tensor, axis1=q, axis2=q + n - 1)
            n -= 1

        dim_keep = 2 ** len(keep_qubits)
        return rho_tensor.reshape(dim_keep, dim_keep)

    def fidelity(self, other: 'QuantumState') -> float:
        """
        Compute fidelity F(ρ, σ) between this state and another.

        PHYSICS: For pure states, F = |⟨ψ|φ⟩|²
        For mixed states, F = (Tr√(√ρ σ √ρ))²
        Fidelity = 1 means identical states; 0 means orthogonal.
        """
        if self._is_pure and other._is_pure:
            return float(np.abs(np.dot(self._statevector.conj(), other._statevector)) ** 2)

        rho = self.density_matrix
        sigma = other.density_matrix

        sqrt_rho = _matrix_sqrt(rho)
        product = sqrt_rho @ sigma @ sqrt_rho
        sqrt_product = _matrix_sqrt(product)
        return float(np.real(np.trace(sqrt_product)) ** 2)

    def von_neumann_entropy(self) -> float:
        """
        Compute von Neumann entropy S(ρ) = -Tr(ρ log₂ ρ).

        PHYSICS: S = 0 for pure states, S = N for maximally mixed N-qubit state.
        This quantifies how "mixed" (noisy) the state is.
        """
        if self._is_pure:
            return 0.0

        eigenvalues = np.linalg.eigvalsh(self._density_matrix)
        # Filter out zero/negative eigenvalues (numerical noise)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]
        return float(-np.sum(eigenvalues * np.log2(eigenvalues)))

    def bloch_vector(self, qubit: int) -> Tuple[float, float, float]:
        """
        Get the Bloch vector (r_x, r_y, r_z) for a single qubit.

        PHYSICS: Any single-qubit density matrix can be written as
            ρ = (I + r⃗ · σ⃗) / 2
        where σ⃗ = (σ_x, σ_y, σ_z) are Pauli matrices.
        |r⃗| = 1 for pure states, |r⃗| < 1 for mixed states.
        The Bloch sphere gives geometric intuition for qubit states.
        """
        rho_q = self.partial_trace([qubit])

        # Pauli matrices
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

        rx = float(np.real(np.trace(rho_q @ sigma_x)))
        ry = float(np.real(np.trace(rho_q @ sigma_y)))
        rz = float(np.real(np.trace(rho_q @ sigma_z)))

        return (rx, ry, rz)

    def __repr__(self) -> str:
        state_type = "pure" if self._is_pure else "mixed"
        return f"QuantumState(num_qubits={self.num_qubits}, type={state_type})"

    def pretty_print(self):
        """Print state in Dirac notation."""
        probs = self.get_probabilities()
        print(f"  {'State':<{self.num_qubits+4}}  {'Amplitude':<20}  {'Probability':<12}")
        print(f"  {'─'*(self.num_qubits+4)}  {'─'*20}  {'─'*12}")

        if self._is_pure:
            sv = self._statevector
        else:
            sv = None

        for k in range(self.dim):
            if probs[k] > 1e-10:
                basis = format(k, f'0{self.num_qubits}b')
                if sv is not None:
                    amp = sv[k]
                    amp_str = f"{amp.real:+.4f}{amp.imag:+.4f}j"
                else:
                    amp_str = "  (mixed state)"
                print(f"  |{basis}⟩  {amp_str:<20}  {probs[k]:.6f}")


def _matrix_sqrt(M: np.ndarray) -> np.ndarray:
    """Compute matrix square root via eigendecomposition."""
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    eigenvalues = np.maximum(eigenvalues, 0)  # Clip negative values from numerical noise
    return eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.conj().T
