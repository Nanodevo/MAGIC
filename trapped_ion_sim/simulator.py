"""
Simulation Engine
=================

Ties together all components into a high-level interface for building
and running trapped-ion quantum circuits.

ARCHITECTURE:
    1. Create a TrappedIonSimulator with a specific ion species and trap
    2. Build a quantum circuit by adding gates
    3. Run the circuit (with optional noise)
    4. Measure and analyze results

This mirrors the actual experimental workflow:
    Load ions → Cool to ground state → Apply gate sequence → Measure fluorescence
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field

from .quantum_state import QuantumState
from .ion_trap import IonTrap, IonSpecies, TrapParameters, CALCIUM_40, YTTERBIUM_171
from .gates import (
    SingleQubitGate, VirtualZGate, MolmerSorensenGate, CNOTGate,
    GateTimingModel, Rx, Ry, Rz, Hadamard, PauliX
)
from .measurement import measure, measure_and_collapse
from .decoherence import TrappedIonDecoherence


@dataclass
class CircuitInstruction:
    """A single instruction in the quantum circuit."""
    gate: object                      # Gate object
    gate_type: str                    # 'single', 'two_qubit', 'virtual_z', 'measure'
    target_qubits: List[int]          # Which qubits are involved
    duration_us: float = 0.0         # Gate duration in microseconds
    label: str = ""                  # Human-readable label


class TrappedIonSimulator:
    """
    High-level simulator for a trapped-ion quantum computer.

    ╔═══════════════════════════════════════════════════════════════════╗
    ║  SIMULATION WORKFLOW                                             ║
    ║                                                                  ║
    ║  1. INITIALIZATION                                               ║
    ║     - Define ion species (Ca⁺, Yb⁺, Be⁺)                       ║
    ║     - Set trap parameters (frequencies, voltages)               ║
    ║     - Compute ion chain equilibrium and normal modes            ║
    ║     - Prepare all qubits in |0⟩ (Doppler + sideband cooling)    ║
    ║                                                                  ║
    ║  2. CIRCUIT CONSTRUCTION                                         ║
    ║     - Add single-qubit rotations  (carrier Rabi pulses)         ║
    ║     - Add entangling MS gates     (sideband transitions)        ║
    ║     - Optionally enable noise model                             ║
    ║                                                                  ║
    ║  3. EXECUTION                                                    ║
    ║     - Apply gates sequentially to quantum state                  ║
    ║     - Insert decoherence between/during gates if noise enabled  ║
    ║     - Track total circuit time                                   ║
    ║                                                                  ║
    ║  4. MEASUREMENT                                                  ║
    ║     - State-dependent fluorescence on each ion                   ║
    ║     - Count photons → determine |0⟩ (bright) or |1⟩ (dark)     ║
    ║     - Repeat many shots for statistics                           ║
    ╚═══════════════════════════════════════════════════════════════════╝

    Example:
        sim = TrappedIonSimulator(num_qubits=2, species='ytterbium')
        sim.h(0)                    # Hadamard on ion 0
        sim.ms(0, 1)               # Entangling gate on ions 0,1
        results = sim.run(shots=1000)
        sim.print_results(results)
    """

    def __init__(self, num_qubits: int,
                 species: str = 'calcium',
                 trap_params: Optional[TrapParameters] = None,
                 noise: bool = False,
                 noise_model: Optional[TrappedIonDecoherence] = None):
        """
        Args:
            num_qubits: Number of ions/qubits.
            species: Ion species ('calcium', 'ytterbium', 'beryllium').
            trap_params: Custom trap parameters (uses defaults if None).
            noise: Enable realistic noise model.
            noise_model: Custom noise parameters (uses species defaults if None).
        """
        self.num_qubits = num_qubits

        # Select ion species
        species_map = {
            'calcium': CALCIUM_40, 'ca': CALCIUM_40, 'ca+': CALCIUM_40,
            'ytterbium': YTTERBIUM_171, 'yb': YTTERBIUM_171, 'yb+': YTTERBIUM_171,
        }
        species_key = species.lower()
        if species_key not in species_map:
            raise ValueError(f"Unknown species '{species}'. Choose from: {list(species_map.keys())}")
        self.ion_species = species_map[species_key]

        # Initialize trap
        if trap_params is None:
            trap_params = TrapParameters()
        self.trap = IonTrap(self.ion_species, trap_params, num_qubits)

        # Circuit storage
        self.circuit: List[CircuitInstruction] = []
        self.timing = GateTimingModel()

        # Noise
        self.noise_enabled = noise
        if noise and noise_model is None:
            self.noise_model = TrappedIonDecoherence(
                t1=self.ion_species.t1_seconds,
                t2=self.ion_species.t2_seconds,
            )
        else:
            self.noise_model = noise_model

        # Phase tracking for virtual Z gates
        self._phase_offsets = np.zeros(num_qubits)

    # ───────────────────────────────────────────────────────────
    # Circuit Construction API
    # ───────────────────────────────────────────────────────────

    def rx(self, theta: float, qubit: int) -> 'TrappedIonSimulator':
        """Add R_x(θ) rotation on a qubit. Physical carrier pulse with φ=0."""
        gate = Rx(theta, qubit)
        self.circuit.append(CircuitInstruction(
            gate=gate, gate_type='single', target_qubits=[qubit],
            duration_us=self.timing.single_qubit_us,
            label=f"Rx({theta:.3f}) q{qubit}"
        ))
        return self

    def ry(self, theta: float, qubit: int) -> 'TrappedIonSimulator':
        """Add R_y(θ) rotation. Physical carrier pulse with φ=π/2."""
        gate = Ry(theta, qubit)
        self.circuit.append(CircuitInstruction(
            gate=gate, gate_type='single', target_qubits=[qubit],
            duration_us=self.timing.single_qubit_us,
            label=f"Ry({theta:.3f}) q{qubit}"
        ))
        return self

    def rz(self, theta: float, qubit: int) -> 'TrappedIonSimulator':
        """Add R_z(θ) rotation. Virtual gate (zero time, perfect fidelity)."""
        gate = Rz(theta, qubit)
        self.circuit.append(CircuitInstruction(
            gate=gate, gate_type='virtual_z', target_qubits=[qubit],
            duration_us=0.0,
            label=f"Rz({theta:.3f}) q{qubit} [virtual]"
        ))
        return self

    def h(self, qubit: int) -> 'TrappedIonSimulator':
        """Add Hadamard gate. Decomposed into native Rabi pulse."""
        gate = Hadamard(qubit)
        self.circuit.append(CircuitInstruction(
            gate=gate, gate_type='single', target_qubits=[qubit],
            duration_us=self.timing.single_qubit_us,
            label=f"H q{qubit}"
        ))
        return self

    def x(self, qubit: int) -> 'TrappedIonSimulator':
        """Add Pauli-X (NOT) gate. π-pulse about X axis."""
        gate = PauliX(qubit)
        self.circuit.append(CircuitInstruction(
            gate=gate, gate_type='single', target_qubits=[qubit],
            duration_us=self.timing.single_qubit_us,
            label=f"X q{qubit}"
        ))
        return self

    def ms(self, qubit0: int, qubit1: int, chi: float = np.pi/4) -> 'TrappedIonSimulator':
        """
        Add Mølmer-Sørensen entangling gate.

        PHYSICS: Applies exp(-i·χ·σ_x⊗σ_x) via shared phonon mode.
        χ = π/4 gives maximally entangling gate (creates Bell states).
        """
        gate = MolmerSorensenGate((qubit0, qubit1), chi=chi)
        self.circuit.append(CircuitInstruction(
            gate=gate, gate_type='two_qubit', target_qubits=[qubit0, qubit1],
            duration_us=self.timing.ms_gate_us,
            label=f"MS({chi:.3f}) q{qubit0},q{qubit1}"
        ))
        return self

    def cnot(self, control: int, target: int) -> 'TrappedIonSimulator':
        """
        Add CNOT gate (decomposed into MS + single-qubit gates).

        WARNING: This is NOT a native gate! It costs 1 MS + 3 single-qubit gates.
        """
        cnot = CNOTGate(control, target)
        self.circuit.append(CircuitInstruction(
            gate=cnot, gate_type='two_qubit',
            target_qubits=[control, target],
            duration_us=self.timing.ms_gate_us + 3 * self.timing.single_qubit_us,
            label=f"CNOT q{control}→q{target}"
        ))
        return self

    def barrier(self) -> 'TrappedIonSimulator':
        """Add a barrier (visual separator, no physical effect)."""
        self.circuit.append(CircuitInstruction(
            gate=None, gate_type='barrier', target_qubits=[],
            duration_us=0.0, label="barrier"
        ))
        return self

    # ───────────────────────────────────────────────────────────
    # Execution
    # ───────────────────────────────────────────────────────────

    def run(self, shots: int = 1024,
            initial_state: Optional[np.ndarray] = None) -> Dict[str, int]:
        """
        Execute the circuit and measure all qubits.

        PHYSICS SEQUENCE:
            1. Initialize all ions to |0⟩ via optical pumping + sideband cooling
            2. Apply gate sequence (laser pulses)
            3. Detect via state-dependent fluorescence

        Args:
            shots: Number of measurement repetitions.
            initial_state: Optional initial statevector (default: |00...0⟩).

        Returns:
            Dictionary of measurement outcomes and counts.
        """
        state = QuantumState(self.num_qubits, initial_state)

        total_time_us = 0.0

        for instruction in self.circuit:
            if instruction.gate_type == 'barrier':
                continue

            if instruction.gate is None:
                continue

            # Apply the ideal gate
            U = instruction.gate.full_matrix(self.num_qubits)
            state.apply_unitary(U)

            # Apply noise if enabled
            if self.noise_enabled and self.noise_model is not None:
                # Gate error
                if instruction.gate_type in ('single', 'two_qubit'):
                    kraus = self.noise_model.gate_error_kraus(
                        instruction.gate_type,
                        instruction.target_qubits,
                        self.num_qubits
                    )
                    state.apply_kraus(kraus)

                # Idle decoherence during gate (on all qubits)
                if instruction.duration_us > 0:
                    duration_s = instruction.duration_us * 1e-6
                    for q in range(self.num_qubits):
                        kraus = self.noise_model.idle_decoherence(
                            duration_s, q, self.num_qubits
                        )
                        state.apply_kraus(kraus)

            total_time_us += instruction.duration_us

        self._last_state = state
        self._total_time_us = total_time_us

        # Measure
        readout_err = self.noise_model.readout_error if (
            self.noise_enabled and self.noise_model) else 0.0
        return measure(state, shots=shots, readout_error=readout_err)

    def get_statevector(self, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Execute circuit without measurement, return final statevector.

        Only works for noise-free simulation (pure state).
        """
        state = QuantumState(self.num_qubits, initial_state)

        for instruction in self.circuit:
            if instruction.gate is None:
                continue
            U = instruction.gate.full_matrix(self.num_qubits)
            state.apply_unitary(U)

        return state

    def get_unitary(self) -> np.ndarray:
        """Compute the total unitary matrix for the circuit."""
        dim = 2 ** self.num_qubits
        U_total = np.eye(dim, dtype=complex)

        for instruction in self.circuit:
            if instruction.gate is None:
                continue
            U = instruction.gate.full_matrix(self.num_qubits)
            U_total = U @ U_total

        return U_total

    # ───────────────────────────────────────────────────────────
    # Analysis & Visualization
    # ───────────────────────────────────────────────────────────

    def print_circuit(self):
        """Print the circuit in a readable format."""
        print("\n  ╔══════════════════════════════════════════════╗")
        print("  ║         TRAPPED ION QUANTUM CIRCUIT          ║")
        print("  ╚══════════════════════════════════════════════╝")
        print(f"  Qubits: {self.num_qubits} ({self.ion_species.name} ions)")
        print(f"  Noise: {'ON' if self.noise_enabled else 'OFF'}")
        print()

        total_time = 0
        for i, inst in enumerate(self.circuit):
            if inst.gate_type == 'barrier':
                print(f"  {'─'*50}")
                continue

            time_str = f"{inst.duration_us:>6.1f} μs" if inst.duration_us > 0 else "  0    μs (virtual)"
            marker = "★" if inst.gate_type == 'two_qubit' else "●"
            print(f"  {marker} Step {i:>3}: {inst.label:<35} [{time_str}]")
            total_time += inst.duration_us

        print(f"\n  Total circuit time: {total_time:.1f} μs ({total_time/1e3:.3f} ms)")
        gate_counts = {}
        for inst in self.circuit:
            gate_counts[inst.gate_type] = gate_counts.get(inst.gate_type, 0) + 1
        print(f"  Gate counts: {gate_counts}")

    def print_results(self, results: Dict[str, int], top_n: int = 16):
        """Pretty-print measurement results."""
        total = sum(results.values())
        sorted_results = sorted(results.items(), key=lambda x: -x[1])

        print(f"\n  ── Measurement Results ({total} shots) ──")
        print(f"  {'Outcome':<{self.num_qubits+4}}  {'Count':>6}  {'Probability':>11}  Bar")
        print(f"  {'─'*(self.num_qubits+4)}  {'─'*6}  {'─'*11}  {'─'*30}")

        for bitstring, count in sorted_results[:top_n]:
            prob = count / total
            bar_len = int(prob * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"  |{bitstring}⟩  {count:>6}  {prob:>11.4f}  {bar}")

        if len(sorted_results) > top_n:
            print(f"  ... and {len(sorted_results) - top_n} more outcomes")

    def reset(self):
        """Clear the circuit (start a new experiment)."""
        self.circuit.clear()
        self._phase_offsets = np.zeros(self.num_qubits)
        return self

    # ───────────────────────────────────────────────────────────
    # Predefined Circuit Patterns
    # ───────────────────────────────────────────────────────────

    def bell_state(self, qubit0: int = 0, qubit1: int = 1,
                    bell_type: str = 'phi+') -> 'TrappedIonSimulator':
        """
        Prepare a Bell state.

        PHYSICS:
            Bell states are maximally entangled two-qubit states:
                |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
                |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
                |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
                |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2

            In trapped ions, the most natural is |Φ⁺⟩, created by
            a single MS gate on |00⟩:
                MS(π/4)|00⟩ = (|00⟩ + i|11⟩)/√2

            This was first demonstrated by Turchette et al. (NIST, 1998).
        """
        if bell_type == 'phi+':
            self.ms(qubit0, qubit1)
        elif bell_type == 'phi-':
            self.ms(qubit0, qubit1)
            self.rz(np.pi, qubit0)
        elif bell_type == 'psi+':
            self.x(qubit0)
            self.ms(qubit0, qubit1)
        elif bell_type == 'psi-':
            self.x(qubit0)
            self.ms(qubit0, qubit1)
            self.rz(np.pi, qubit0)
        return self

    def ghz_state(self, qubits: Optional[List[int]] = None) -> 'TrappedIonSimulator':
        """
        Prepare a GHZ (Greenberger-Horne-Zeilinger) state.

        PHYSICS:
            |GHZ⟩ = (|00...0⟩ + |11...1⟩)/√2
            This is a maximally entangled N-qubit state.

            In trapped ions with a global MS gate:
                1. Apply R_y(π/2) on first ion
                2. Apply MS gates to entangle each subsequent ion

            GHZ states are important for:
                - Quantum error correction
                - Quantum metrology (Heisenberg-limited sensing)
                - Tests of quantum nonlocality
        """
        if qubits is None:
            qubits = list(range(self.num_qubits))

        if len(qubits) < 2:
            raise ValueError("GHZ state requires at least 2 qubits")

        self.h(qubits[0])
        for i in range(1, len(qubits)):
            self.cnot(qubits[0], qubits[i])
        return self

    def quantum_teleportation(self) -> 'TrappedIonSimulator':
        """
        Quantum teleportation circuit (requires 3 qubits).

        PHYSICS:
            Teleport the state of qubit 0 to qubit 2 using shared
            entanglement (Bell pair on qubits 1,2) and classical
            communication (measurements + corrections).

            Steps:
            1. Create Bell pair between qubits 1 and 2
            2. Bell measurement on qubits 0 and 1
            3. Conditional corrections on qubit 2

            First demonstrated with trapped ions by Riebe et al.
            and Barrett et al. (2004).
        """
        if self.num_qubits < 3:
            raise ValueError("Teleportation requires at least 3 qubits")

        # Create Bell pair (qubits 1,2)
        self.h(1)
        self.cnot(1, 2)
        self.barrier()

        # Bell measurement (qubits 0,1)
        self.cnot(0, 1)
        self.h(0)

        return self
