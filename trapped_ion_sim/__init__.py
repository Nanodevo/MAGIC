"""
Trapped Ion Quantum Computer Simulator
=======================================

A physics-based simulator for trapped-ion quantum computing.

Modules:
    quantum_state  - Hilbert space, state vectors, density matrices
    ion_trap       - Paul trap physics, normal modes, ion chain equilibrium
    gates          - Single- and two-qubit gates via laser-ion interaction
    decoherence    - Noise models: dephasing, spontaneous emission, motional heating
    measurement    - Projective measurement with state-dependent fluorescence
    simulator      - High-level simulation engine tying everything together
"""

from .simulator import TrappedIonSimulator
from .quantum_state import QuantumState
from .ion_trap import IonTrap
from .gates import SingleQubitGate, MolmerSorensenGate
from .measurement import measure

__version__ = "1.0.0"
