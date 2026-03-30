# ⚛ Trapped Ion Quantum Computer Simulator

A physics-based simulator for trapped-ion quantum computing with detailed explanations of the underlying physics at every layer.

## Quick Start

```bash
pip install numpy scipy

# Run the interactive tutorial
python demo.py

# Or use as a library
python -c "
from trapped_ion_sim import TrappedIonSimulator

sim = TrappedIonSimulator(num_qubits=2, species='calcium')
sim.ms(0, 1)  # Mølmer-Sørensen entangling gate → Bell state
results = sim.run(shots=1000)
sim.print_results(results)
"
```

## How a Trapped-Ion Quantum Computer Works

### 1. Trapping Ions (Paul Trap)

Individual atoms are ionized (one electron removed) and confined in a **Paul trap** — a device that uses rapidly oscillating radio-frequency (RF) electric fields to create a stable trapping potential.

```
        ╭─── RF electrodes ───╮
        │   +     -     +     │
        │      ●  ●  ●       │   ← ions confined along the axis
        │   -     +     -     │
        ╰─────────────────────╯
            DC endcaps → axial confinement
```

**Key physics:**
- **Earnshaw's theorem** forbids static electric field trapping, so we use time-varying fields
- The **Mathieu equation** governs ion stability: the parameter $q = \frac{2eV_{RF}}{m\Omega_{RF}^2 r_0^2}$ must satisfy $q < 0.908$
- Ions oscillate at the **secular frequency** $\omega_{sec} \approx \frac{q}{2\sqrt{2}} \Omega_{RF}$ (typically 1–5 MHz)
- Multiple ions form a **linear crystal** due to Coulomb repulsion vs. trap confinement

### 2. Encoding Qubits

Each ion's two internal energy levels define a qubit:

| Species | Qubit Type | States | Transition | T₂ |
|---------|-----------|--------|------------|-----|
| ⁴⁰Ca⁺ | Optical | S₁/₂ ↔ D₅/₂ | 729 nm laser | ~10 ms |
| ¹⁷¹Yb⁺ | Hyperfine | ²S₁/₂ F=0 ↔ F=1 | 12.6 GHz μwave | ~1.5 s |
| ⁹Be⁺ | Hyperfine | ²S₁/₂ F=1 ↔ F=2 | 1.25 GHz μwave | ~1.5 s |

### 3. Single-Qubit Gates (Rabi Oscillations)

A resonant laser drives **Rabi oscillations** between |0⟩ and |1⟩:

$$|\psi(t)\rangle = \cos\frac{\Omega t}{2}|0\rangle + i e^{i\phi}\sin\frac{\Omega t}{2}|1\rangle$$

where $\Omega$ is the **Rabi frequency** (proportional to laser intensity) and $\phi$ is the laser phase.

The rotation operator:
$$R(\theta, \phi) = \cos\frac{\theta}{2} I - i\sin\frac{\theta}{2}(\cos\phi\;\sigma_x + \sin\phi\;\sigma_y)$$

- **π/2 pulse** ($\theta = \pi/2$): creates superposition
- **π pulse** ($\theta = \pi$): flips the qubit (NOT gate)
- **R_z gate**: implemented virtually by shifting the laser phase reference (zero time, zero error!)

### 4. Two-Qubit Gates (Mølmer-Sørensen Gate)

The **Mølmer-Sørensen (MS) gate** creates entanglement using shared motional modes as a quantum bus:

1. Two laser beams drive **spin-dependent forces** on the ions
2. The shared vibrational mode traces a **loop in phase space** (different for different spin states)
3. After one complete loop, the motion **disentangles** from the spins
4. The spins acquire a **geometric phase** (Berry phase) ∝ enclosed area

$$U_{MS} = \exp\left(-i\frac{\pi}{4}\sigma_x \otimes \sigma_x\right)$$

This directly creates Bell states: $|00\rangle \xrightarrow{MS} \frac{|00\rangle + i|11\rangle}{\sqrt{2}}$

### 5. Measurement (Electron Shelving)

State detection uses **state-dependent fluorescence**:

```
    P₁/₂  ─────  (short-lived)
     ↑  ↓ cycling
|0⟩ S₁/₂  ─────  ← detection laser → BRIGHT (fluorescence) ☀
    
|1⟩ D₅/₂  ─────  (metastable, "shelved") → DARK (no fluorescence) ●
```

Fidelity: > 99.99% single-shot readout!

### 6. Decoherence

Quantum states degrade through interaction with the environment:

- **T₁ (amplitude damping)**: spontaneous decay |1⟩ → |0⟩
- **T₂ (dephasing)**: random phase noise from magnetic field fluctuations
- **Motional heating**: electric field noise heats the ion's motion
- **Gate errors**: laser intensity/phase noise, off-resonant coupling

Modeled via the **Lindblad master equation**:
$$\frac{d\rho}{dt} = -\frac{i}{\hbar}[H, \rho] + \sum_k \gamma_k \left(L_k \rho L_k^\dagger - \frac{1}{2}\{L_k^\dagger L_k, \rho\}\right)$$

## Architecture

```
trapped_ion_sim/
├── __init__.py          # Package entry point
├── quantum_state.py     # Hilbert space, density matrices, Bloch vectors
├── ion_trap.py          # Paul trap physics, normal modes, Lamb-Dicke params
├── gates.py             # Single-qubit (Rabi) and two-qubit (MS) gates
├── decoherence.py       # T₁, T₂, gate errors, Kraus operators
├── measurement.py       # State-dependent fluorescence, Born rule
└── simulator.py         # High-level circuit builder and runner

demo.py                  # Interactive physics tutorial
```

## API Reference

### Creating a Simulator

```python
from trapped_ion_sim import TrappedIonSimulator

# Simple (defaults to Ca⁺ ions, no noise)
sim = TrappedIonSimulator(num_qubits=3, species='calcium')

# With noise
sim = TrappedIonSimulator(num_qubits=2, species='ytterbium', noise=True)

# Custom trap parameters
from trapped_ion_sim.ion_trap import TrapParameters
trap = TrapParameters(rf_voltage=300, rf_frequency_hz=40e6)
sim = TrappedIonSimulator(num_qubits=4, trap_params=trap)
```

### Building Circuits

```python
# Single-qubit gates (physical laser pulses)
sim.rx(theta, qubit)    # X rotation (carrier pulse, φ=0)
sim.ry(theta, qubit)    # Y rotation (carrier pulse, φ=π/2)
sim.rz(theta, qubit)    # Z rotation (virtual — free!)
sim.h(qubit)            # Hadamard
sim.x(qubit)            # Pauli-X (π pulse)

# Two-qubit gates
sim.ms(qubit0, qubit1)  # Mølmer-Sørensen (native!)
sim.cnot(control, target)  # CNOT (decomposed: 1 MS + 3 single-qubit)

# Pre-built circuits
sim.bell_state(0, 1)    # Create Bell state
sim.ghz_state()         # Create GHZ state
```

### Running and Measuring

```python
# Sample measurements
results = sim.run(shots=10000)
sim.print_results(results)

# Get the exact quantum state (no measurement)
state = sim.get_statevector()
state.pretty_print()

# Analyze the state
print(state.von_neumann_entropy())
print(state.bloch_vector(qubit=0))
print(state.fidelity(other_state))
```

### Exploring Trap Physics

```python
from trapped_ion_sim.ion_trap import IonTrap, TrapParameters, CALCIUM_40

trap = IonTrap(CALCIUM_40, TrapParameters(), num_ions=5)
trap.print_trap_info()

# Access physical quantities
print(trap.equilibrium_positions_meters)  # Ion chain positions
print(trap.mode_frequencies)              # Normal mode frequencies
print(trap.lamb_dicke_parameter(mode=0, ion_index=0))  # η parameter
print(trap.motional_heating_rate())       # Phonons/sec
```

## Key Physics Concepts Explained in Code

| Concept | Where to Find It |
|---------|------------------|
| Paul trap pseudopotential | `ion_trap.py` — `_compute_trap_frequencies()` |
| Mathieu equation stability | `ion_trap.py` — `q_parameter` |
| Ion chain equilibrium | `ion_trap.py` — `_compute_equilibrium_positions()` |
| Normal modes (phonons) | `ion_trap.py` — `_compute_normal_modes()` |
| Lamb-Dicke parameter | `ion_trap.py` — `lamb_dicke_parameter()` |
| Rabi oscillations | `gates.py` — `SingleQubitGate` |
| Virtual Z gate | `gates.py` — `VirtualZGate` |
| Mølmer-Sørensen gate | `gates.py` — `MolmerSorensenGate` |
| Geometric phase | `gates.py` — MS gate docstring |
| Born rule | `measurement.py` — `measure()` |
| Wavefunction collapse | `measurement.py` — `measure_and_collapse()` |
| Electron shelving | `measurement.py` — module docstring |
| Amplitude damping (T₁) | `decoherence.py` — `amplitude_damping_kraus()` |
| Phase damping (T₂) | `decoherence.py` — `phase_damping_kraus()` |
| Kraus operators | `decoherence.py` — all channel functions |
| Density matrices | `quantum_state.py` — `QuantumState` |
| Partial trace | `quantum_state.py` — `partial_trace()` |
| Bloch sphere | `quantum_state.py` — `bloch_vector()` |
| Von Neumann entropy | `quantum_state.py` — `von_neumann_entropy()` |
| Quantum fidelity | `quantum_state.py` — `fidelity()` |

## References

- Bruzewicz, C. D. et al., "Trapped-ion quantum computing: Progress and challenges," *Applied Physics Reviews* 6, 021314 (2019)
- Häffner, H., Roos, C. F., & Blatt, R., "Quantum computing with trapped ions," *Physics Reports* 469, 155–203 (2008)
- Sørensen, A. & Mølmer, K., "Quantum computation with ions in thermal motion," *Physical Review Letters* 82, 1971 (1999)
- Ballance, C. J. et al., "High-fidelity quantum logic gates using trapped-ion hyperfine qubits," *Physical Review Letters* 117, 060504 (2016)
