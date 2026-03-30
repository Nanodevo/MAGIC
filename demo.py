#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════
  TRAPPED ION QUANTUM COMPUTER SIMULATOR — Interactive Tutorial
═══════════════════════════════════════════════════════════════════════

  This demo walks you through the physics of trapped-ion quantum
  computing step by step, with working simulations at each stage.

  Run with:  python demo.py
═══════════════════════════════════════════════════════════════════════
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trapped_ion_sim import TrappedIonSimulator, QuantumState
from trapped_ion_sim.ion_trap import IonTrap, TrapParameters, CALCIUM_40, YTTERBIUM_171
from trapped_ion_sim.gates import Rx, Ry, Rz, Hadamard, MolmerSorensenGate
from trapped_ion_sim.decoherence import TrappedIonDecoherence
from trapped_ion_sim.measurement import measure, expectation_value


def section(title):
    print(f"\n{'═'*70}")
    print(f"  {title}")
    print(f"{'═'*70}\n")


def pause():
    input("  [Press Enter to continue...]\n")


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║     ⚛  TRAPPED ION QUANTUM COMPUTER SIMULATOR  ⚛                ║
    ║                                                                  ║
    ║     An interactive exploration of the physics behind             ║
    ║     one of the leading quantum computing platforms               ║
    ║                                                                  ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    pause()

    # ═══════════════════════════════════════════════════════════
    # PART 1: THE ION TRAP
    # ═══════════════════════════════════════════════════════════
    section("PART 1: THE ION TRAP — Confining Atoms with Electric Fields")

    print("""
  HOW DO YOU TRAP A SINGLE ATOM?

  You can't use a static electric field cage — Earnshaw's theorem says
  it's impossible to create a stable 3D electrostatic trap for charged
  particles. Instead, we use a Paul trap (Nobel Prize 1989).

  The trick: use rapidly oscillating (radio-frequency) electric fields
  that create a time-averaged "pseudopotential" — like balancing a ball
  on a vibrating saddle.

        ╭─── RF electrodes ───╮
        │   +     -     +     │
        │      ●  ●  ●       │   ← ions trapped along axis
        │   -     +     -     │
        ╰─────────────────────╯
          + DC endcaps on sides → axial confinement

  Let's set up a trap with ⁴⁰Ca⁺ ions (a common choice):
    """)

    # Create a calcium ion trap
    trap_params = TrapParameters(
        rf_voltage=200.0,           # 200 V RF amplitude
        rf_frequency_hz=30e6,       # 30 MHz drive frequency
        r0_meters=200e-6,           # 200 μm electrode distance
        endcap_voltage=5.0,         # 5 V on endcaps
        z0_meters=2.0e-3,           # 2 mm endcap separation
    )

    trap = IonTrap(CALCIUM_40, trap_params, num_ions=5)
    trap.print_trap_info()

    pause()

    # ═══════════════════════════════════════════════════════════
    # PART 2: QUBITS FROM IONS
    # ═══════════════════════════════════════════════════════════
    section("PART 2: QUBITS — Two Energy Levels of a Single Ion")

    print("""
  Each trapped ion is a natural qubit! We use two internal energy levels:

  For ⁴⁰Ca⁺ (optical qubit):

    Energy
      ↑
      │   |1⟩ = D₅/₂  ─────────  (metastable, lifetime 1.17 s)
      │                    ↕ 729 nm laser (qubit transition)
      │   |0⟩ = S₁/₂  ─────────  (ground state)
      │

  For ¹⁷¹Yb⁺ (hyperfine qubit):

    Energy
      ↑
      │   |1⟩ = ²S₁/₂ F=1  ───  ↕ 12.6 GHz microwave
      │   |0⟩ = ²S₁/₂ F=0  ───  (both in ground manifold!)
      │

  Hyperfine qubits have MUCH longer coherence times (seconds vs milliseconds)
  because both states are in the ground electronic state.

  Let's initialize a single qubit in |0⟩ and display it:
    """)

    state = QuantumState(1)
    print("  Initial state: |0⟩")
    state.pretty_print()
    bv = state.bloch_vector(0)
    print(f"\n  Bloch vector: (x={bv[0]:.3f}, y={bv[1]:.3f}, z={bv[2]:.3f})")
    print(f"  → Points to North pole of Bloch sphere (pure |0⟩)")

    pause()

    # ═══════════════════════════════════════════════════════════
    # PART 3: SINGLE-QUBIT GATES
    # ═══════════════════════════════════════════════════════════
    section("PART 3: SINGLE-QUBIT GATES — Rabi Oscillations")

    print("""
  To manipulate a qubit, we shine a resonant laser on the ion.
  This drives Rabi oscillations between |0⟩ and |1⟩:

         Population
    |1⟩  ┤ ╭──╮    ╭──╮    ╭──╮
         │╱    ╲  ╱    ╲  ╱    ╲
    |0⟩  ┤      ╲╱      ╲╱      ╲───
         └──────────────────────────── time
              π     2π     3π

  The qubit state after a pulse of duration t:
    |ψ(t)⟩ = cos(Ωt/2)|0⟩ + i·e^{iφ}·sin(Ωt/2)|1⟩

  where Ω is the Rabi frequency (∝ laser intensity) and φ is the laser phase.

  Key pulse areas:
    π/2 pulse: creates superposition  |0⟩ → (|0⟩ + i|1⟩)/√2
    π pulse:   flips the qubit        |0⟩ → i|1⟩
    2π pulse:  full rotation           |0⟩ → -|0⟩ → |0⟩
    """)

    print("  DEMONSTRATION: Rabi oscillation (varying pulse area)")
    print(f"  {'Pulse area θ':>14}  {'P(|0⟩)':>8}  {'P(|1⟩)':>8}  Bloch z")
    print(f"  {'─'*14}  {'─'*8}  {'─'*8}  {'─'*10}")

    for theta_frac in [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]:
        theta = theta_frac * np.pi
        state = QuantumState(1)
        gate = Rx(theta, 0)
        state.apply_unitary(gate.matrix)
        probs = state.get_probabilities()
        bz = state.bloch_vector(0)[2]
        label = ""
        if theta_frac == 0.5: label = "  ← π/2 (superposition)"
        if theta_frac == 1.0: label = "  ← π (bit flip)"
        if theta_frac == 2.0: label = "  ← 2π (full cycle)"
        print(f"  {theta_frac:>10.2f}·π  {probs[0]:>8.4f}  {probs[1]:>8.4f}  {bz:>+.4f}{label}")

    pause()

    print("""
  THE VIRTUAL Z GATE — a trapped-ion superpower!

  Z rotations DON'T need a physical laser pulse. Instead, we just
  update the reference frame (phase) for all future pulses.

  This means Rz gates are:
    ✓ Instantaneous (zero time)
    ✓ Perfect (zero error)
    ✓ Free (no laser needed)

  Combined with Rx and Ry, we have universal single-qubit control.
    """)

    pause()

    # ═══════════════════════════════════════════════════════════
    # PART 4: ENTANGLEMENT VIA MØLMER-SØRENSEN GATE
    # ═══════════════════════════════════════════════════════════
    section("PART 4: ENTANGLEMENT — The Mølmer-Sørensen Gate")

    print("""
  The Mølmer-Sørensen (MS) gate is THE key innovation for trapped-ion QC.
  It creates entanglement between two ions using their shared motion.

  THE MECHANISM (simplified):

  1. Two laser beams create a spin-dependent force on the ions:
     - |↑↑⟩ and |↓↓⟩ states push the ions in OPPOSITE directions
     - |↑↓⟩ and |↓↑⟩ states: forces cancel (no push)

  2. The ions' shared vibrational mode traces a LOOP in phase space:

              p (momentum)
              ↑     ╭──╮
              │    ╱    ╲     ← |↑↑⟩ loop
              │   ╱      ╲
              ●──╱────────╲── → x (position)
              │   ╲      ╱
              │    ╲    ╱     ← |↓↓⟩ loop (opposite)
              │     ╰──╯

  3. After one complete loop, the motion returns to its original state
     BUT the spins acquire a GEOMETRIC PHASE (Berry phase) proportional
     to the enclosed area.

  4. This geometric phase entangles the qubits!

  The MS gate unitary: U = exp(-i·(π/4)·σ_x⊗σ_x)
    """)

    print("  DEMONSTRATION: Creating a Bell state with one MS gate\n")

    sim = TrappedIonSimulator(num_qubits=2, species='calcium')
    sim.ms(0, 1)  # Single MS gate
    sim.print_circuit()

    print("\n  Running circuit...")
    state = sim.get_statevector()
    print("\n  Final state:")
    state.pretty_print()

    print("\n  Measuring 10000 times:")
    results = sim.run(shots=10000)
    sim.print_results(results)

    print("""
  Notice: we get |00⟩ and |11⟩ with equal probability, but NEVER
  |01⟩ or |10⟩. The qubits are perfectly correlated — this is
  ENTANGLEMENT! Einstein called it "spooky action at a distance."

  Technically this is the Bell state (|00⟩ + i|11⟩)/√2.
    """)

    pause()

    # ═══════════════════════════════════════════════════════════
    # PART 5: BUILDING CIRCUITS
    # ═══════════════════════════════════════════════════════════
    section("PART 5: QUANTUM CIRCUITS — Composing Gates")

    print("""
  With single-qubit rotations + MS gate, we have a UNIVERSAL gate set:
  any quantum computation can be built from these primitives.

  Example: CNOT gate (controlled-NOT) decomposition:

    ── Ry(-π/2) ── MS ── Rx(-π/2) ── Ry(π/2) ──   (control)
                   MS ── Rx(-π/2) ──────────────    (target)

  Note: CNOT costs 1 MS gate + 3 single-qubit gates.
  The MS gate is the expensive part (~100 μs vs ~5 μs).

  Let's build a 3-qubit GHZ state: (|000⟩ + |111⟩)/√2
    """)

    sim = TrappedIonSimulator(num_qubits=3, species='calcium')
    sim.ghz_state()
    sim.print_circuit()

    print("\n  Running circuit...")
    results = sim.run(shots=10000)
    sim.print_results(results)

    print("""
  Again, only |000⟩ and |111⟩ appear — all three qubits are
  entangled in a GHZ state. If you measure one, you instantly
  know the state of all others!
    """)

    pause()

    # ═══════════════════════════════════════════════════════════
    # PART 6: DECOHERENCE
    # ═══════════════════════════════════════════════════════════
    section("PART 6: DECOHERENCE — The Enemy of Quantum Computing")

    print("""
  Real quantum computers aren't perfect. The quantum state gradually
  loses its "quantumness" through decoherence — unwanted interaction
  with the environment.

  TWO KEY TIMESCALES:

  T₁ (relaxation): |1⟩ spontaneously decays to |0⟩
     → like a ball rolling downhill
     → Ca⁺: ~1.17 s, Yb⁺: essentially infinite

  T₂ (dephasing): the phase between |0⟩ and |1⟩ randomizes
     → like a spinning top wobbling
     → Ca⁺: ~10 ms, Yb⁺: ~1.5 s
     → always T₂ ≤ 2·T₁

  Let's compare IDEAL vs NOISY simulation of a Bell state:
    """)

    print("  ── IDEAL (no noise) ──")
    sim_ideal = TrappedIonSimulator(num_qubits=2, species='calcium', noise=False)
    sim_ideal.ms(0, 1)
    results_ideal = sim_ideal.run(shots=10000)
    sim_ideal.print_results(results_ideal)

    print("\n  ── NOISY (realistic Ca⁺ parameters) ──")
    sim_noisy = TrappedIonSimulator(num_qubits=2, species='calcium', noise=True)
    sim_noisy.ms(0, 1)
    results_noisy = sim_noisy.run(shots=10000)
    sim_noisy.print_results(results_noisy)

    print("""
  With noise, we see small contributions from |01⟩ and |10⟩ — these
  are ERRORS caused by decoherence and imperfect gates.

  Error budget for this circuit:
    """)
    sim_noisy.noise_model.print_error_budget(circuit_depth=1, num_two_qubit_gates=1)

    pause()

    # ═══════════════════════════════════════════════════════════
    # PART 7: COMPARING ION SPECIES
    # ═══════════════════════════════════════════════════════════
    section("PART 7: ION SPECIES COMPARISON — Ca⁺ vs Yb⁺")

    print("""
  Different ion species have different trade-offs:

  ┌────────────────┬──────────────────┬──────────────────┐
  │                │   ⁴⁰Ca⁺          │   ¹⁷¹Yb⁺         │
  ├────────────────┼──────────────────┼──────────────────┤
  │ Qubit type     │ Optical (729nm)  │ Hyperfine (μwave) │
  │ T₁             │ 1.17 s           │ ~10¹⁰ s (∞)      │
  │ T₂             │ ~10 ms           │ ~1.5 s            │
  │ Gate method     │ Direct laser     │ Raman beams       │
  │ Wavelength      │ 729 nm (red)     │ 369.5 nm (UV)     │
  │ Advantage       │ Simple optics    │ Long coherence     │
  │ Disadvantage    │ Short T₂         │ Needs UV lasers    │
  └────────────────┴──────────────────┴──────────────────┘

  Companies using trapped ions:
    IonQ         → ¹⁷¹Yb⁺
    Quantinuum   → ¹³⁸Ba⁺ (similar to Ca⁺)
    AQT          → ⁴⁰Ca⁺
    eleQtron     → ⁴⁰Ca⁺ (microwave-driven)
    """)

    print("  ── Ytterbium trap configuration ──\n")
    trap_yb = IonTrap(YTTERBIUM_171, TrapParameters(), num_ions=3)
    trap_yb.print_trap_info()

    pause()

    # ═══════════════════════════════════════════════════════════
    # PART 8: ADVANCED — Quantum Algorithms
    # ═══════════════════════════════════════════════════════════
    section("PART 8: PUTTING IT TOGETHER — Quantum Algorithms")

    print("""
  Let's run a simple quantum algorithm: Deutsch-Jozsa.

  PROBLEM: Given a function f: {0,1} → {0,1}, determine if f is
  "constant" (same output for all inputs) or "balanced" (different
  outputs for different inputs) — with a SINGLE query.

  Classically: need 2 queries. Quantum: need only 1!

  Circuit for balanced f (f(x) = x):
    q0: |0⟩ ── H ── CNOT ── H ── Measure
    q1: |1⟩ ── H ── CNOT ── H ── Measure

  If q0 measures |1⟩ → f is balanced
  If q0 measures |0⟩ → f is constant
    """)

    # Deutsch-Jozsa for balanced function
    sim = TrappedIonSimulator(num_qubits=2, species='calcium')

    # Prepare |01⟩ (second qubit in |1⟩)
    sim.x(1)
    sim.barrier()

    # Hadamard both
    sim.h(0)
    sim.h(1)
    sim.barrier()

    # Oracle for balanced f(x) = x: CNOT
    sim.cnot(0, 1)
    sim.barrier()

    # Hadamard on query qubit
    sim.h(0)

    sim.print_circuit()

    print("\n  Running Deutsch-Jozsa algorithm...")
    results = sim.run(shots=10000)
    sim.print_results(results)

    # Check the first qubit
    ones_count = sum(v for k, v in results.items() if k[0] == '1')
    zero_count = sum(v for k, v in results.items() if k[0] == '0')
    print(f"\n  First qubit: |0⟩ count = {zero_count}, |1⟩ count = {ones_count}")
    if ones_count > zero_count:
        print("  → Function is BALANCED ✓ (correct!)")
    else:
        print("  → Function is CONSTANT")

    pause()

    # ═══════════════════════════════════════════════════════════
    # PART 9: STATE ANALYSIS
    # ═══════════════════════════════════════════════════════════
    section("PART 9: QUANTUM STATE ANALYSIS")

    print("""
  Let's examine quantum states in detail — entanglement, entropy,
  and the Bloch sphere representation.
    """)

    # Create and analyze a Bell state
    sim = TrappedIonSimulator(num_qubits=2, species='calcium')
    sim.ms(0, 1)
    state = sim.get_statevector()

    print("  Bell state (|00⟩ + i|11⟩)/√2:")
    state.pretty_print()

    entropy = state.von_neumann_entropy()
    print(f"\n  Von Neumann entropy S(ρ) = {entropy:.4f}")
    print(f"  (0 = pure state, {state.num_qubits} = maximally mixed)")

    for q in range(2):
        bv = state.bloch_vector(q)
        length = np.sqrt(sum(x**2 for x in bv))
        print(f"\n  Ion {q} Bloch vector: ({bv[0]:+.4f}, {bv[1]:+.4f}, {bv[2]:+.4f})")
        print(f"  |r| = {length:.4f} {'(pure)' if length > 0.99 else '(MIXED — entangled with other ion!)'}")

    # Fidelity comparison
    print("\n  ── Fidelity Analysis ──")
    state_ideal = sim.get_statevector()

    sim_noisy = TrappedIonSimulator(num_qubits=2, species='calcium', noise=True)
    sim_noisy.ms(0, 1)
    # Run to get noisy state
    sim_noisy.run(shots=1)
    state_noisy = sim_noisy._last_state

    fid = state_ideal.fidelity(state_noisy)
    print(f"  Fidelity(ideal, noisy) = {fid:.6f}")
    print(f"  Infidelity = {1-fid:.6f}")

    pause()

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════
    section("SUMMARY: How a Trapped-Ion Quantum Computer Works")

    print("""
  ┌─────────────────────────────────────────────────────────────────────┐
  │                THE COMPLETE PICTURE                                │
  │                                                                    │
  │  1. TRAP IONS                                                      │
  │     • Load atoms into a Paul trap (RF electric fields)             │
  │     • Cool to near absolute zero via laser cooling                 │
  │     • Ions form a linear crystal, spaced ~5 μm apart              │
  │                                                                    │
  │  2. INITIALIZE QUBITS                                              │
  │     • Optical pumping → all ions in |0⟩                            │
  │     • Sideband cooling → motional ground state |n=0⟩               │
  │                                                                    │
  │  3. APPLY GATES                                                    │
  │     • Single-qubit: focused laser → Rabi oscillations (~5 μs)     │
  │     • Two-qubit: Mølmer-Sørensen gate via shared phonons (~100 μs)│
  │     • Z rotations: virtual (free, perfect)                         │
  │                                                                    │
  │  4. MEASURE                                                        │
  │     • Shine detection laser on each ion                            │
  │     • |0⟩ glows (fluorescence), |1⟩ is dark                       │
  │     • Count photons → determine qubit state (>99.99% fidelity)    │
  │                                                                    │
  │  5. REPEAT                                                         │
  │     • Run many times for statistics (quantum → classical output)   │
  │                                                                    │
  │  ADVANTAGES of trapped ions:                                       │
  │     ✓ All-to-all connectivity (any ion can talk to any other)     │
  │     ✓ Highest gate fidelities of any platform (>99.9%)            │
  │     ✓ Long coherence times (seconds for hyperfine qubits)         │
  │     ✓ Identical qubits (atoms of the same species are identical)  │
  │                                                                    │
  │  CHALLENGES:                                                       │
  │     ✗ Slow gates compared to superconducting qubits (~100× slower)│
  │     ✗ Scaling beyond ~50 ions in one trap is hard                 │
  │     ✗ Complex laser systems required                               │
  │     ✗ Motional heating limits gate quality                        │
  │                                                                    │
  │  CURRENT STATE OF THE ART (as of 2025):                           │
  │     • IonQ: 35+ qubits, algorithmic qubits via QEC               │
  │     • Quantinuum: 56 qubits (H2), highest quantum volume          │
  │     • >99.9% two-qubit gate fidelity demonstrated                 │
  │     • First logical qubit demonstrations with real error correction│
  └─────────────────────────────────────────────────────────────────────┘
    """)

    print("  Thank you for exploring trapped-ion quantum computing! ⚛\n")


if __name__ == "__main__":
    main()
