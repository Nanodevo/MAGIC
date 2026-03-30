#!/usr/bin/env python3
"""Quick test of the trapped ion simulator."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trapped_ion_sim import TrappedIonSimulator
from trapped_ion_sim.ion_trap import IonTrap, TrapParameters, CALCIUM_40

# Test 1: Basic Bell state
print("Test 1: Bell state")
sim = TrappedIonSimulator(num_qubits=2, species='calcium')
sim.ms(0, 1)
results = sim.run(shots=1000)
print("  Results:", results)

# Test 2: State vector
print("\nTest 2: Statevector")
state = sim.get_statevector()
state.pretty_print()

# Test 3: Trap physics
print("\nTest 3: Trap physics")
trap = IonTrap(CALCIUM_40, TrapParameters(), num_ions=3)
print(f"  Axial freq: {trap.freq_axial_hz/1e6:.4f} MHz")
print(f"  Radial freq: {trap.freq_radial_hz/1e6:.4f} MHz")
print(f"  Ion positions (um): {trap.equilibrium_positions_meters*1e6}")
print(f"  Lamb-Dicke eta: {trap.lamb_dicke_parameter(0, 0):.4f}")

# Test 4: Noisy simulation
print("\nTest 4: Noisy Bell state")
sim_noisy = TrappedIonSimulator(num_qubits=2, species='calcium', noise=True)
sim_noisy.ms(0, 1)
results_noisy = sim_noisy.run(shots=1000)
print("  Results:", results_noisy)

# Test 5: GHZ state
print("\nTest 5: GHZ state (3 qubits)")
sim3 = TrappedIonSimulator(num_qubits=3, species='calcium')
sim3.ghz_state()
results3 = sim3.run(shots=1000)
print("  Results:", results3)

# Test 6: Bloch vector & entropy
print("\nTest 6: State analysis")
state2 = sim.get_statevector()
bv = state2.bloch_vector(0)
print(f"  Bloch vector qubit 0: ({bv[0]:.4f}, {bv[1]:.4f}, {bv[2]:.4f})")
print(f"  Von Neumann entropy: {state2.von_neumann_entropy():.4f}")

print("\nAll tests passed!")
