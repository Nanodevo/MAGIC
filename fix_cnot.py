"""Find correct CNOT decomposition using MS gate."""
import numpy as np
from scipy.linalg import expm

sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

def Rx(t): return expm(-1j * t / 2 * sx)
def Ry(t): return expm(-1j * t / 2 * sy)
def Rz(t): return expm(-1j * t / 2 * sz)

# MS gate: exp(-i pi/4 sigma_x x sigma_x)
XX = np.kron(sx, sx)
MS = expm(-1j * np.pi / 4 * XX)

def check_cnot(U, label):
    basis = ['|00>', '|01>', '|10>', '|11>']
    print(f'{label}:')
    for i in range(4):
        out = U @ np.eye(4)[i]
        probs = np.abs(out) ** 2
        result = [(f'{probs[j]:.3f}', basis[j]) for j in range(4) if probs[j] > 0.01]
        print(f'  {basis[i]} -> {result}')
    print()

# Expected CNOT:
# |00> -> |00>, |01> -> |01>, |10> -> |11>, |11> -> |10>

# Decomposition from Maslov 2017 / IonQ:
# CNOT(0,1) = Ry(pi/2)_1 . MS . Rx(-pi/2)_0 . Rx(-pi/2)_1 . Ry(-pi/2)_1
U1 = np.kron(I2, Ry(np.pi/2)) @ MS @ np.kron(Rx(-np.pi/2), Rx(-np.pi/2)) @ np.kron(I2, Ry(-np.pi/2))
check_cnot(U1, "Try 1: Ry_t MS Rx_c Rx_t Ry_t^dag")

# Alternative: swap control/target roles
U2 = np.kron(Ry(np.pi/2), I2) @ MS @ np.kron(Rx(-np.pi/2), Rx(-np.pi/2)) @ np.kron(Ry(-np.pi/2), I2)
check_cnot(U2, "Try 2: Ry_c MS Rx_c Rx_t Ry_c^dag")

# From Sørensen & Mølmer: CNOT = (I x Ry(-pi/2)) . MS . (Rz(-pi) x Rx(pi/2))
# or CNOT = local phases  . MS . local rotations
U3 = np.kron(I2, Ry(-np.pi/2)) @ MS @ np.kron(Rz(-np.pi), Rx(np.pi/2))
check_cnot(U3, "Try 3: Ry(-pi/2)_t MS Rz(-pi)_c Rx(pi/2)_t")

# From Kirchmair et al: CNOT = Ry(pi/2)_t . XX(pi/4) . Ry(-pi/2)_c . Ry(-pi/2)_t
U4 = np.kron(I2, Ry(np.pi/2)) @ MS @ np.kron(Ry(-np.pi/2), Ry(-np.pi/2))
check_cnot(U4, "Try 4: Ry(pi/2)_t MS Ry(-pi/2)_c Ry(-pi/2)_t")

# Simple: Ry(-pi/2)_c . MS(pi/4) . Ry(pi/2)_c . Ry(pi/2)_t  
U5 = np.kron(Ry(-np.pi/2), Ry(np.pi/2)) @ MS @ np.kron(Ry(np.pi/2), I2)
check_cnot(U5, "Try 5")

# Brute force: try all combos of Ry(+/-pi/2) pre/post on both qubits
from itertools import product
angles = [-np.pi/2, np.pi/2]
found = False
for a1, a2, b1, b2 in product(angles, repeat=4):
    U = np.kron(Ry(b1), Ry(b2)) @ MS @ np.kron(Ry(a1), Ry(a2))
    # Check CNOT action
    ok = True
    expected = [(0, 0), (1, 1), (3, 2), (2, 3)]  # in->out index
    for inp, outp in expected:
        out = U @ np.eye(4)[inp]
        if np.abs(out[outp])**2 < 0.95:
            ok = False
            break
    if ok:
        print(f"FOUND with Ry: pre=({a1/np.pi:.2f}pi, {a2/np.pi:.2f}pi), "
              f"post=({b1/np.pi:.2f}pi, {b2/np.pi:.2f}pi)")
        check_cnot(U, "Found CNOT")
        found = True

if not found:
    # Try with Rx pre/post
    for a1, a2, b1, b2 in product(angles, repeat=4):
        U = np.kron(Rx(b1), Rx(b2)) @ MS @ np.kron(Rx(a1), Rx(a2))
        ok = True
        expected = [(0, 0), (1, 1), (3, 2), (2, 3)]
        for inp, outp in expected:
            out = U @ np.eye(4)[inp]
            if np.abs(out[outp])**2 < 0.95:
                ok = False
                break
        if ok:
            print(f"FOUND with Rx: pre=({a1/np.pi:.2f}pi, {a2/np.pi:.2f}pi), "
                  f"post=({b1/np.pi:.2f}pi, {b2/np.pi:.2f}pi)")
            check_cnot(U, "Found CNOT (Rx)")

# Also try mixed Rx/Ry
print("\nTrying mixed rotations...")
rot_ops = [(Rx, 'Rx'), (Ry, 'Ry')]
for (op1,n1), (op2,n2), (op3,n3), (op4,n4) in product(rot_ops, repeat=4):
    for a1, a2, b1, b2 in product(angles, repeat=4):
        U = np.kron(op3(b1), op4(b2)) @ MS @ np.kron(op1(a1), op2(a2))
        ok = True
        expected = [(0, 0), (1, 1), (3, 2), (2, 3)]
        for inp, outp in expected:
            out = U @ np.eye(4)[inp]
            if np.abs(out[outp])**2 < 0.95:
                ok = False
                break
        if ok:
            print(f"FOUND: {n3}({b1/np.pi:.2f}pi)_c {n4}({b2/np.pi:.2f}pi)_t "
                  f"MS {n1}({a1/np.pi:.2f}pi)_c {n2}({a2/np.pi:.2f}pi)_t")
            check_cnot(U, "Found CNOT (mixed)")
            break
    else:
        continue
    break
