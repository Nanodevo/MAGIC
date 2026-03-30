"""
Ion Trap Physics Module
=======================

THE PHYSICS OF ION TRAPPING:

    ┌─────────────────────────────────────────────────────────────────┐
    │                     PAUL TRAP (RF TRAP)                        │
    │                                                                │
    │  Earnshaw's theorem forbids trapping a charged particle with   │
    │  static electric fields alone. The Paul trap solves this by    │
    │  using oscillating (RF) fields to create a time-averaged       │
    │  confining pseudopotential.                                    │
    │                                                                │
    │  The electric potential in a linear Paul trap:                  │
    │                                                                │
    │    Φ(x,y,t) = (V_RF cos(Ω_RF t) / r₀²)(x² - y²) / 2        │
    │             + (κ U_end / z₀²)(z² - (x²+y²)/2)                │
    │                                                                │
    │  where:                                                        │
    │    V_RF   = RF voltage amplitude                               │
    │    Ω_RF   = RF drive frequency (typically 10-100 MHz)          │
    │    r₀     = electrode distance from trap axis                  │
    │    U_end  = DC endcap voltage for axial confinement            │
    │    z₀     = endcap separation                                  │
    │    κ      = geometric factor                                   │
    │                                                                │
    │  This creates:                                                 │
    │    - Radial confinement from time-averaged RF pseudopotential   │
    │    - Axial confinement from static DC endcap voltages          │
    │                                                                │
    │  The ion motion has two components:                            │
    │    1. SECULAR MOTION: slow oscillation at frequency ω_sec      │
    │       (the "useful" motion, typically 0.5 - 5 MHz)             │
    │    2. MICROMOTION: fast oscillation at Ω_RF driven by the      │
    │       RF field (minimized at trap center)                      │
    └─────────────────────────────────────────────────────────────────┘

    ION CHAIN EQUILIBRIUM:
    Multiple ions in the same trap repel each other via Coulomb force
    and arrange in a linear crystal along the weakest (axial) axis:

        ← ● ─── ● ─── ● ─── ● →
        ion 1  ion 2  ion 3  ion 4

    Equilibrium positions are found by minimizing:
        V_total = Σ_i (½ m ω_z² z_i²) + Σ_{i<j} e²/(4πε₀|z_i - z_j|)
                  ↑ harmonic trap          ↑ Coulomb repulsion

    NORMAL MODES OF MOTION:
    Small oscillations around equilibrium decompose into collective
    normal modes — these are the quantum bus for entangling gates:

        - Center-of-mass (COM) mode: all ions move together
          → ● → ● → ● →   frequency = ω_z

        - Stretch (breathing) mode: alternating motion
          → ● ← ● → ● ←   frequency = √3 · ω_z (for 2 ions)

        - Higher modes with more complex patterns

    Each mode is a quantum harmonic oscillator with energy levels
    |n⟩ = |0⟩, |1⟩, |2⟩, ... (phonon number states).
    The Mølmer-Sørensen gate exploits these shared phonon modes to
    create entanglement between ions.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from scipy.optimize import minimize
from scipy.constants import e, epsilon_0, atomic_mass, pi, k as k_B


# ═══════════════════════════════════════════════════════════════════
# Physical Constants & Ion Species
# ═══════════════════════════════════════════════════════════════════

@dataclass
class IonSpecies:
    """
    Physical properties of an ion species used for qubit encoding.

    Common trapped-ion qubit species:
        ⁴⁰Ca⁺  : optical qubit (S₁/₂ ↔ D₅/₂), 729 nm laser
        ¹⁷¹Yb⁺ : hyperfine qubit (²S₁/₂ F=0 ↔ F=1), 12.6 GHz microwave
        ⁹Be⁺   : hyperfine qubit, 1.25 GHz microwave
        ¹³⁸Ba⁺ : optical qubit, good for networking
    """
    name: str
    mass_amu: float                     # Mass in atomic mass units
    qubit_transition_freq_hz: float     # |0⟩ ↔ |1⟩ transition frequency
    qubit_wavelength_nm: float          # Corresponding wavelength
    t1_seconds: float                   # Excited state lifetime (T₁)
    t2_seconds: float                   # Coherence time (T₂)
    scattering_rate_hz: float           # Off-resonant photon scattering rate

    @property
    def mass_kg(self) -> float:
        return self.mass_amu * atomic_mass


# Pre-defined ion species
CALCIUM_40 = IonSpecies(
    name="⁴⁰Ca⁺",
    mass_amu=39.9625906,
    qubit_transition_freq_hz=4.11e14,       # S₁/₂ ↔ D₅/₂ at 729 nm
    qubit_wavelength_nm=729.0,
    t1_seconds=1.168,                        # D₅/₂ metastable lifetime ~1.17s
    t2_seconds=0.01,                         # Typical T₂ ~10 ms (field-dependent)
    scattering_rate_hz=1e3,
)

YTTERBIUM_171 = IonSpecies(
    name="¹⁷¹Yb⁺",
    mass_amu=170.936323,
    qubit_transition_freq_hz=12.642812e9,    # Hyperfine splitting
    qubit_wavelength_nm=369.5,               # Detection/cooling transition
    t1_seconds=1e10,                         # Hyperfine: essentially infinite T₁
    t2_seconds=1.5,                          # Demonstrated >1 s with DD sequences
    scattering_rate_hz=0.5,
)

BERYLLIUM_9 = IonSpecies(
    name="⁹Be⁺",
    mass_amu=9.0121831,
    qubit_transition_freq_hz=1.25e9,         # Hyperfine splitting
    qubit_wavelength_nm=313.0,               # Detection transition
    t1_seconds=1e10,
    t2_seconds=1.5,
    scattering_rate_hz=1.0,
)


# ═══════════════════════════════════════════════════════════════════
# Paul Trap Model
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TrapParameters:
    """
    Parameters defining a linear Paul trap.

    PHYSICS:
        The Mathieu stability parameter q determines whether an ion is stably
        trapped:  q = 2eV_RF / (m Ω_RF² r₀²)
        Stable trapping requires q < 0.908 (first stability region).

        The secular (harmonic) frequencies are:
            ω_radial ≈ q·Ω_RF / (2√2)    (pseudopotential approximation)
            ω_axial  = √(2κeU_end / (mz₀²))

        Typical values:
            ω_radial / 2π ~ 1-10 MHz
            ω_axial  / 2π ~ 0.5-3 MHz
            ω_radial > ω_axial ensures a linear chain
    """
    rf_voltage: float = 200.0           # V_RF in volts
    rf_frequency_hz: float = 30e6       # Ω_RF in Hz
    r0_meters: float = 200e-6           # Electrode-axis distance
    endcap_voltage: float = 5.0         # U_end in volts
    z0_meters: float = 2.0e-3           # Endcap half-separation
    geometric_factor: float = 0.29      # κ, geometry-dependent


class IonTrap:
    """
    Simulates the physics of ions in a linear Paul trap.

    Computes:
        - Trap frequencies (secular motion)
        - Mathieu stability parameters
        - Ion chain equilibrium positions
        - Normal modes of collective motion
        - Lamb-Dicke parameters (laser-motion coupling)
    """

    def __init__(self, species: IonSpecies, trap_params: TrapParameters, num_ions: int):
        self.species = species
        self.trap = trap_params
        self.num_ions = num_ions

        # Derived quantities
        self._compute_trap_frequencies()
        self._compute_equilibrium_positions()
        self._compute_normal_modes()

    def _compute_trap_frequencies(self):
        """
        Compute secular trap frequencies.

        PHYSICS:
            The Mathieu q-parameter:
                q = 2eV_RF / (m Ω²_RF r₀²)

            In the pseudopotential approximation (q << 1), the radial
            secular frequency is:
                ω_r = (q / 2√2) · Ω_RF = eV_RF / (√2 · m · Ω_RF · r₀²)

            The axial frequency from the DC endcaps:
                ω_z = √(2κeU_end / (m z₀²))

            For a linear chain: ω_r >> ω_z (strong radial confinement).
        """
        m = self.species.mass_kg
        Omega = 2 * pi * self.trap.rf_frequency_hz

        # Mathieu q parameter (should be < 0.908 for stability)
        self.q_parameter = (2 * e * self.trap.rf_voltage) / (m * Omega**2 * self.trap.r0_meters**2)

        if self.q_parameter > 0.908:
            raise ValueError(
                f"Mathieu q = {self.q_parameter:.3f} > 0.908: "
                f"ions are UNSTABLE! Reduce V_RF or increase Ω_RF."
            )

        # Radial secular frequency (pseudopotential approximation)
        self.omega_radial = (self.q_parameter / (2 * np.sqrt(2))) * Omega

        # Axial secular frequency (DC endcaps)
        self.omega_axial = np.sqrt(
            2 * self.trap.geometric_factor * e * self.trap.endcap_voltage
            / (m * self.trap.z0_meters**2)
        )

        self.freq_radial_hz = self.omega_radial / (2 * pi)
        self.freq_axial_hz = self.omega_axial / (2 * pi)

    def _compute_equilibrium_positions(self):
        """
        Find equilibrium positions of the ion chain.

        PHYSICS:
            Each ion i sits at position z_i along the trap axis.
            The total potential energy is:

            V = Σ_i ½mω_z² z_i²  +  Σ_{i<j} e²/(4πε₀|z_i - z_j|)
                ↑ harmonic trap        ↑ Coulomb repulsion

            We non-dimensionalize using the length scale:
                l₀ = (e²/(4πε₀ m ω_z²))^(1/3)

            In these units, positions u_i = z_i/l₀ satisfy:
                u_i = Σ_{j≠i} sign(u_i - u_j) / (u_i - u_j)²

            For 1 ion:  u = [0]
            For 2 ions: u = [-0.63, +0.63] (= ±(1/4)^(1/3))
            For 3 ions: u = [-1.08, 0, +1.08]
        """
        N = self.num_ions
        m = self.species.mass_kg
        omega_z = self.omega_axial

        # Characteristic length scale
        self.length_scale = (e**2 / (4 * pi * epsilon_0 * m * omega_z**2))**(1.0/3.0)

        if N == 1:
            self.equilibrium_positions_scaled = np.array([0.0])
            self.equilibrium_positions_meters = np.array([0.0])
            return

        # Minimize total potential in scaled coordinates
        def potential(u):
            """Dimensionless potential energy."""
            V_harm = 0.5 * np.sum(u**2)
            V_coul = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dist = abs(u[i] - u[j])
                    if dist < 1e-15:
                        return 1e20  # Regularize collision
                    V_coul += 1.0 / dist
            return V_harm + V_coul

        # Initial guess: evenly spaced
        u0 = np.linspace(-(N-1)/2, (N-1)/2, N) * 1.2

        result = minimize(potential, u0, method='BFGS')
        self.equilibrium_positions_scaled = np.sort(result.x)
        self.equilibrium_positions_meters = self.equilibrium_positions_scaled * self.length_scale

    def _compute_normal_modes(self):
        """
        Compute normal modes of the ion chain.

        PHYSICS:
            Small oscillations δz_i around equilibrium are coupled through
            the Coulomb interaction. We Taylor-expand the potential to
            second order to get the Hessian matrix A:

                A_ij = δ²V / (δu_i δu_j)

            where V is the dimensionless potential. The eigenvalues of A
            give squared mode frequencies: ω_p² = λ_p · ω_z²
            The eigenvectors give the mode participation vectors b_p,i
            (how much ion i participates in mode p).

            Mode 0 (COM):   all b equal, ω = ω_z
            Mode 1 (stretch): alternating b, ω = √3·ω_z (for N=2)

            These modes are quantum harmonic oscillators:
                H_mode = ℏω_p (â†â + ½)
            where â, â† create/destroy phonons.
        """
        N = self.num_ions
        u = self.equilibrium_positions_scaled

        if N == 1:
            self.mode_frequencies = np.array([self.omega_axial])
            self.mode_vectors = np.array([[1.0]])
            self.mode_freq_ratios = np.array([1.0])
            return

        # Build Hessian matrix of the dimensionless potential
        A = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                if i == j:
                    # Diagonal: trap + Coulomb from all other ions
                    A[i, i] = 1.0  # Trap contribution
                    for k in range(N):
                        if k != i:
                            A[i, i] += 2.0 / abs(u[i] - u[k])**3
                else:
                    # Off-diagonal: Coulomb coupling
                    A[i, j] = -2.0 / abs(u[i] - u[j])**3

        # Eigendecomposition → normal modes
        eigenvalues, eigenvectors = np.linalg.eigh(A)

        # Sort by frequency (lowest first = COM mode)
        sort_idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]

        # Mode frequencies: ω_p = √(λ_p) · ω_z
        self.mode_freq_ratios = np.sqrt(np.maximum(eigenvalues, 0))
        self.mode_frequencies = self.mode_freq_ratios * self.omega_axial

        # Mode participation vectors (columns = modes, rows = ions)
        # Normalize so Σ_i b_{p,i}² = 1
        self.mode_vectors = eigenvectors

    def lamb_dicke_parameter(self, mode_index: int, ion_index: int,
                              laser_wavevector: Optional[float] = None) -> float:
        """
        Compute the Lamb-Dicke parameter η for a given ion and mode.

        PHYSICS:
            η_{p,i} = k · b_{p,i} · z_zpf_p

            where:
                k = 2π/λ is the laser wavevector component along the trap axis
                b_{p,i} is ion i's participation in mode p
                z_zpf_p = √(ℏ/(2mω_p)) is the zero-point fluctuation of mode p

            The Lamb-Dicke parameter determines the coupling strength
            between the laser and the ion's motion. It appears in:

                H_int ∝ Ω(|1⟩⟨0| ⊗ exp(iη(â+â†)) + h.c.)

            In the Lamb-Dicke regime (η√(n̄+1) << 1), we can expand:
                exp(iη(â+â†)) ≈ 1 + iη(â + â†)

            This gives three types of transitions:
                - CARRIER:    |↓,n⟩ → |↑,n⟩     (no phonon change)
                - RED SIDEBAND:  |↓,n⟩ → |↑,n-1⟩  (remove one phonon)
                - BLUE SIDEBAND: |↓,n⟩ → |↑,n+1⟩  (add one phonon)

            Typical values: η ~ 0.05 - 0.3
        """
        from scipy.constants import hbar

        if laser_wavevector is None:
            laser_wavevector = 2 * pi / (self.species.qubit_wavelength_nm * 1e-9)

        omega_p = self.mode_frequencies[mode_index]
        m = self.species.mass_kg
        b_pi = self.mode_vectors[ion_index, mode_index]

        # Zero-point fluctuation
        z_zpf = np.sqrt(hbar / (2 * m * omega_p))

        return float(abs(laser_wavevector * b_pi * z_zpf))

    def motional_heating_rate(self, temperature_K: float = 300.0,
                               electrode_distance_m: Optional[float] = None) -> float:
        """
        Estimate motional heating rate (phonons/second) for the COM mode.

        PHYSICS:
            Electric field noise from trap electrodes causes the ion's
            motional state to heat up: ⟨n⟩ increases over time.

            The heating rate scales approximately as:
                dn/dt ∝ S_E(ω) · e² / (4mℏω)

            where S_E(ω) is the electric field noise spectral density.
            Empirically, S_E scales as ~ d⁻⁴ (distance to electrodes)
            and ~ ω⁻¹ to ω⁻² (frequency scaling).

            Typical values: 10 - 10⁴ phonons/s for room temperature traps,
            much less for cryogenic traps.

            This is a major source of decoherence for motional modes and
            limits entangling gate fidelity.
        """
        from scipy.constants import hbar

        if electrode_distance_m is None:
            electrode_distance_m = self.trap.r0_meters

        omega = self.mode_frequencies[0]  # COM mode
        m = self.species.mass_kg

        # Empirical model: S_E ≈ S₀ · (d₀/d)⁴ · (ω₀/ω)
        # with S₀ ~ 10⁻¹¹ V²/m² Hz at d₀=100μm, ω₀/2π=1MHz at 300K
        S0 = 1e-11 * (temperature_K / 300.0)
        d0 = 100e-6
        omega0 = 2 * pi * 1e6

        S_E = S0 * (d0 / electrode_distance_m)**4 * (omega0 / omega)

        # Heating rate
        dn_dt = e**2 * S_E / (4 * m * hbar * omega)
        return float(dn_dt)

    def print_trap_info(self):
        """Print a detailed summary of the trap configuration."""
        print("=" * 65)
        print("        TRAPPED ION QUANTUM COMPUTER CONFIGURATION")
        print("=" * 65)

        print(f"\n  Ion Species: {self.species.name}")
        print(f"  Mass: {self.species.mass_amu:.4f} amu")
        print(f"  Qubit transition: {self.species.qubit_wavelength_nm} nm "
              f"({self.species.qubit_transition_freq_hz/1e9:.3f} GHz)")
        print(f"  T₁ (relaxation): {self.species.t1_seconds:.3g} s")
        print(f"  T₂ (coherence):  {self.species.t2_seconds:.3g} s")

        print(f"\n  ── Trap Parameters ──")
        print(f"  RF voltage V_RF = {self.trap.rf_voltage} V")
        print(f"  RF frequency Ω_RF/2π = {self.trap.rf_frequency_hz/1e6:.1f} MHz")
        print(f"  Electrode distance r₀ = {self.trap.r0_meters*1e6:.0f} μm")
        print(f"  Endcap voltage U_end = {self.trap.endcap_voltage} V")
        print(f"  Endcap separation 2z₀ = {2*self.trap.z0_meters*1e3:.1f} mm")

        print(f"\n  ── Stability & Frequencies ──")
        print(f"  Mathieu q parameter: {self.q_parameter:.4f} "
              f"({'STABLE' if self.q_parameter < 0.908 else 'UNSTABLE!'})")
        print(f"  Radial secular freq ω_r/2π = {self.freq_radial_hz/1e6:.3f} MHz")
        print(f"  Axial secular freq  ω_z/2π = {self.freq_axial_hz/1e6:.3f} MHz")
        print(f"  Frequency ratio ω_r/ω_z = {self.freq_radial_hz/self.freq_axial_hz:.2f} "
              f"({'>1 ✓ linear chain' if self.freq_radial_hz > self.freq_axial_hz else '<1 ✗ may buckle!'})")

        print(f"\n  ── Ion Chain ({self.num_ions} ion{'s' if self.num_ions>1 else ''}) ──")
        print(f"  Length scale l₀ = {self.length_scale*1e6:.3f} μm")
        if self.num_ions <= 20:
            print(f"  Equilibrium positions (μm):")
            for i, z in enumerate(self.equilibrium_positions_meters):
                print(f"    Ion {i}: z = {z*1e6:+.3f} μm")

            # ASCII art of ion chain
            print(f"\n  Ion chain (not to scale):")
            chain = "    "
            for i in range(self.num_ions):
                chain += f"── ●(ion{i}) "
            chain += "──"
            print(chain)

        print(f"\n  ── Normal Modes ──")
        mode_names = ["COM", "stretch", "scissors"]
        for p in range(min(self.num_ions, 10)):
            name = mode_names[p] if p < len(mode_names) else f"mode {p}"
            freq_ratio = self.mode_freq_ratios[p]
            freq_mhz = self.mode_frequencies[p] / (2 * pi * 1e6)
            eta_str = ""
            if self.num_ions <= 5:
                etas = [self.lamb_dicke_parameter(p, i) for i in range(self.num_ions)]
                eta_str = f"  η = [{', '.join(f'{h:.4f}' for h in etas)}]"
            print(f"    Mode {p} ({name}): ω/ω_z = {freq_ratio:.4f}, "
                  f"f = {freq_mhz:.4f} MHz{eta_str}")

        heating = self.motional_heating_rate()
        print(f"\n  ── Motional Heating ──")
        print(f"  Estimated COM heating rate: {heating:.1f} phonons/s")
        print("=" * 65)
