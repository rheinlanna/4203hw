#!/usr/bin/env python3
import numpy as np
import math
import matplotlib.pyplot as plt

# -------------------------
# Given / chosen parameters
# -------------------------
a = 5.67         # bohr (3 Å = 5.67 bohr)  [given in problem]
tmax = 10        # truncate lattice sums: t = -tmax ... tmax
alpha = 8.0 / (9.0 * math.pi)   # bohr^{-2}  (from lecture)

HARTREE_TO_EV = 27.211386245988


# -------------------------
# Matrix elements for unnormalized Gaussian:
# phi_t(r) = exp(-alpha |r - R_t|^2),  R_t = t*a
# -------------------------
def S_t(alpha: float, t: int, a: float) -> float:
    """Overlap S_t = <phi_0|phi_t>"""
    R = abs(t) * a
    return (math.pi / (2.0 * alpha)) ** 1.5 * math.exp(-0.5 * alpha * R * R)

def T_t(alpha: float, t: int, a: float) -> float:
    """Kinetic T_t = <phi_0| -1/2 ∇^2 |phi_t>"""
    R = abs(t) * a
    pref = (math.pi / (2.0 * alpha)) ** 1.5
    expo = math.exp(-0.5 * alpha * R * R)
    return pref * expo * (alpha / 2.0) * (3.0 - alpha * R * R)

def V_single_nucleus(alpha: float, t: int, a: float) -> float:
    """
    V0(t) = <phi_0| -1/|r-R0| |phi_t>, nucleus at the phi_0 site.
    For t=0: limit is V0(0) = -2π/α (for this unnormalized Gaussian convention).
    For t≠0: erf formula.
    """
    if t == 0:
        return -2.0 * math.pi / alpha

    R = abs(t) * a
    expo = math.exp(-0.5 * alpha * R * R)

    factor = math.sqrt(math.pi / 2.0) / (math.sqrt(alpha) * R)
    arg = math.sqrt(alpha / 2.0) * R
    return -(math.pi / alpha) * expo * factor * math.erf(arg)

def V_loc_t(alpha: float, t: int, a: float) -> float:
    """
    Local approximation:
      t=0  : include only on-site nucleus once
      t≠0  : include nuclei at both involved sites (0 and t)
    By symmetry in this chain: V_loc(t≠0) = 2 * V0(t)
    """
    if t == 0:
        return V_single_nucleus(alpha, 0, a)
    return 2.0 * V_single_nucleus(alpha, t, a)

def H_t(alpha: float, t: int, a: float) -> float:
    """H_t = <phi_0|H|phi_t> under local approximation"""
    return T_t(alpha, t, a) + V_loc_t(alpha, t, a)


# -------------------------
# Band energy: non-orthogonal Bloch reduction
# k is reduced in [0,0.5] (Γ to zone boundary)
# phase = exp(-i 2π k t)
# -------------------------
def E_k(alpha: float, k_red: float, a: float, tmax: int) -> float:
    num = 0.0 + 0.0j
    den = 0.0 + 0.0j
    for t in range(-tmax, tmax + 1):
        phase = complex(math.cos(-2.0 * math.pi * k_red * t),
                        math.sin(-2.0 * math.pi * k_red * t))
        num += H_t(alpha, t, a) * phase
        den += S_t(alpha, t, a) * phase
    return float((num / den).real)

def total_energy_per_cell(alpha: float, a: float, tmax: int, nk: int = 4001) -> float:
    """
    Total electronic energy per unit cell for half-filling (1 electron per site, spin-deg 2),
    using band energy E_k(alpha, k_red, a, tmax) in Hartree.

    E_tot/N = 4 * ∫_{0}^{0.25} E(k_red) dk_red
    """
    kF = 0.25
    ks = np.linspace(0.0, kF, nk)
    Es = np.array([E_k(alpha, float(k), a, tmax) for k in ks])  # Hartree
    integral = np.trapz(Es, ks)   # ∫ E dk_red
    return 4.0 * integral         # Hartree per unit cell


def main():
    print(f"[b] Using a = {a:.4f} bohr, tmax = {tmax}")
    print(f"[b] Using alpha = 8/(9π) = {alpha:.10f} bohr^-2")

    # Γ -> BZ edge in reduced units
    ks = np.linspace(0.0, 0.5, 401)
    Es = np.array([E_k(alpha, float(k), a, tmax) for k in ks])

    # Print basic band info
    Emin = Es.min()
    Emax = Es.max()
    print(f"[b] Band min: {Emin:.8f} Ha = {Emin*HARTREE_TO_EV:.6f} eV")
    print(f"[b] Band max: {Emax:.8f} Ha = {Emax*HARTREE_TO_EV:.6f} eV")
    print(f"[b] Bandwidth: {(Emax-Emin):.8f} Ha = {(Emax-Emin)*HARTREE_TO_EV:.6f} eV")
    
    Etot_cell = total_energy_per_cell(alpha, a, tmax, nk=4001)
    print(f"[d] Total electronic energy per unit cell (half-filled): {Etot_cell:.10f} Ha "f"= {Etot_cell*HARTREE_TO_EV:.6f} eV")
    # Plot in eV
    plt.figure()
    plt.plot(ks, Es * HARTREE_TO_EV)
    plt.xlabel("k (reduced units, Γ: 0 → BZ edge: 0.5)")
    plt.ylabel("ε(k) (eV)")
    plt.title("1D H chain band (Gaussian basis + local approx)\n"
              f"a={a:.2f} bohr, |t|≤{tmax}, α=8/(9π)={alpha:.4f} bohr$^{{-2}}$")
    plt.grid(True)
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    main()
