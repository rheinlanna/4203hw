import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Graphene NN tight-binding
# ----------------------------

def reciprocal_vectors(a1, a2):
    """
    Given 2D lattice vectors a1, a2 (Cartesian), return reciprocal vectors b1, b2
    such that a_i · b_j = 2π δ_ij.
    """
    A = np.column_stack([a1, a2])          # 2x2 matrix: [a1 a2]
    B = 2*np.pi * np.linalg.inv(A).T       # B = 2π (A^{-T})
    b1 = B[:, 0]
    b2 = B[:, 1]
    return b1, b2

def k_path(points, n_per_segment=200):
    """
    Build a piecewise-linear k-path through 'points' (list of 2D vectors).
    Returns:
      ks: (N,2) array of k-points in Cartesian coords
      s:  (N,)  cumulative distance along the path (for x-axis)
      tick_pos: positions in s for high-symmetry points
    """
    ks = []
    s = []
    tick_pos = [0.0]

    total = 0.0
    for i in range(len(points)-1):
        k0 = points[i]
        k1 = points[i+1]
        seg = np.linspace(0.0, 1.0, n_per_segment, endpoint=False)
        for t in seg:
            k = (1-t)*k0 + t*k1
            ks.append(k)
            if len(s) == 0:
                s.append(0.0)
            else:
                dk = np.linalg.norm(ks[-1] - ks[-2])
                total += dk
                s.append(total)

        # add tick position at end of the segment
        # (distance from last added point to the segment end)
        dk_end = np.linalg.norm(k1 - ks[-1])
        total += dk_end
        tick_pos.append(total)

    # append final point
    ks.append(points[-1])
    s.append(total)

    return np.array(ks), np.array(s), tick_pos

def graphene_energies(k_cart, t=2.5, a=1.0):
    """
    Nearest-neighbor graphene TB: E±(k) = ± t |f(k)|
    using 3 NN vectors delta_i (Cartesian).
    'a' here is NN bond length used to define delta vectors (scale choice).
    """
    # 3 nearest-neighbor vectors (one common convention)
    d1 = np.array([0.0, a])
    d2 = np.array([np.sqrt(3)*a/2, -a/2])
    d3 = np.array([-np.sqrt(3)*a/2, -a/2])

    phase = lambda d: np.exp(1j * (k_cart @ d))
    f = phase(d1) + phase(d2) + phase(d3)
    E = t * np.abs(f)
    return -E, E  # (valence, conduction)

def main():
    # ----------------------------
    # Lattice vectors from HW figure:
    # a1 = a*(sqrt(3)/2 i + 1/2 j), a2 = a*(sqrt(3)/2 i - 1/2 j)
    # We'll set a=1 for geometry scale (energies scale with t anyway).
    # ----------------------------
    a = 1.0
    a1 = a * np.array([np.sqrt(3)/2,  1/2])
    a2 = a * np.array([np.sqrt(3)/2, -1/2])

    b1, b2 = reciprocal_vectors(a1, a2)

    # High-symmetry points in reciprocal space (Cartesian):
    Gamma = np.array([0.0, 0.0])
    # Common choices for hexagonal BZ:
    M = 0.5 * (b1 + b2)
    K = (2*b1 + b2) / 3.0

    # Build path Γ → M → K → Γ
    ks, s, tick_pos = k_path([Gamma, M, K, Gamma], n_per_segment=250)

    # Compute energies along path
    Ev, Ec = graphene_energies(ks, t=2.5, a=a)  # eV

    # Plot
    plt.figure()
    plt.plot(s, Ev)
    plt.plot(s, Ec)
    plt.axhline(0.0, linewidth=1)

    plt.xticks(tick_pos, [r"$\Gamma$", r"$M$", r"$K$", r"$\Gamma$"])
    plt.xlabel("k-path")
    plt.ylabel("E (eV)")
    plt.title("Graphene band structure (NN TB, t = 2.5 eV)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
