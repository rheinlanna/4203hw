#!/usr/bin/env python3
import numpy as np
import math
import matplotlib.pyplot as plt

# -------------------------
# Parameters
# -------------------------
a = 5.67  # bohr
t1_max = 10
t2_max = 10
alpha = 8.0 / (9.0 * math.pi)  # 8/(9π)

HARTREE_TO_EV = 27.211386245988


# -------------------------
# Distance in 2D lattice
# -------------------------
def R_of_t(t1, t2):
    return a * math.sqrt(t1*t1 + t2*t2)


# -------------------------
# Matrix elements
# -------------------------
def S_t(t1, t2):
    R = R_of_t(t1, t2)
    return (math.pi/(2*alpha))**1.5 * math.exp(-0.5*alpha*R*R)

def T_t(t1, t2):
    R = R_of_t(t1, t2)
    pref = (math.pi/(2*alpha))**1.5
    expo = math.exp(-0.5*alpha*R*R)
    return pref * expo * (alpha/2.0) * (3.0 - alpha*R*R)

def V_single(t1, t2):
    if t1 == 0 and t2 == 0:
        return -2.0 * math.pi / alpha

    R = R_of_t(t1, t2)
    expo = math.exp(-0.5*alpha*R*R)
    factor = math.sqrt(math.pi/2.0)/(math.sqrt(alpha)*R)
    arg = math.sqrt(alpha/2.0)*R
    return -(math.pi/alpha)*expo*factor*math.erf(arg)

def V_loc(t1, t2):
    if t1 == 0 and t2 == 0:
        return V_single(0,0)
    return 2.0 * V_single(t1,t2)

def H_t(t1, t2):
    return T_t(t1,t2) + V_loc(t1,t2)


# -------------------------
# 2D band energy
# -------------------------
def E_k(k1, k2):
    num = 0.0 + 0.0j
    den = 0.0 + 0.0j

    for t1 in range(-t1_max, t1_max+1):
        for t2 in range(-t2_max, t2_max+1):
            phase_angle = -2.0*math.pi*(k1*t1 + k2*t2)
            phase = complex(math.cos(phase_angle),
                            math.sin(phase_angle))

            num += H_t(t1,t2) * phase
            den += S_t(t1,t2) * phase

    return float((num/den).real)


# -------------------------
# High symmetry path Γ–X–M–Γ
# -------------------------
def make_path(nseg=120):
    G = np.array([0.0, 0.0])
    X = np.array([0.5, 0.0])
    M = np.array([0.5, 0.5])

    def segment(p,q,n):
        return np.linspace(p,q,n,endpoint=False)

    kpts = np.vstack([
        segment(G,X,nseg),
        segment(X,M,nseg),
        np.linspace(M,G,nseg+1)
    ])

    dist = np.zeros(len(kpts))
    for i in range(1,len(kpts)):
        dist[i] = dist[i-1] + np.linalg.norm(kpts[i]-kpts[i-1])

    idx_G = 0
    idx_X = nseg
    idx_M = 2*nseg
    idx_G2 = len(kpts)-1

    return kpts, dist, (idx_G, idx_X, idx_M, idx_G2)


# -------------------------
# Main
# -------------------------
def main():
    print("2D band (absolute energy)")
    print(f"a = {a} bohr")
    print(f"alpha = {alpha:.8f}")
    print(f"cutoff = {t1_max}")

    kpts, dist, (iG,iX,iM,iG2) = make_path()

    Es = np.zeros(len(kpts))
    for i,(k1,k2) in enumerate(kpts):
        Es[i] = E_k(float(k1), float(k2))

    Es_ev = Es * HARTREE_TO_EV  # absolute energy

    plt.figure()
    plt.plot(dist, Es_ev)
    plt.ylabel("Energy (eV)")
    plt.xlabel("k-path (Γ → X → M → Γ)")
    plt.title("2D H lattice band (absolute energy)")

    plt.axvline(dist[iG], linewidth=1)
    plt.axvline(dist[iX], linewidth=1)
    plt.axvline(dist[iM], linewidth=1)
    plt.axvline(dist[iG2], linewidth=1)

    plt.xticks(
        [dist[iG], dist[iX], dist[iM], dist[iG2]],
        ["Γ", "X", "M", "Γ"]
    )

    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()