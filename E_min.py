import math
import csv
import matplotlib.pyplot as plt

# 0) Physical constants (atomic units)
MP = 1836.152673     # proton mass (m_e = 1)
MU = MP / 2.0        # reduced mass for H–H

# 1) error function with coefficient F0(T)
def F0(T: float) -> float:
    if T < 1e-12:
        return 1.0
    return 0.5 * math.sqrt(math.pi) * math.erf(math.sqrt(T)) / math.sqrt(T)


# 2) E(alpha, d)

def energy_H2plus(alpha: float, d: float) -> float:
    if alpha <= 0 or d <= 0:
        return float("inf")

    # common factor
    S0 = (math.pi / (2.0 * alpha)) ** 1.5
    eta = math.exp(-alpha * d * d / 2.0)

    # Overlap
    S11 = S0
    S12 = S0 * eta

    # Kinetic energy
    T11 = (3.0 * alpha / 2.0) * S0
    T12 = S0 * eta * (alpha / 2.0) * (3.0 - alpha * d * d)

    # Potential energy
    V11 = -(math.pi / alpha) * (1.0 + F0(2.0 * alpha * d * d))
    V12 = -(2.0 * math.pi / alpha) * eta * F0(alpha * d * d / 2.0)

    h11 = T11 + V11
    h12 = T12 + V12

    # Bonding eigenvalue
    eps_plus = (h11 + h12) / (S11 + S12)

    # Add nuclear–nuclear repulsion
    return eps_plus + 1.0 / d

# 3) 1D minimization tools
def bracket_minimum(f, x0=0.5, step=0.2, grow=1.6, max_iter=60):
    a = max(1e-6, x0)
    fa = f(a)
    b = max(1e-6, a + step)
    fb = f(b)

    if fb > fa:
        a, b = b, a
        fa, fb = fb, fa
        step = -step

    c = max(1e-6, b + step)
    fc = f(c)

    for _ in range(max_iter):
        if fb < fa and fb < fc:
            return a, b, c
        a, fa = b, fb
        b, fb = c, fc
        step *= grow
        c = max(1e-6, b + step)
        fc = f(c)

    return a, b, c

def golden_section_search(f, a, b, c, tol=1e-8):
    lo, hi = (a, c) if a < c else (c, a)
    gr = (math.sqrt(5.0) - 1.0) / 2.0

    x1 = hi - gr * (hi - lo)
    x2 = lo + gr * (hi - lo)
    f1, f2 = f(x1), f(x2)

    while abs(hi - lo) > tol:
        if f1 > f2:
            lo = x1
            x1, f1 = x2, f2
            x2 = lo + gr * (hi - lo)
            f2 = f(x2)
        else:
            hi = x2
            x2, f2 = x1, f1
            x1 = hi - gr * (hi - lo)
            f1 = f(x1)

    return (x1, f1) if f1 < f2 else (x2, f2)

def minimize_alpha_for_d(d: float):
    # coarse scan
    alphas = [0.02 * i for i in range(1, 401)]
    energies = [energy_H2plus(a, d) for a in alphas]
    alpha0 = alphas[min(range(len(energies)), key=lambda i: energies[i])]

    f = lambda a: energy_H2plus(a, d)
    aL, aM, aR = bracket_minimum(f, x0=alpha0, step=0.1)
    return golden_section_search(f, aL, aM, aR)

# 4) Q2: Generate BO curve
def generate_BO_curve(d_min=0.6, d_max=6.0, d_step=0.05):
    ds, alphas, Es = [], [], []
    n = int(round((d_max - d_min) / d_step)) + 1

    for i in range(n):
        d = d_min + i * d_step
        alpha_star, E_star = minimize_alpha_for_d(d)
        ds.append(d)
        alphas.append(alpha_star)
        Es.append(E_star)
        print(f"[Q2] d={d:.3f}  alpha*={alpha_star:.6f}  E_BO={E_star:.10f}")

    return ds, alphas, Es

# 5) Q3: Minimum and dissociation energy
def find_minimum(ds, Es, alphas):
    i0 = min(range(len(Es)), key=lambda i: Es[i])
    return i0, ds[i0], Es[i0], alphas[i0]

# 6) Q4: Taylor expansion vibrational frequency
def quadratic_fit(ds, Es, i0, half_window=4):
    xs = ds[i0-half_window:i0+half_window+1]
    ys = Es[i0-half_window:i0+half_window+1]

    n = len(xs)
    Sx1 = sum(xs)
    Sx2 = sum(x*x for x in xs)
    Sx3 = sum(x**3 for x in xs)
    Sx4 = sum(x**4 for x in xs)
    Sy1 = sum(x*y for x, y in zip(xs, ys))
    Sy2 = sum(x*x*y for x, y in zip(xs, ys))

    denom = (n*Sx4 - Sx2*Sx2)
    A = (n*Sy2 - Sx2*Sy1) / denom
    B = (Sy1 - A*Sx2) / Sx1

    k = 2.0 * A
    d0 = -B / (2.0*A)

    omega = math.sqrt(k / MU)
    nu_cm = omega / (2.0*math.pi) * 219474.63

    return d0, k, omega, nu_cm

# 7) Main: Q2 + Q3 + Q4 + plot
def main():
    ds, alphas, Es = generate_BO_curve()

    # Save data
    with open("BO_curve.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["d_bohr", "alpha_star", "E_BO_Hartree"])
        for d, a, E in zip(ds, alphas, Es):
            w.writerow([d, a, E])

    # Plot BO curve
    plt.figure()
    plt.plot(ds, Es, "-o", markersize=3)
    plt.xlabel("d (bohr)")
    plt.ylabel("E_BO(d) (Hartree)")
    plt.title("H2+ Born–Oppenheimer curve (2 Gaussian basis)")
    plt.grid(True)
    plt.show()

    # Q3
    i0, d0, Emin, alpha0 = find_minimum(ds, Es, alphas)
    print("\n[Q3]")
    print(f"d0    = {d0:.6f} bohr")
    print(f"Emin  = {Emin:.10f} Hartree")

    # Q4
    d0_fit, k, omega, nu_cm = quadratic_fit(ds, Es, i0)
    print("\n[Q4]")
    print("E_BO(d) ≈ Emin + 1/2 k (d − d0)^2")
    print(f"k     = {k:.6e} Hartree/bohr^2")
    print(f"ω     = {omega:.6e} a.u.")
    print(f"ν~    = {nu_cm:.2f} cm^-1")

if __name__ == "__main__":
    main()
