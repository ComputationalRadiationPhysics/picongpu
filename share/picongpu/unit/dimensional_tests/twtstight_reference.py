#!/usr/bin/env python3
"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Alexander Debus's LLM agent
License: GPLv3+

Reference values for TWTSTight.cpp; Python 3 + mpmath 1.3.0.

Run: python3 twtstight_reference.py
Uses 100 decimal digits and direct Bessel functions, not the scaled asymptotic
implementation. Decimal input strings avoid initial Python float rounding.
The field formulas are the EField.tpp/BField.tpp expressions in SI units,
specialized to the test parameters (positive phi, beta0=1).
"""

from mpmath import mp

mp.dps = 100


def row(values):
    return "{" + ", ".join(mp.nstr(v, 22) for v in values) + "}"


def ratio_references():
    # a, Re(q), Im(q): all exactly representable in float and double.
    cases = [
        ("49.9990234375", "3", "49.9990234375"),
        ("50", "3", "50"),
        ("50.0009765625", "3", "50.0009765625"),
        ("50", "3", "49"),
        ("50", "3", "51"),
        ("50", "0", "0"),
        ("50", "3", "0"),
        ("50", "1", "2"),
        ("100", "0", "100"),
        ("1000", "0", "1000"),
        ("1000", "0", "0"),
    ]
    print("// a, Re(q), Im(q), Re(R0), Im(R0), Re(R1), Im(R1)")
    for inputs in cases:
        a, qr, qi = map(mp.mpf, inputs)
        q = mp.mpc(qr, qi)
        r0, r1 = [mp.besselj(n, q) / mp.besseli(0, a) for n in (0, 1)]
        if qr == 0:
            # J0(i*y) is real; J1(i*y) is imaginary. Remove tiny numerical
            # remnants in components that vanish identically by symmetry.
            r0, r1 = mp.mpc(r0.real, 0), mp.mpc(0, r1.imag)
        print(row([a, qr, qi, r0.real, r0.imag, r1.real, r1.imag]))
    # For negative Im(q), conjugate both ratios: J_n(conj(q))=conj(J_n(q)).
    # At a=1000, q=0, R0 is nonzero here but underflows to zero in float/double;
    # the C++ table therefore stores zero for this entry.


def fields(waist_um, z_sign):
    c = mp.mpf(299792458)
    wavelength = mp.mpf("800e-9")
    phi, polarization = mp.pi / 36, mp.pi / 6
    tauG = mp.mpf("60e-15")  # twice the 30 fs constructor parameter
    w0 = mp.mpf(waist_um) * mp.mpf("1e-6")
    x = y = mp.mpf("1e-6")
    z = z_sign * mp.mpf("150e-6")
    t = mp.mpf("1e-15")
    # At this time no wavelength-period coordinate reduction is needed.
    k = 2 * mp.pi / wavelength
    omega = c * k
    s, C = mp.sin(phi), mp.cos(phi)
    s2, C2 = s * s, C * C
    P, Q = mp.sin(polarization), mp.cos(polarization)
    cot = C / s
    tanAlpha = (1 - C) / s  # beta0=1
    nu = (y * C + z * s) / c
    xi = (-z * C + y * s) * tanAlpha / c
    D = omega * tauG**2 - 2j * (nu - xi) * cot**2
    D += 2j * (2 * nu - xi) * cot / s - 2j * nu / s2
    envelope = tauG / (mp.sqrt(2) * mp.exp(omega * (t - nu - xi) ** 2 / D) * mp.sqrt(D / (2 * omega)))

    Xm = -z - mp.j * k * w0**2 / 2
    Xm2, x2 = Xm**2, x**2
    rho2 = Xm2 + x2
    rho = mp.sqrt(rho2)  # principal complex square root
    q, a = k * rho * s, k**2 * w0**2 * s / 2
    J0, J1 = mp.besselj(0, q), mp.besselj(1, q)
    # Direct, unscaled numerator and denominator are safe with mpmath.
    F = mp.exp(mp.j * (omega * t - k * y * C)) * envelope * (2 / k) / mp.besseli(0, a)

    # Shorthand above: s=sin(phi), C=cos(phi), P=sin(pol), Q=cos(pol).
    Ex = (
        mp.j
        * F
        / (4 * rho * rho2)
        * (
            k * rho * J0 * ((rho2 - x2 + x * Xm * C) * P * s2 + Q * (rho2 + rho2 * C2 - x2 * s2 - x * C * s2 * Xm))
            + J1
            * s
            * (
                P * (-rho2 + 2 * x2 - mp.j * rho2 * Xm * k * s + x * C * (-2 * Xm - mp.j * rho2 * k * s))
                + Q * (-rho2 + 2 * x2 + mp.j * rho2 * Xm * k * s + x * C * (2 * Xm + mp.j * rho2 * k * s))
            )
        )
    )
    Ey = (
        F
        * k
        * s
        / (4 * rho)
        * (J1 * (Q * (Xm - 2 * x * C - Xm * C2) + (1 + C2) * P * Xm) + mp.j * rho * J0 * (Q - P) * s2)
    )
    Ez = (
        mp.j
        * F
        / (8 * rho * rho2)
        * (
            2 * k * rho * J0 * (x * (Q + P) * s2 * Xm + C * (Q * s2 * Xm2 + P * (2 * rho2 - Xm2 * s2)))
            + J1
            * s
            * (
                Q * (-4 * x * Xm + 2 * C * (rho2 - 2 * Xm2) + 2j * rho2 * (x - Xm * C) * k * s)
                + P
                * (
                    -4 * x * Xm
                    - 2 * C * (rho2 - 2 * Xm2)
                    - 2j * rho2 * k * x * s
                    + mp.j * rho2 * Xm * k * mp.sin(2 * phi)
                )
            )
        )
    )
    Bx = (
        -mp.j
        * F
        / (4 * c * rho * rho2)
        * (
            k * rho * J0 * (Q * (-rho2 + x2 + x * C * Xm) * s2 - P * (rho2 + rho2 * C2 - x2 * s2 + x * C * s2 * Xm))
            + J1
            * (
                Q * s * (rho2 - 2 * x2 + mp.j * Xm * rho2 * k * s + x * C * (-2 * Xm - mp.j * rho2 * k * s))
                + P * ((rho2 - 2 * x2) * s + mp.j * rho2 * (-Xm + x * C) * k * s2 + x * mp.sin(2 * phi) * Xm)
            )
        )
    )
    By = (
        F
        * k
        * s
        / (4 * c * rho)
        * (-J1 * (Q * (1 + C2) * Xm + (Xm + 2 * x * C - Xm * C2) * P) + mp.j * rho * J0 * (Q - P) * s2)
    )
    Bz = (
        -mp.j
        * F
        / (4 * c * rho * rho2)
        * (
            J1
            * s
            * (
                P * (x * (2 * Xm - mp.j * k * rho2 * s) + C * (rho2 - 2 * Xm2 - mp.j * Xm * k * rho2 * s))
                + Q * (x * (2 * Xm + mp.j * k * rho2 * s) + C * (-rho2 + 2 * Xm2 + mp.j * Xm * k * rho2 * s))
            )
            + k * rho * J0 * (Xm * (-x + Xm * C) * P * s2 - Q * (x * s2 * Xm + C * (-2 * rho2 + Xm2 * s2)))
        )
    )
    return [v.real for v in (Ex, Ey, Ez, Bx, By, Bz)]


if __name__ == "__main__":
    ratio_references()
    print("\n// waist [m], zSign, {Ex, Ey, Ez, Bx, By, Bz}")
    for waist in ("2.5", "16", "20", "32"):
        for sign in (1, -1):
            print(
                "{" + mp.nstr(mp.mpf(waist) * mp.mpf("1e-6")) + ", " + str(sign) + ", " + row(fields(waist, sign)) + "}"
            )
    # The original +z, 2.5 um test retains its original Mathematica reference;
    # the corresponding mpmath row is printed for comparison only.
