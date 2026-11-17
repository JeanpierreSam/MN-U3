"""
examen3_solver.py
-----------------
Examen 3 de Métodos Numéricos (UPeU) - programa propio de verificación.

Problema 1: energía de un rack (datos tabulados con dos espaciados):
            Simpson 1/3 en [0, 30] (h = 5) y Simpson 3/8 en [30, 60] (h = 10).
Problema 2: PVI  x dy - (2x + y) dx = 0,  y(1) = 1  ->  y' = (2x + y)/x
            con RK4 (h y h/2) y comparación con RK2 (Heun).

Ejecutar:
    python examen3_solver.py
"""

import math

import numpy as np

# =====================================================================
# PROBLEMA 1
# =====================================================================
T1 = np.array([0, 5, 10, 15, 20, 25, 30], dtype=float)
P1 = np.array([120, 134, 151, 163, 158, 147, 140], dtype=float)
T2 = np.array([30, 40, 50, 60], dtype=float)
P2 = np.array([140, 152, 171, 165], dtype=float)


def simpson13(y, h):
    """Simpson 1/3 compuesto sobre datos equiespaciados (n par)."""
    n = len(y) - 1
    if n % 2 != 0:
        raise ValueError(f"Simpson 1/3 requiere n par (n = {n}).")
    c = np.ones(n + 1)
    c[1:-1:2] = 4
    c[2:-1:2] = 2
    return h / 3 * float(np.dot(c, y))


def simpson38(y, h):
    """Simpson 3/8 compuesto sobre datos equiespaciados (n múltiplo de 3)."""
    n = len(y) - 1
    if n % 3 != 0:
        raise ValueError(f"Simpson 3/8 requiere n múltiplo de 3 (n = {n}).")
    c = np.full(n + 1, 3.0)
    c[0] = c[-1] = 1
    c[3:-1:3] = 2
    return 3 * h / 8 * float(np.dot(c, y))


def problema1():
    h1, h2 = 5.0, 10.0
    I1 = simpson13(P1, h1)          # kW*min
    I2 = simpson38(P2, h2)          # kW*min
    total = I1 + I2
    E_kwh = total / 60.0
    p_prom = total / 60.0           # intervalo de 60 min = 1 h
    print("=" * 60)
    print("PROBLEMA 1")
    print("=" * 60)
    print(f"Tramo [0, 30]  Simpson 1/3  h={h1:g}  n={len(P1) - 1}  I1 = {I1:.6f} kW*min")
    print(f"Tramo [30, 60] Simpson 3/8  h={h2:g}  n={len(P2) - 1}  I2 = {I2:.6f} kW*min")
    print(f"Total = {total:.6f} kW*min")
    print(f"b) E = {E_kwh:.6f} kWh")
    print(f"d) Potencia promedio = {p_prom:.6f} kW")


# =====================================================================
# PROBLEMA 2
# =====================================================================
def f(x, y):
    return (2 * x + y) / x


def exacta(x):
    return x * (2 * math.log(x) + 1)


def paso_rk2(x, y, h):  # Heun
    k1 = f(x, y)
    k2 = f(x + h, y + h * k1)
    return y + h / 2 * (k1 + k2)


def paso_rk4(x, y, h):
    k1 = f(x, y)
    k2 = f(x + h / 2, y + h / 2 * k1)
    k3 = f(x + h / 2, y + h / 2 * k2)
    k4 = f(x + h, y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def resolver(paso, x0, y0, xf, h, imprimir=False):
    n = int(round((xf - x0) / h))
    x, y = x0, y0
    for i in range(n):
        y_sig = paso(x, y, h)
        if imprimir:
            print(f"  i={i:2d}  x={x:.1f}  y={y:.6f}  ->  y_(i+1)={y_sig:.6f}")
        x, y = x + h, y_sig
    return y


def problema2():
    x0, y0, xf = 1.0, 1.0, 2.6
    y_ex = 7.568659  # valor dado en el examen
    print("=" * 60)
    print("PROBLEMA 2")
    print("=" * 60)
    print(f"Exacta (formula) y(2.6) = {exacta(xf):.6f}; valor del examen = {y_ex}")
    resumen = {}
    for nombre, paso in (("RK4", paso_rk4), ("RK2", paso_rk2)):
        errs = []
        for h in (0.2, 0.1):
            print(f"\n{nombre}  h = {h}  (n = {int(round((xf - x0) / h))})")
            y = resolver(paso, x0, y0, xf, h, imprimir=(nombre == "RK4"))
            er = abs(y_ex - y) / abs(y_ex)
            errs.append(er)
            print(f"  y(2.6) ~ {y:.8f}   error relativo = {er:.4e}")
        resumen[nombre] = errs[0] / errs[1]
    print("\nFactor de reducción del error relativo (h -> h/2):")
    for nombre, factor in resumen.items():
        print(f"  {nombre}: {factor:.4f}")
    print("Referencia 2^p: 2^4 = 16, 2^2 = 4")


if __name__ == "__main__":
    problema1()
    print()
    problema2()
