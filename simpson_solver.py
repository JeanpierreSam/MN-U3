"""
simpson_solver.py
-----------------
Nucleo numerico de la Sesion 13 - Integracion Numerica: Reglas de Simpson
(UPeU, Metodos Numericos, Unidad 3). Logica pura con NumPy, sin interfaz.

Basado en la Guia Teorica (GT) de la Sesion 13:

  Simpson 1/3 simple (n = 2, 3 puntos, polinomio de Lagrange de grado 2):
      I ~ (h/3) [f(x0) + 4 f(x1) + f(x2)]
      |E_S| <= h^5/90 * max |f''''(x)|

  Simpson 1/3 compuesto (n par, h = (b - a)/n):
      I ~ (h/3) [f(x0) + 4 f(x1) + 2 f(x2) + 4 f(x3) + ... + 2 f(x_{n-2}) + 4 f(x_{n-1}) + f(x_n)]
      |E_S| <= n h^5/180 * max |f''''(x)|  =  (b - a) h^4/180 * max |f''''(x)|

  Simpson 3/8 (n multiplo de 3, polinomio de grado 3):
      I ~ (3h/8) [f(x0) + 3 f(x1) + 3 f(x2) + 2 f(x3) + 3 f(x4) + ... + 3 f(x_{n-1}) + f(x_n)]
      E_t = -(b - a)^5/6480 * f''''(xi)     (aplicacion simple, n = 3)
      compuesto: |E| <= (b - a) h^4/80 * max |f''''(x)|

Caso GAA: consumo energetico de un centro de datos de microservicios,
    P(t) = 10 + 5 t sin(pi t / 6) + e^(0.2 t)   [kW],   E = int_0^6 P(t) dt  [kWh]
    n = 6 -> h = 1 -> I_simpson ~ 128.9212 kWh ; exacto = 55 + 180/pi + 5 e^1.2 = 128.8964 kWh
"""

import math

import numpy as np

from integracion_solver import _evaluar, trapecio_compuesto


# ---------------------------------------------------------------------
# Excepciones y validacion (GAA: "validar mediante una excepcion que n
# sea un numero entero par")
# ---------------------------------------------------------------------
class NNoParError(ValueError):
    """Se lanza cuando Simpson 1/3 recibe un n que no es entero par."""


class NNoMultiplo3Error(ValueError):
    """Se lanza cuando Simpson 3/8 recibe un n que no es multiplo de 3."""


def _validar_intervalo(a, b):
    if not (np.isfinite(a) and np.isfinite(b)):
        raise ValueError("Los limites a y b deben ser numeros finitos.")
    if b <= a:
        raise ValueError("Se requiere a < b.")


def validar_n_par(n):
    """Retorna n como int si es entero par >= 2; si no, lanza NNoParError."""
    try:
        es_entero = float(n) == int(float(n))
    except (TypeError, ValueError):
        es_entero = False
    if not es_entero:
        raise NNoParError(f"n = {n!r} no es un numero entero. Simpson 1/3 requiere n entero par.")
    n = int(float(n))
    if n < 2 or n % 2 != 0:
        raise NNoParError(f"n = {n} no es par. Simpson 1/3 requiere un numero PAR de subintervalos (n >= 2).")
    return n


def validar_n_multiplo3(n):
    try:
        es_entero = float(n) == int(float(n))
    except (TypeError, ValueError):
        es_entero = False
    if not es_entero or int(float(n)) < 3 or int(float(n)) % 3 != 0:
        raise NNoMultiplo3Error(f"n = {n} no es multiplo de 3. Simpson 3/8 requiere n = 3, 6, 9, ...")
    return int(float(n))


# ---------------------------------------------------------------------
# Coeficientes vectorizados (GAA: "vectorizando los coeficientes
# [1, 4, 2, 4, ..., 2, 4, 1]")
# ---------------------------------------------------------------------
def coeficientes_simpson13(n):
    """Vector [1, 4, 2, 4, ..., 2, 4, 1] de longitud n + 1 (n par)."""
    n = validar_n_par(n)
    c = np.full(n + 1, 2.0)
    c[1::2] = 4.0          # indices impares -> 4
    c[0] = c[-1] = 1.0     # extremos -> 1
    return c


def coeficientes_simpson38(n):
    """Vector [1, 3, 3, 2, 3, 3, 2, ..., 3, 3, 1] de longitud n + 1 (n multiplo de 3)."""
    n = validar_n_multiplo3(n)
    c = np.full(n + 1, 3.0)
    c[3:-1:3] = 2.0        # union de bloques de 3 -> 2
    c[0] = c[-1] = 1.0
    return c


# ---------------------------------------------------------------------
# Reglas de Simpson
# ---------------------------------------------------------------------
def simpson13(f, a, b, n=2):
    """
    Simpson 1/3 compuesto (n = 2 da la regla simple). Calculo optimizado:
    un solo producto punto entre el vector de coeficientes y f(x).

        I ~ (h/3) * (c . f(x)),   c = [1, 4, 2, 4, ..., 2, 4, 1]
    """
    _validar_intervalo(a, b)
    c = coeficientes_simpson13(n)
    n = len(c) - 1
    h = (b - a) / n
    x = a + h * np.arange(n + 1)
    return float(h / 3 * np.dot(c, _evaluar(f, x)))


def simpson38(f, a, b, n=3):
    """Simpson 3/8 compuesto (n = 3 da la regla simple)."""
    _validar_intervalo(a, b)
    c = coeficientes_simpson38(n)
    n = len(c) - 1
    h = (b - a) / n
    x = a + h * np.arange(n + 1)
    return float(3 * h / 8 * np.dot(c, _evaluar(f, x)))


def simpson13_datos(ys, h):
    """Simpson 1/3 sobre datos tabulados equiespaciados (len(ys) - 1 debe ser par)."""
    ys = np.asarray(ys, dtype=float)
    c = coeficientes_simpson13(len(ys) - 1)
    return float(h / 3 * np.dot(c, ys))


def simpson38_datos(ys, h):
    ys = np.asarray(ys, dtype=float)
    c = coeficientes_simpson38(len(ys) - 1)
    return float(3 * h / 8 * np.dot(c, ys))


def tabla_simpson13(f, a, b, n):
    """Paso a paso: i, x_i, f(x_i), coeficiente c_i y c_i * f(x_i)."""
    _validar_intervalo(a, b)
    c = coeficientes_simpson13(n)
    n = len(c) - 1
    h = (b - a) / n
    x = a + h * np.arange(n + 1)
    y = _evaluar(f, x)
    filas = [
        {"i": i, "x_i": float(xi), "f(x_i)": float(yi), "coef c_i": int(ci), "c_i·f(x_i)": float(ci * yi)}
        for i, (xi, yi, ci) in enumerate(zip(x, y, c))
    ]
    return h, filas


# ---------------------------------------------------------------------
# Error (GT: limitante superior)
# ---------------------------------------------------------------------
def cota_error_simpson13(a, b, n, M):
    """|E_S| <= n h^5/180 * M = (b - a) h^4/180 * M,  M = max|f''''|."""
    n = validar_n_par(n)
    h = (b - a) / n
    return float((b - a) * h ** 4 / 180 * M)


def cota_error_simpson38(a, b, n, M):
    """|E| <= (b - a) h^4/80 * M  (n = 3 -> (b - a)^5/6480 * M, como en la GT)."""
    n = validar_n_multiplo3(n)
    h = (b - a) / n
    return float((b - a) * h ** 4 / 80 * M)


def max_abs_cuarta_derivada(f, a, b, puntos=4001):
    """
    Estimacion numerica de max |f''''(x)| en [a, b] con la diferencia
    central de 5 puntos (para comprobar el M que da la guia).
    """
    x = np.linspace(a, b, puntos)
    d = 1e-2 * max(1.0, b - a) / 4
    f4 = (_evaluar(f, x + 2 * d) - 4 * _evaluar(f, x + d) + 6 * _evaluar(f, x)
          - 4 * _evaluar(f, x - d) + _evaluar(f, x - 2 * d)) / d ** 4
    return float(np.nanmax(np.abs(f4)))


def n_minimo_simpson13(a, b, tolerancia, M):
    """Menor n PAR tal que (b - a) h^4/180 * M <= tolerancia. Retorna (n_min, h_max)."""
    if tolerancia <= 0:
        raise ValueError("La tolerancia debe ser positiva.")
    if M == 0:
        return 2, float(b - a)
    h_max = (180 * tolerancia / ((b - a) * M)) ** 0.25
    n = max(2, math.ceil((b - a) / h_max - 1e-12))
    return n + (n % 2), h_max


# ---------------------------------------------------------------------
# Analisis de convergencia (GAA: n en {2, 4, 6, 12, 24})
# ---------------------------------------------------------------------
def tabla_convergencia_simpson(f, a, b, M, ns=(2, 4, 6, 12, 24), valor_exacto=None):
    """Para cada n: h, integral Simpson 1/3, cota |E_t| y (opcional) error real."""
    filas = []
    for n in ns:
        n = validar_n_par(n)
        h = (b - a) / n
        aprox = simpson13(f, a, b, n)
        fila = {"n": n, "h": h, "I_simpson": aprox, "cota |E_t|": cota_error_simpson13(a, b, n, M)}
        if valor_exacto is not None:
            fila["error real"] = float(abs(valor_exacto - aprox))
        filas.append(fila)
    return filas


# =====================================================================
# Casos de las guias de la Sesion 13
# =====================================================================
# --- GAA: potencia P(t) [kW] de un centro de datos en [0, 6] h ---
def P_potencia(t):
    t = np.asarray(t, dtype=float)
    return 10 + 5 * t * np.sin(np.pi * t / 6) + np.exp(0.2 * t)


A_GAA13, B_GAA13, N_GAA13 = 0.0, 6.0, 6
EXACTO_GAA13 = 55 + 180 / math.pi + 5 * math.exp(1.2)   # 128.896364...
NS_GAA13 = (2, 4, 6, 12, 24)
M_GUIA_GAA13 = 0.45                                      # valor dado en la GAA (kW/h^4)
# P''''(t) = (5 pi^4/1296) t sin(pi t/6) - (5 pi^3/54) cos(pi t/6) + e^(t/5)/625
M_REAL_GAA13 = max_abs_cuarta_derivada(P_potencia, A_GAA13, B_GAA13)   # ~3.4332 en t ~ 5.07

# --- GT ejemplos: int_3^3.6 1/x (n=2), int_0^1 e^x (n=2) ---
EXACTO_INV_X = math.log(1.2)
EXACTO_EXP = math.e - 1
# --- GT: int_0^1.2 e^x cos x (Simpson 1/3 y 3/8, n = 6) ---
EXACTO_EXPCOS = (math.exp(1.2) * (math.cos(1.2) + math.sin(1.2)) - 1) / 2
# --- GT: int_1^5 ln x (n = 10) ---
EXACTO_LNX = 5 * math.log(5) - 4
# --- GT Ejercicio: int_1^3 ln(x+1) dx con error < 1e-3 ; f'''' = -6/(x+1)^4 -> M = 6/16
EXACTO_LN1 = 4 * math.log(4) - 2 * math.log(2) - 2
M_LN1 = 6 / 16
TOL_LN1 = 1e-3
# --- GT Problema: perfil de elevacion del terreno (h = 5 m) ---
DIST_TERRENO = np.array([0, 5, 10, 15, 20, 25, 30], dtype=float)
ELEV_TERRENO = np.array([2.1, 2.4, 2.9, 3.0, 2.8, 2.5, 2.2])
# --- GP Ejercicio (comparacion con trapecio): int_0^4 (1 + x^2)^(-1/2) = asinh(4)
EXACTO_EJERCICIO = math.asinh(4)


if __name__ == "__main__":
    print("=== GAA: centro de datos, Simpson 1/3 compuesto n = 6 ===")
    h, filas = tabla_simpson13(P_potencia, A_GAA13, B_GAA13, N_GAA13)
    print("h =", h)
    for fila in filas:
        print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in fila.items()})
    I6 = simpson13(P_potencia, A_GAA13, B_GAA13, N_GAA13)
    print(f"I_simpson(n=6) = {I6:.6f} kWh | exacto = {EXACTO_GAA13:.6f} kWh | error real = {abs(EXACTO_GAA13 - I6):.6f}")
    print(f"Cota con M = 0.45 (guia): {cota_error_simpson13(0, 6, 6, M_GUIA_GAA13):.6f} kWh")
    print(f"max|P''''| numerico = {M_REAL_GAA13:.4f} -> cota = {cota_error_simpson13(0, 6, 6, M_REAL_GAA13):.6f} kWh")

    print("\nConvergencia (M de la guia y error real):")
    for fila in tabla_convergencia_simpson(P_potencia, 0, 6, M_GUIA_GAA13, NS_GAA13, EXACTO_GAA13):
        print({k: (round(v, 10) if isinstance(v, float) else v) for k, v in fila.items()})

    print("\nValidacion de n impar:")
    try:
        simpson13(P_potencia, 0, 6, 5)
    except NNoParError as exc:
        print("NNoParError ->", exc)

    print("\n=== GT: ejemplos Simpson 1/3 (n = 2) ===")
    print("int 1/x [3, 3.6]:", round(simpson13(lambda x: 1 / x, 3, 3.6, 2), 6), "| exacto", round(EXACTO_INV_X, 6))  # 0.1823
    print("int e^x [0, 1]  :", round(simpson13(np.exp, 0, 1, 2), 6), "| exacto", round(EXACTO_EXP, 6))               # 1.7189
    ec = lambda x: np.exp(x) * np.cos(x)
    print("int e^x cos x n=6 (1/3):", round(simpson13(ec, 0, 1.2, 6), 6), "| (3/8):", round(simpson38(ec, 0, 1.2, 6), 6),
          "| exacto", round(EXACTO_EXPCOS, 6))                                                                       # 1.648 / 1.649
    print("int ln x [1, 5] n=10:", round(simpson13(np.log, 1, 5, 10), 6), "| exacto", round(EXACTO_LNX, 6))          # 4.047

    print("\n=== GT Ejercicio: int_1^3 ln(x+1), error < 1e-3 ===")
    n_min, h_max = n_minimo_simpson13(1, 3, TOL_LN1, M_LN1)
    print("h_max =", round(h_max, 6), "| n_min (par) =", n_min)
    for n in (2, n_min):
        v = simpson13(lambda x: np.log(x + 1), 1, 3, n)
        print(f"n={n}: I={v:.6f} | cota={cota_error_simpson13(1, 3, n, M_LN1):.6f} | error real={abs(EXACTO_LN1 - v):.6f}")

    print("\n=== GT Problema: area bajo el perfil del terreno ===")
    print("Trapecio   :", round(float(np.sum(5 * (ELEV_TERRENO[:-1] + ELEV_TERRENO[1:]) / 2)), 4))   # 78.75
    print("Simpson 1/3:", round(simpson13_datos(ELEV_TERRENO, 5), 4))                                 # 78.8333
    print("Simpson 3/8:", round(simpson38_datos(ELEV_TERRENO, 5), 4))                                 # 78.9375

    print("\n=== GP Ejercicio (1+x^2)^(-1/2) en [0, 4]: trapecio vs Simpson ===")
    g = lambda x: (1 + x ** 2) ** -0.5
    for n in (2, 8):
        print(f"n={n}: trapecio={trapecio_compuesto(g, 0, 4, n):.6f} | simpson={simpson13(g, 0, 4, n):.6f} "
              f"| exacto={EXACTO_EJERCICIO:.6f}")
