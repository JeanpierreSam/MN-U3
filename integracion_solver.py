"""
integracion_solver.py
---------------------
Nucleo numerico de la Sesion 12 - Integracion Numerica (UPeU, Metodos
Numericos, Unidad 3). Logica pura con NumPy, sin interfaz.

Basado en la Guia Teorica (GT) de la Sesion 12:

  Regla de los rectangulos (suma de Riemann), h = (b - a)/n:
      S_n = sum_{i=1}^{n} h * f(c_i)                c_i en [x_{i-1}, x_i]

  Regla del trapecio simple (RTS), n = 1, h = b - a:
      I ~ (h/2) [f(x0) + f(x1)]
      |E_Ts| <= h^3/12 * max |f''(x)|

  Regla del trapecio compuesto (RTC), n subintervalos, h = (b - a)/n:
      I ~ (h/2) [f(x0) + 2(f(x1) + ... + f(x_{n-1})) + f(x_n)]
      |E_TR| <= n h^3/12 * max |f''(x)|  =  (b - a) h^2/12 * max |f''(x)|

Caso GAA: trafico de red de un microservicio en un datacenter,
    R(t) = 10 + 5 sin(pi t / 4) + e^(0.15 t)   [Mbps],   D = int_0^12 R(t) dt
    n = 4 -> h = 3 -> I_trapecio ~ 160.4437 Mb ; exacto = 166.3967 Mb
"""

import math

import numpy as np


# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------
def _validar(a, b, n):
    if not (np.isfinite(a) and np.isfinite(b)):
        raise ValueError("Los limites a y b deben ser numeros finitos.")
    if b <= a:
        raise ValueError("Se requiere a < b.")
    if int(n) != n or n < 1:
        raise ValueError("n debe ser un entero >= 1.")
    return int(n)


def _evaluar(f, x):
    """Evalua f en un arreglo; acepta funciones no vectorizadas."""
    x = np.asarray(x, dtype=float)
    try:
        y = np.asarray(f(x), dtype=float)
        if y.shape != x.shape:                     # p. ej. f = lambda t: 5.0
            y = np.broadcast_to(y, x.shape).astype(float)
    except Exception:
        y = np.array([float(f(v)) for v in x])
    return y


# ---------------------------------------------------------------------
# Regla de los rectangulos (GT, diapositivas 4-7)
# ---------------------------------------------------------------------
def rectangulos(f, a, b, n, punto="medio"):
    """
    Suma de Riemann con n rectangulos de base h. `punto` elige c_i:
    'izquierdo', 'medio' (como el ejemplo de la GT) o 'derecho'.
    """
    n = _validar(a, b, n)
    h = (b - a) / n
    x = a + h * np.arange(n + 1)
    if punto == "izquierdo":
        c = x[:-1]
    elif punto == "derecho":
        c = x[1:]
    else:
        c = (x[:-1] + x[1:]) / 2
    return float(h * _evaluar(f, c).sum())


# ---------------------------------------------------------------------
# Regla del trapecio (GT, diapositivas 12-24)
# ---------------------------------------------------------------------
def malla(a, b, n):
    """Paso h y nodos equiespaciados x_i = a + i h, i = 0..n."""
    n = _validar(a, b, n)
    h = (b - a) / n
    return h, a + h * np.arange(n + 1)


def trapecio_simple(f, a, b):
    """RTS: I ~ (h/2)[f(a) + f(b)], con h = b - a."""
    return trapecio_compuesto(f, a, b, 1)


def trapecio_compuesto(f, a, b, n):
    """
    RTC (funcion pedida por la GAA): integral de f en [a, b] con n
    subintervalos de igual ancho, para cualquier n >= 1.

        I ~ (h/2) [f(x0) + 2 sum_{i=1}^{n-1} f(x_i) + f(x_n)]
    """
    h, x = malla(a, b, n)
    y = _evaluar(f, x)
    return float(h / 2 * (y[0] + 2 * y[1:-1].sum() + y[-1]))


def trapecio_datos(xs, ys):
    """
    Trapecio sobre datos tabulados (GT, 'Observaciones': cuando solo se
    tiene una tabla de puntos). Admite espaciado no uniforme.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.ndim != 1 or xs.shape != ys.shape or len(xs) < 2:
        raise ValueError("X e Y deben ser vectores del mismo tamano (>= 2).")
    if np.any(np.diff(xs) <= 0):
        raise ValueError("Las abscisas deben estar en orden estrictamente creciente.")
    return float(np.sum(np.diff(xs) * (ys[:-1] + ys[1:]) / 2))


def tabla_trapecio(f, a, b, n):
    """
    Detalle paso a paso: i, x_i, f(x_i), peso c_i (1 en los extremos,
    2 en los interiores) y c_i * f(x_i). La integral es h/2 * suma.
    """
    h, x = malla(a, b, n)
    y = _evaluar(f, x)
    pesos = np.full(len(x), 2.0)
    pesos[0] = pesos[-1] = 1.0
    filas = [
        {"i": i, "x_i": float(xi), "f(x_i)": float(yi), "peso c_i": int(ci), "c_i·f(x_i)": float(ci * yi)}
        for i, (xi, yi, ci) in enumerate(zip(x, y, pesos))
    ]
    return h, filas


# ---------------------------------------------------------------------
# Error (GT, diapositivas 25-29)
# ---------------------------------------------------------------------
def max_abs_segunda_derivada(f, a, b, puntos=4001):
    """
    Estimacion numerica de max |f''(x)| en [a, b] con diferencias
    centrales sobre una malla fina (para cuando no se deriva a mano).
    """
    x = np.linspace(a, b, puntos)
    d = 1e-4 * max(1.0, b - a)
    f2 = (_evaluar(f, x + d) - 2 * _evaluar(f, x) + _evaluar(f, x - d)) / d ** 2
    return float(np.nanmax(np.abs(f2)))


def cota_error_trapecio(a, b, n, max_f2):
    """|E_TR| <= n h^3 / 12 * max|f''|  (n = 1 da la cota de la RTS)."""
    n = _validar(a, b, n)
    h = (b - a) / n
    return float(n * h ** 3 / 12 * max_f2)


def n_minimo_trapecio(a, b, tolerancia, max_f2):
    """
    Menor n tal que (b - a) h^2/12 * max|f''| <= tolerancia (GP, ejemplo
    de e^x cos x). Retorna (n_min, h_max).
    """
    if tolerancia <= 0:
        raise ValueError("La tolerancia debe ser positiva.")
    if max_f2 == 0:
        return 1, float(b - a)
    h_max = math.sqrt(12 * tolerancia / ((b - a) * max_f2))
    return max(1, math.ceil((b - a) / h_max - 1e-12)), h_max


# ---------------------------------------------------------------------
# Analisis de convergencia (GAA, Parte 2)
# ---------------------------------------------------------------------
def tabla_convergencia(f, a, b, valor_exacto, ns=(4, 10, 50, 100, 1000), max_f2=None):
    """
    Para cada n: h, valor calculado, error absoluto real |I - I_n|,
    porcentaje de error 100 |I - I_n| / |I|, y (opcional) la cota teorica.
    """
    filas = []
    for n in ns:
        h = (b - a) / n
        aprox = trapecio_compuesto(f, a, b, n)
        err = float(abs(valor_exacto - aprox))
        fila = {
            "n": int(n),
            "h": h,
            "I_trapecio": aprox,
            "error absoluto": err,
            "error %": 100 * err / abs(valor_exacto) if valor_exacto != 0 else float("nan"),
        }
        if max_f2 is not None:
            fila["cota teórica"] = cota_error_trapecio(a, b, n, max_f2)
        filas.append(fila)
    return filas


# ---------------------------------------------------------------------
# Funciones escritas por el usuario (Modo libre)
# ---------------------------------------------------------------------
_PERMITIDOS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "log": np.log,
    "ln": np.log, "log10": np.log10, "sqrt": np.sqrt, "abs": np.abs,
    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh, "asinh": np.arcsinh,
    "pi": np.pi, "e": np.e,
}


def construir_funcion(expresion, var="x"):
    """
    Convierte un texto como 'exp(x)*cos(x)' en una funcion vectorizada.
    Solo expone funciones matematicas de NumPy (sin builtins). Acepta '^'.
    """
    texto = expresion.strip().replace("^", "**")
    if not texto:
        raise ValueError("Escribe una funcion.")
    codigo = compile(texto, "<f>", "eval")
    for nombre in codigo.co_names:
        if nombre not in _PERMITIDOS and nombre != var:
            raise ValueError(f"Nombre no permitido en la funcion: '{nombre}'.")

    def f(x):
        return eval(codigo, {"__builtins__": {}}, {**_PERMITIDOS, var: x})

    f(np.array([0.5]))  # prueba rapida de sintaxis
    return f


# =====================================================================
# Casos de las guias de la Sesion 12
# =====================================================================
# --- GAA: trafico de red R(t) [Mbps] en [0, 12] s, n = 4 ---
def R_trafico(t):
    return 10 + 5 * np.sin(np.pi * t / 4) + np.exp(0.15 * t)


A_GAA, B_GAA, N_GAA = 0.0, 12.0, 4
EXACTO_GAA = 120 + 40 / math.pi + (math.exp(1.8) - 1) / 0.15  # 166.396712...
NS_GAA = (4, 10, 50, 100, 1000)
# R''(t) = -(5 pi^2/16) sin(pi t/4) + 0.0225 e^(0.15 t)
MAXF2_GAA = max_abs_segunda_derivada(R_trafico, A_GAA, B_GAA)

# --- GT: rectangulos  int_{-1}^{1} (1 - x^2) dx, n = 4 (punto medio) -> 1.375 ; exacto 4/3
EXACTO_GT_RECT = 4 / 3

# --- GT / GP "Aplica": int_3^3.6 1/x dx  y  int_0^1 e^x dx
EXACTO_INV_X = math.log(1.2)                                   # 0.182322
EXACTO_EXP = math.e - 1                                        # 1.718282

# --- GP / GT "Ejercicio": int_0^4 (1 + x^2)^(-1/2) dx = asinh(4) = 2.0947
EXACTO_EJERCICIO = math.asinh(4)

# --- GP / GT "Ejemplo": A = int_0^1.2 e^x cos x dx, error < 0.5e-3
#     f''(x) = -2 e^x sin x  ->  max|f''| en [0, 1.2] = 2 e^1.2 sin 1.2 = 6.1889
EXACTO_EXPCOS = (math.exp(1.2) * (math.cos(1.2) + math.sin(1.2)) - 1) / 2
MAXF2_EXPCOS = 2 * math.exp(1.2) * math.sin(1.2)
TOL_EXPCOS = 0.5e-3


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=6)

    print("=== GAA: trafico de red, trapecio compuesto n = 4 ===")
    h, filas = tabla_trapecio(R_trafico, A_GAA, B_GAA, N_GAA)
    print("h =", h)
    for fila in filas:
        print(fila)
    I4 = trapecio_compuesto(R_trafico, A_GAA, B_GAA, N_GAA)
    print(f"I_trapecio(n=4) = {I4:.6f} Mb | exacto = {EXACTO_GAA:.6f} Mb")
    print(f"max|R''| ~ {MAXF2_GAA:.6f}")
    print("\nConvergencia:")
    for fila in tabla_convergencia(R_trafico, A_GAA, B_GAA, EXACTO_GAA, NS_GAA, MAXF2_GAA):
        print({k: (round(v, 10) if isinstance(v, float) else v) for k, v in fila.items()})

    print("\n=== GT: rectangulos int_-1^1 (1 - x^2), n = 4 ===")
    r = rectangulos(lambda x: 1 - x ** 2, -1, 1, 4)
    print(r, "| error abs =", round(abs(EXACTO_GT_RECT - r), 4),
          "| error rel % =", round(100 * abs(EXACTO_GT_RECT - r) / EXACTO_GT_RECT, 2))   # 1.375 | 0.0417 | 3.13

    print("\n=== GP/GT: trapecio simple ===")
    print("int 1/x  [3, 3.6] n=1:", trapecio_simple(lambda x: 1 / x, 3, 3.6), "| exacto", EXACTO_INV_X)  # 0.18333
    print("int e^x  [0, 1]   n=1:", trapecio_simple(np.exp, 0, 1), "| exacto", EXACTO_EXP)               # 1.85914
    print("\n=== GT: trapecio compuesto ===")
    print("int 1/x  n=6 :", trapecio_compuesto(lambda x: 1 / x, 3, 3.6, 6))     # 0.18233
    print("int e^x  n=10:", trapecio_compuesto(np.exp, 0, 1, 10))               # 1.71971
    print("cotas e^x: n=1", cota_error_trapecio(0, 1, 1, math.e), "| n=10", cota_error_trapecio(0, 1, 10, math.e))  # 0.2265 | 0.0023

    print("\n=== GP: Ejercicio (1 + x^2)^(-1/2) en [0, 4] ===")
    g = lambda x: (1 + x ** 2) ** -0.5
    for n in (1, 2, 8):
        v = trapecio_compuesto(g, 0, 4, n)
        print(f"n={n}: {v:.6f} | error abs {abs(EXACTO_EJERCICIO - v):.6f}")

    print("\n=== GP: Ejemplo e^x cos x en [0, 1.2], error < 0.5e-3 ===")
    print("max|f''| =", round(MAXF2_EXPCOS, 4), "| numerico:",
          round(max_abs_segunda_derivada(lambda x: np.exp(x) * np.cos(x), 0, 1.2), 4))
    n_min, h_max = n_minimo_trapecio(0, 1.2, TOL_EXPCOS, MAXF2_EXPCOS)
    print("h_max =", round(h_max, 6), "| n_min =", n_min)                         # 0.028424 | 43
    for n in (n_min - 1, n_min, 48):
        v = trapecio_compuesto(lambda x: np.exp(x) * np.cos(x), 0, 1.2, n)
        print(f"n={n}: I={v:.6f} | cota={cota_error_trapecio(0, 1.2, n, MAXF2_EXPCOS):.6f} | "
              f"error real={abs(EXACTO_EXPCOS - v):.6f}")

    print("\n=== Datos tabulados (GT, n=6 de 1/x) ===")
    xs = np.linspace(3, 3.6, 7)
    print(trapecio_datos(xs, 1 / xs))
