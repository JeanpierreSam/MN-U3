"""
edo_solver.py
-------------
Nucleo numerico de la Sesion 14 - Ecuaciones Diferenciales Ordinarias
(UPeU, Metodos Numericos, Unidad 3). Logica pura con NumPy, sin interfaz.

PVI:   dy/dx = f(x, y),   y(x0) = y0        (metodos de un paso, h = tamaño de paso)

Basado en la Guia Teorica (GT) de la Sesion 14 - Metodos de Runge-Kutta:

  Euler (orden 1):
      y_{i+1} = y_i + h f(x_i, y_i)

  Euler mejorado / Heun (orden 2)  -> predice y luego corrige:
      z_{i+1} = y_i + h f(x_i, y_i)                          (predictor: Euler)
      y_{i+1} = y_i + (h/2) [ f(x_i, y_i) + f(x_{i+1}, z_{i+1}) ]   (corrector)

  Runge-Kutta de orden 3 (RK3, forma de la GT):
      k1 = f(x_i, y_i)
      k2 = f(x_i + h/2, y_i + (h/2) k1)
      k3 = f(x_i + h,   y_i - h k1 + 2 h k2)
      y_{i+1} = y_i + (h/6) (k1 + 4 k2 + k3)

  Runge-Kutta de orden 4 (RK4):
      k1 = f(x_i, y_i)
      k2 = f(x_i + h/2, y_i + (h/2) k1)
      k3 = f(x_i + h/2, y_i + (h/2) k2)
      k4 = f(x_i + h,   y_i + h k3)
      y_{i+1} = y_i + (h/6) (k1 + 2 k2 + 2 k3 + k4)

Caso GAA:  (y^2 - x^2) dx + x y dy = 0,  y(1) = 2   ->   y' = (x^2 - y^2)/(x y)
    Solucion exacta (ec. homogenea, y = v x):  2 x^2 y^2 - x^4 = 7
        y(x) = sqrt((7 + x^4) / (2 x^2))      ->   y(1.2) = 1.774...
    Se aproxima y(1.2) con h = 0.1 (2 pasos) por Euler y Euler mejorado.
"""

import math

import numpy as np

from integracion_solver import _PERMITIDOS


# ---------------------------------------------------------------------
# Excepciones y validacion
# ---------------------------------------------------------------------
class PasoInvalidoError(ValueError):
    """h no es positivo, o (xf - x0)/h no es (casi) un entero >= 1."""


def numero_pasos(x0, xf, h):
    """Retorna n = (xf - x0)/h como entero; lanza PasoInvalidoError si no cuadra."""
    if not all(np.isfinite(v) for v in (x0, xf, h)):
        raise PasoInvalidoError("x0, xf y h deben ser numeros finitos.")
    if h <= 0:
        raise PasoInvalidoError("El tamaño de paso h debe ser positivo.")
    if xf <= x0:
        raise PasoInvalidoError("Se requiere xf > x0.")
    n = (xf - x0) / h
    if n < 1 - 1e-9 or abs(n - round(n)) > 1e-9:
        raise PasoInvalidoError(
            f"(xf - x0)/h = {n:g} no es entero: h = {h:g} no divide al intervalo [{x0:g}, {xf:g}]."
        )
    return int(round(n))


def construir_funcion_xy(expresion):
    """
    Convierte un texto como '(x^2 - y^2)/(x*y)' en una funcion f(x, y).
    Solo expone funciones matematicas de NumPy (sin builtins). Acepta '^'.
    """
    texto = expresion.strip().replace("^", "**")
    if not texto:
        raise ValueError("Escribe f(x, y).")
    codigo = compile(texto, "<f>", "eval")
    for nombre in codigo.co_names:
        if nombre not in _PERMITIDOS and nombre not in ("x", "y"):
            raise ValueError(f"Nombre no permitido en f(x, y): '{nombre}'.")

    def f(x, y):
        return float(eval(codigo, {"__builtins__": {}}, {**_PERMITIDOS, "x": x, "y": y}))

    f(1.0, 1.0)  # prueba rapida de sintaxis
    return f


# ---------------------------------------------------------------------
# Un paso de cada metodo: devuelve (y_siguiente, detalle)
# ---------------------------------------------------------------------
def paso_euler(f, x, y, h):
    k1 = f(x, y)
    return y + h * k1, {"k1": k1}


def paso_heun(f, x, y, h):
    k1 = f(x, y)
    z = y + h * k1                                  # predictor
    k2 = f(x + h, z)
    return y + h / 2 * (k1 + k2), {"k1": k1, "z": z, "k2": k2}


def paso_rk3(f, x, y, h):
    k1 = f(x, y)
    k2 = f(x + h / 2, y + h / 2 * k1)
    k3 = f(x + h, y - h * k1 + 2 * h * k2)
    return y + h / 6 * (k1 + 4 * k2 + k3), {"k1": k1, "k2": k2, "k3": k3}


def paso_rk4(f, x, y, h):
    k1 = f(x, y)
    k2 = f(x + h / 2, y + h / 2 * k1)
    k3 = f(x + h / 2, y + h / 2 * k2)
    k4 = f(x + h, y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), {"k1": k1, "k2": k2, "k3": k3, "k4": k4}


METODOS = {
    "Euler": (paso_euler, 1),
    "Euler mejorado (Heun)": (paso_heun, 2),
    "Runge-Kutta 3": (paso_rk3, 3),
    "Runge-Kutta 4": (paso_rk4, 4),
}


def resolver(f, x0, y0, xf, h, metodo="Euler"):
    """
    Integra el PVI de x0 a xf con paso h. Retorna (xs, ys, detalles) donde
    detalles[i] son las pendientes usadas en el paso i -> i+1.
    """
    paso = METODOS[metodo][0]
    n = numero_pasos(x0, xf, h)
    xs = x0 + h * np.arange(n + 1)
    ys = np.empty(n + 1)
    ys[0] = y0
    detalles = []
    for i in range(n):
        ys[i + 1], det = paso(f, xs[i], ys[i], h)
        detalles.append(det)
    return xs, ys, detalles


def euler(f, x0, y0, xf, h):
    return resolver(f, x0, y0, xf, h, "Euler")[:2]


def euler_mejorado(f, x0, y0, xf, h):
    return resolver(f, x0, y0, xf, h, "Euler mejorado (Heun)")[:2]


def rk3(f, x0, y0, xf, h):
    return resolver(f, x0, y0, xf, h, "Runge-Kutta 3")[:2]


def rk4(f, x0, y0, xf, h):
    return resolver(f, x0, y0, xf, h, "Runge-Kutta 4")[:2]


# ---------------------------------------------------------------------
# Tablas (paso a paso y comparacion con la solucion exacta)
# ---------------------------------------------------------------------
def tabla_paso_a_paso(f, x0, y0, xf, h, metodo="Euler"):
    """Filas: i, x_i, y_i, pendientes k, y_{i+1} (para el desarrollo a mano)."""
    xs, ys, det = resolver(f, x0, y0, xf, h, metodo)
    filas = []
    for i, d in enumerate(det):
        fila = {"i": i, "x_i": float(xs[i]), "y_i": float(ys[i])}
        fila.update({k: float(v) for k, v in d.items()})
        fila["y_{i+1}"] = float(ys[i + 1])
        filas.append(fila)
    return filas


def error_relativo_pct(aprox, exacto):
    """Error relativo porcentual |exacto - aprox| / |exacto| * 100 (0 si exacto = 0 y coinciden)."""
    if exacto == 0:
        return 0.0 if aprox == 0 else float("inf")
    return float(abs(exacto - aprox) / abs(exacto) * 100)


def tabla_comparacion(xs, ys, exacto_fn):
    """x_n, y_n, exacto, error absoluto y error relativo % (como las tablas de la GT)."""
    filas = []
    for x, y in zip(xs, ys):
        ex = float(exacto_fn(x))
        filas.append({"x_n": float(x), "y_n": float(y), "exacto": ex,
                      "error abs": abs(ex - float(y)), "E_n (%)": error_relativo_pct(float(y), ex)})
    return filas


def tabla_metodos(f, x0, y0, xf, h, exacto_fn, metodos=("Euler", "Euler mejorado (Heun)")):
    """Compara metodos en x = xf: aproximacion, error absoluto y error relativo %."""
    ex = float(exacto_fn(xf))
    filas = []
    for m in metodos:
        _, ys, _ = resolver(f, x0, y0, xf, h, m)
        filas.append({"método": m, "orden": METODOS[m][1], "y(xf) aprox": float(ys[-1]), "exacto": ex,
                      "error abs": abs(ex - float(ys[-1])), "E_n (%)": error_relativo_pct(float(ys[-1]), ex)})
    return filas


def tabla_convergencia_h(f, x0, y0, xf, hs, exacto_fn, metodo="Euler"):
    """Error en xf para varios h: al dividir h entre 2 el error cae ~2 (Euler), ~4 (Heun), ~16 (RK4)."""
    ex = float(exacto_fn(xf))
    filas, previo = [], None
    for h in hs:
        _, ys, _ = resolver(f, x0, y0, xf, h, metodo)
        err = abs(ex - float(ys[-1]))
        filas.append({"h": h, "n": numero_pasos(x0, xf, h), "y(xf)": float(ys[-1]), "error abs": err,
                      "razón error previo/actual": (previo / err if previo and err else float("nan"))})
        previo = err
    return filas


# =====================================================================
# Casos de las guias de la Sesion 14
# =====================================================================
# --- GAA: (y^2 - x^2) dx + x y dy = 0, y(1) = 2, y(1.2) con h = 0.1 ---
def f_gaa14(x, y):
    return (x ** 2 - y ** 2) / (x * y)


def exacto_gaa14(x):
    return math.sqrt((7 + x ** 4) / (2 * x ** 2))     # 2 x^2 y^2 - x^4 = 7


X0_GAA14, Y0_GAA14, XF_GAA14, H_GAA14 = 1.0, 2.0, 1.2, 0.1
EXPR_GAA14 = "(x^2 - y^2)/(x*y)"

# --- GT/GP ejemplo: y' = x + y, y(0) = 1  (exacto 2 e^x - x - 1; y(1) = 3.43656) ---
def f_lineal(x, y):
    return x + y


def exacto_lineal(x):
    return 2 * math.exp(x) - x - 1

# --- GT ejercicio: y' = 2 x y, y(1) = 1, y(1.5) con h = 0.1 (exacto e^(x^2 - 1)) ---
def f_2xy(x, y):
    return 2 * x * y


def exacto_2xy(x):
    return math.exp(x ** 2 - 1)

# --- GT problema: y' = e^(x+y), y(1) = 0, y(1.2) con h = 0.1 (exacto -ln(1 + e - e^x)) ---
def f_expxy(x, y):
    return math.exp(x + y)


def exacto_expxy(x):
    return -math.log(1 + math.e - math.exp(x))


if __name__ == "__main__":
    def imprimir(titulo, filas, fmt="{:.6f}"):
        print(titulo)
        for fila in filas:
            print("  ", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in fila.items()})

    print("=== GAA: (y^2 - x^2)dx + xy dy = 0, y(1) = 2, y(1.2), h = 0.1 ===")
    for m in ("Euler", "Euler mejorado (Heun)"):
        imprimir(f"-- {m}: paso a paso", tabla_paso_a_paso(f_gaa14, 1, 2, 1.2, 0.1, m))
    imprimir("-- Comparacion en x = 1.2", tabla_metodos(f_gaa14, 1, 2, 1.2, 0.1, exacto_gaa14,
                                                       ("Euler", "Euler mejorado (Heun)", "Runge-Kutta 3", "Runge-Kutta 4")))
    print(f"exacto y(1.2) = {exacto_gaa14(1.2):.8f}")

    print("\n=== GT ejemplo: y' = x + y, y(0) = 1, y(1) ===")
    print("Euler h=0.2   :", round(euler(f_lineal, 0, 1, 1, 0.2)[1][-1], 5), "(GT: 2.97664)")
    print("Euler h=0.1   :", round(euler(f_lineal, 0, 1, 1, 0.1)[1][-1], 5), "(GT: 3.18748)")
    print("Heun  h=0.2   :", round(euler_mejorado(f_lineal, 0, 1, 1, 0.2)[1][-1], 5), "(GT: 3.40542)")
    print("exacto        :", round(exacto_lineal(1), 5))

    print("\n=== GT ejercicio: y' = 2xy, y(1) = 1, y(1.5), h = 0.1 ===")
    imprimir("", tabla_metodos(f_2xy, 1, 1, 1.5, 0.1, exacto_2xy))

    print("\n=== GT problema: y' = e^(x+y), y(1) = 0, y(1.2), h = 0.1 ===")
    imprimir("", tabla_metodos(f_expxy, 1, 0, 1.2, 0.1, exacto_expxy,
                               ("Euler", "Euler mejorado (Heun)", "Runge-Kutta 3", "Runge-Kutta 4")))
    imprimir("Heun paso a paso", tabla_paso_a_paso(f_expxy, 1, 0, 1.2, 0.1, "Euler mejorado (Heun)"))
    imprimir("RK3 paso a paso", tabla_paso_a_paso(f_expxy, 1, 0, 1.2, 0.1, "Runge-Kutta 3"))

    print("\n=== Validacion de h ===")
    try:
        resolver(f_gaa14, 1, 2, 1.2, 0.07)
    except PasoInvalidoError as exc:
        print("PasoInvalidoError ->", exc)
