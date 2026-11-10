"""
gaa_s14_edo.py
--------------
Programa de consola pedido por la GAA de la Sesion 14 (EDO - Euler y Euler
mejorado). Resuelve el PVI

    (y^2 - x^2) dx + x y dy = 0,   y(1) = 2      ->   y' = (x^2 - y^2)/(x y)

aproximando y(1.2) con h = 0.1, y compara los errores relativos respecto del
valor verdadero  y(x) = sqrt((7 + x^4)/(2 x^2))  (de 2 x^2 y^2 - x^4 = 7).

Tambien acepta otra EDO escrita como f(x, y) (por ejemplo exp(x+y)); en ese
caso no hay valor exacto y solo se muestran las aproximaciones.

Ejecutar:
    python gaa_s14_edo.py          # Enter en todo = caso de la GAA
"""

import sys

from edo_solver import (
    EXPR_GAA14,
    H_GAA14,
    X0_GAA14,
    XF_GAA14,
    Y0_GAA14,
    PasoInvalidoError,
    construir_funcion_xy,
    error_relativo_pct,
    exacto_gaa14,
    resolver,
    tabla_paso_a_paso,
)


def leer(mensaje, defecto, tipo=float):
    texto = input(f"{mensaje} [{defecto}]: ").strip()
    return tipo(texto) if texto else tipo(defecto)


def main():
    print("=" * 66)
    print(" PVI  y' = f(x, y), y(x0) = y0  -  Euler y Euler mejorado (Heun)")
    print(" GAA: (y^2 - x^2) dx + x y dy = 0,  y(1) = 2,  y(1.2) con h = 0.1")
    print("=" * 66)

    try:
        expr = leer("f(x, y) = dy/dx", EXPR_GAA14, str)
        f = construir_funcion_xy(expr)
        x0 = leer("x0", X0_GAA14)
        y0 = leer("y0 = y(x0)", Y0_GAA14)
        xf = leer("x final", XF_GAA14)
        h = leer("tamaño de paso h", H_GAA14)
    except (ValueError, SyntaxError) as exc:
        print(f"\n[ERROR] Entrada invalida: {exc}")
        sys.exit(1)

    es_caso = (expr.replace(" ", "") == EXPR_GAA14.replace(" ", ""), x0, y0) == (True, X0_GAA14, Y0_GAA14)
    resultados = {}
    try:
        for metodo in ("Euler", "Euler mejorado (Heun)"):
            print(f"\n--- {metodo}: paso a paso ---")
            for fila in tabla_paso_a_paso(f, x0, y0, xf, h, metodo):
                extra = "  ".join(f"{k}={v:.6f}" for k, v in fila.items() if k not in ("i", "x_i", "y_i", "y_{i+1}"))
                print(f"i={fila['i']}  x={fila['x_i']:.4f}  y={fila['y_i']:.6f}  {extra}  ->  y_{fila['i'] + 1}={fila['y_{i+1}']:.6f}")
            resultados[metodo] = resolver(f, x0, y0, xf, h, metodo)[1][-1]
    except PasoInvalidoError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)

    print(f"\n{'Metodo':<24}{'y(xf) aprox':>14}", end="")
    if es_caso:
        print(f"{'exacto':>12}{'error abs':>12}{'E_rel (%)':>11}")
        ex = exacto_gaa14(xf)
        for m, v in resultados.items():
            print(f"{m:<24}{v:>14.6f}{ex:>12.6f}{abs(ex - v):>12.6f}{error_relativo_pct(v, ex):>11.4f}")
        mejora = abs(ex - resultados['Euler']) / abs(ex - resultados['Euler mejorado (Heun)'])
        print(f"\nEuler mejorado reduce el error unas {mejora:.1f} veces respecto de Euler.")
    else:
        print()
        for m, v in resultados.items():
            print(f"{m:<24}{v:>14.6f}")


if __name__ == "__main__":
    main()
