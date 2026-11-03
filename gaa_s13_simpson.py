"""
gaa_s13_simpson.py
------------------
Programa de consola pedido por la GAA de la Sesion 13 (Integracion -
Simpson 1/3 compuesto). Caso: consumo energetico de un centro de datos,

    P(t) = 10 + 5 t sin(pi t / 6) + e^(0.2 t)  [kW],   E = int_a^b P(t) dt  [kWh]

Requerimientos que cumple:
  1. Entrada y validacion: solicita a, b y n; si n no es entero par lanza
     NNoParError y detiene la ejecucion con un mensaje de error.
  2. Calculo optimizado: Simpson 1/3 con el vector de coeficientes
     [1, 4, 2, 4, ..., 2, 4, 1] y un producto punto (simpson_solver.simpson13).
  3. Analisis de convergencia: tabla con n, h, integral y cota |E_t| para
     n en {2, 4, 6, 12, 24}.

Ejecutar:
    python gaa_s13_simpson.py              # pide a, b, n (Enter = 0, 6, 6)
"""

import sys

from simpson_solver import (
    EXACTO_GAA13,
    M_GUIA_GAA13,
    M_REAL_GAA13,
    NS_GAA13,
    NNoParError,
    P_potencia,
    cota_error_simpson13,
    simpson13,
    tabla_convergencia_simpson,
    tabla_simpson13,
    validar_n_par,
)


def leer_float(mensaje, defecto):
    texto = input(f"{mensaje} [{defecto}]: ").strip()
    return float(texto) if texto else float(defecto)


def main():
    print("=" * 66)
    print(" Consumo energetico E = int_a^b P(t) dt  -  Simpson 1/3 compuesto")
    print(" P(t) = 10 + 5 t sin(pi t/6) + e^(0.2 t)  [kW]")
    print("=" * 66)

    # 1. Entrada y validacion -------------------------------------------
    try:
        a = leer_float("Limite inferior a (h)", 0)
        b = leer_float("Limite superior b (h)", 6)
        if b <= a:
            raise ValueError("Se requiere a < b.")
        texto_n = input("Numero de subintervalos n (entero par) [6]: ").strip() or "6"
        n = validar_n_par(texto_n)          # lanza NNoParError si n es impar o no entero
    except NNoParError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)                         # detiene la ejecucion
    except ValueError as exc:
        print(f"\n[ERROR] Entrada invalida: {exc}")
        sys.exit(1)

    # 2. Calculo optimizado (coeficientes vectorizados) -----------------
    h, filas = tabla_simpson13(P_potencia, a, b, n)
    print(f"\nh = (b - a)/n = ({b:g} - {a:g})/{n} = {h:g}")
    print(f"{'i':>3} {'t_i':>8} {'P(t_i)':>12} {'c_i':>4} {'c_i*P(t_i)':>13}")
    for r in filas:
        print(f"{r['i']:>3} {r['x_i']:>8.4f} {r['f(x_i)']:>12.6f} {r['coef c_i']:>4} {r['c_i·f(x_i)']:>13.6f}")
    suma = sum(r["c_i·f(x_i)"] for r in filas)
    I = simpson13(P_potencia, a, b, n)
    print(f"\nI = h/3 * suma = {h:g}/3 * {suma:.6f} = {I:.6f} kWh")
    print(f"Cota |E_t| <= (b-a) h^4/180 * M  con M = {M_GUIA_GAA13} (guia): "
          f"{cota_error_simpson13(a, b, n, M_GUIA_GAA13):.6f} kWh")
    if (a, b) == (0.0, 6.0):
        print(f"Valor exacto: 55 + 180/pi + 5 e^1.2 = {EXACTO_GAA13:.6f} kWh "
              f"(error real {abs(EXACTO_GAA13 - I):.6f})")
        print(f"Nota: max|P''''| en [0, 6] es ~{M_REAL_GAA13:.4f}, no 0.45; con ese M la cota es "
              f"{cota_error_simpson13(a, b, n, M_REAL_GAA13):.6f} kWh.")

    # 3. Analisis de convergencia ---------------------------------------
    exacto = EXACTO_GAA13 if (a, b) == (0.0, 6.0) else None
    print("\nAnalisis de convergencia (M = 0.45 kW/h^4, segun la guia)")
    cab = f"{'n':>4} | {'h':>7} | {'Integral (kWh)':>16} | {'Cota |E_t|':>12}"
    cab += f" | {'Error real':>12}" if exacto is not None else ""
    print(cab)
    print("-" * len(cab))
    for r in tabla_convergencia_simpson(P_potencia, a, b, M_GUIA_GAA13, NS_GAA13, exacto):
        linea = f"{r['n']:>4} | {r['h']:>7.4f} | {r['I_simpson']:>16.10f} | {r['cota |E_t|']:>12.4e}"
        linea += f" | {r['error real']:>12.4e}" if exacto is not None else ""
        print(linea)


if __name__ == "__main__":
    main()
