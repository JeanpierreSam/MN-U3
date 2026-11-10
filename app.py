"""
app.py
------
Página de inicio de la suite de Métodos Numéricos (MN-U3). Cada sesión
vive en `pages/` como una página independiente de la misma app.

Ejecutar con:
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="Métodos Numéricos - MN-U3", page_icon="📏", layout="wide")

st.title("∫ Suite de Métodos Numéricos — Unidad 3")
st.caption("UPeU · Métodos Numéricos · Integración y EDO")

st.markdown(
    """
Elige una sesión en el menú de la izquierda:

- **Sesión 12 — Integración numérica (regla de los trapecios)**: trapecio
  simple y compuesto, regla de los rectángulos, cota del error
  |E| ≤ n·h³/12·max|f''|, n mínimo para una tolerancia y análisis de
  convergencia. Caso GAA: tráfico de red R(t) de un microservicio en
  [0, 12] s.
- **Sesión 13 — Integración numérica (reglas de Simpson)**: Simpson 1/3
  simple y compuesto (n par, coeficientes vectorizados [1,4,2,…,4,1]),
  Simpson 3/8, cota |E| ≤ (b−a)h⁴/180·max|f⁽⁴⁾|, n mínimo y convergencia.
  Caso GAA: consumo energético P(t) de un centro de datos en [0, 6] h.
- **Sesión 14 — Ecuaciones diferenciales ordinarias (PVI)**: Euler, Euler
  mejorado (Heun), Runge-Kutta de orden 3 y 4, con tablas paso a paso y
  error relativo frente a la solución exacta. Caso GAA:
  (y²−x²)dx + xy dy = 0, y(1) = 2 → y(1.2) con h = 0.1.
"""
)

st.page_link("pages/1_Sesion_12_Integracion_Trapecio.py", label="Ir a Sesión 12 — Integración (Trapecio)", icon="📏")
st.page_link("pages/2_Sesion_13_Integracion_Simpson.py", label="Ir a Sesión 13 — Integración (Simpson)", icon="📐")
st.page_link("pages/3_Sesion_14_EDO.py", label="Ir a Sesión 14 — EDO (Euler / Runge-Kutta)", icon="📈")

st.markdown("---")
st.caption("Repositorio: [JeanpierreSam/MN-U3](https://github.com/JeanpierreSam/MN-U3)")
