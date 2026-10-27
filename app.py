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
st.caption("UPeU · Métodos Numéricos · Integración Numérica")

st.markdown(
    """
Elige una sesión en el menú de la izquierda:

- **Sesión 12 — Integración numérica (regla de los trapecios)**: trapecio
  simple y compuesto, regla de los rectángulos, cota del error
  |E| ≤ n·h³/12·max|f''|, n mínimo para una tolerancia y análisis de
  convergencia. Caso GAA: tráfico de red R(t) de un microservicio en
  [0, 12] s.
"""
)

st.page_link("pages/1_Sesion_12_Integracion_Trapecio.py", label="Ir a Sesión 12 — Integración (Trapecio)", icon="📏")

st.markdown("---")
st.caption("Repositorio: [JeanpierreSam/MN-U3](https://github.com/JeanpierreSam/MN-U3)")
