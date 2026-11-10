"""
3_Sesion_14_EDO.py
------------------
Interfaz Streamlit de la Sesion 14 - Ecuaciones Diferenciales Ordinarias
(Euler, Euler mejorado/Heun, Runge-Kutta 3 y 4). Tres pestañas:

  1. GAA  - (y^2 - x^2)dx + xy dy = 0, y(1) = 2: y(1.2) con h = 0.1 por
            Euler y Euler mejorado, errores relativos vs el valor exacto.
  2. Modo libre - cualquier f(x, y), x0, y0, xf, h y metodo.
  3. GP / GT - ejemplos precargados (y' = x + y, y' = 2xy, y' = e^(x+y)).

Ejecutar con:  streamlit run app.py
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from edo_solver import (
    EXPR_GAA14, H_GAA14, METODOS, X0_GAA14, XF_GAA14, Y0_GAA14, PasoInvalidoError,
    construir_funcion_xy, exacto_2xy, exacto_expxy, exacto_gaa14, exacto_lineal, f_2xy,
    f_expxy, f_gaa14, f_lineal, resolver, tabla_comparacion, tabla_convergencia_h,
    tabla_metodos, tabla_paso_a_paso,
)

st.set_page_config(page_title="EDO - MN-U3", page_icon="📈", layout="wide")
st.title("📈 Ecuaciones Diferenciales Ordinarias — Euler y Runge-Kutta")
st.caption("Métodos Numéricos · Sesión 14 · PVI y' = f(x, y), y(x₀) = y₀ · "
           "Euler: yᵢ₊₁ = yᵢ + h·f(xᵢ, yᵢ) · Heun: yᵢ₊₁ = yᵢ + (h/2)[f(xᵢ, yᵢ) + f(xᵢ₊₁, zᵢ₊₁)], zᵢ₊₁ = yᵢ + h·f(xᵢ, yᵢ).")

FMT = {"x_n": "{:.4f}", "y_n": "{:.6f}", "exacto": "{:.6f}", "error abs": "{:.3e}", "E_n (%)": "{:.4f}",
       "y(xf) aprox": "{:.6f}", "x_i": "{:.4f}", "y_i": "{:.6f}", "k1": "{:.6f}", "k2": "{:.6f}", "k3": "{:.6f}",
       "k4": "{:.6f}", "z": "{:.6f}", "y_{i+1}": "{:.6f}"}


def fmt(df):
    return df.style.format({k: v for k, v in FMT.items() if k in df.columns})


def grafica(series, exacto_fn=None, x_rango=None):
    partes = [pd.DataFrame({"x": xs, "y": ys, "serie": nombre}) for nombre, (xs, ys) in series.items()]
    if exacto_fn is not None:
        xx = np.linspace(x_rango[0], x_rango[1], 200)
        partes.append(pd.DataFrame({"x": xx, "y": [exacto_fn(v) for v in xx], "serie": "exacto"}))
    d = pd.concat(partes, ignore_index=True)
    return alt.Chart(d).mark_line(point=True).encode(
        x=alt.X("x:Q", title="x", scale=alt.Scale(zero=False)),
        y=alt.Y("y:Q", title="y", scale=alt.Scale(zero=False)),
        color=alt.Color("serie:N", title=None)).properties(height=340)


tab_gaa, tab_libre, tab_gp = st.tabs(["📝 GAA", "🧪 Modo libre", "📚 GP / GT"])

# =====================================================================
# 1. GAA
# =====================================================================
with tab_gaa:
    st.subheader("PVI de la GAA")
    st.latex(r"(y^2-x^2)\,dx+xy\,dy=0,\quad y(1)=2\ \Rightarrow\ y'=\frac{x^2-y^2}{xy}")
    st.markdown("Solución exacta (ecuación homogénea, $y=vx$): $2x^2y^2-x^4=7\\ \\Rightarrow\\ "
                "y=\\sqrt{\\dfrac{7+x^4}{2x^2}}$")
    c1, c2, c3 = st.columns(3)
    xf = c1.number_input("x final", value=float(XF_GAA14), format="%.4f", key="xf14g")
    h = c2.number_input("h", value=float(H_GAA14), min_value=0.0001, format="%.4f", key="h14g")
    c3.write("")
    try:
        tablas = {m: tabla_paso_a_paso(f_gaa14, X0_GAA14, Y0_GAA14, xf, h, m)
                  for m in ("Euler", "Euler mejorado (Heun)")}
        comp = pd.DataFrame(tabla_metodos(f_gaa14, X0_GAA14, Y0_GAA14, xf, h, exacto_gaa14,
                                          ("Euler", "Euler mejorado (Heun)")))
    except PasoInvalidoError as exc:
        st.error(f"⛔ {exc}")
        st.stop()
    m1, m2, m3 = st.columns(3)
    m1.metric("Euler", f"{comp['y(xf) aprox'][0]:.6f}", f"E = {comp['E_n (%)'][0]:.4f} %", delta_color="off")
    m2.metric("Euler mejorado", f"{comp['y(xf) aprox'][1]:.6f}", f"E = {comp['E_n (%)'][1]:.4f} %", delta_color="off")
    m3.metric("Valor verdadero", f"{comp['exacto'][0]:.6f}")
    st.dataframe(fmt(comp), width="stretch", hide_index=True)
    for m, filas in tablas.items():
        st.markdown(f"**{m} — paso a paso**")
        st.dataframe(fmt(pd.DataFrame(filas)), width="stretch", hide_index=True)
    series = {m: resolver(f_gaa14, X0_GAA14, Y0_GAA14, xf, h, m)[:2] for m in tablas}
    st.altair_chart(grafica({**series, "exacto": (np.linspace(X0_GAA14, xf, 100),
                                                  [exacto_gaa14(v) for v in np.linspace(X0_GAA14, xf, 100)])}),
                    width="stretch")
    st.markdown("**Efecto de h** (error en x final)")
    hs = [h, h / 2, h / 4, h / 8]
    try:
        for m in ("Euler", "Euler mejorado (Heun)"):
            st.caption(m)
            st.dataframe(pd.DataFrame(tabla_convergencia_h(f_gaa14, X0_GAA14, Y0_GAA14, xf, hs, exacto_gaa14, m)),
                         width="stretch", hide_index=True)
    except PasoInvalidoError:
        st.info("Con este h, las mitades sucesivas no dividen el intervalo.")

# =====================================================================
# 2. MODO LIBRE
# =====================================================================
EJEMPLOS = {
    "GAA — (y²−x²)dx + xy dy = 0": ("(x^2 - y^2)/(x*y)", 1.0, 2.0, 1.2, 0.1, exacto_gaa14),
    "GT/GP — y' = x + y, y(0)=1 (h=0.2)": ("x + y", 0.0, 1.0, 1.0, 0.2, exacto_lineal),
    "GT — Ejercicio y' = 2xy, y(1)=1": ("2*x*y", 1.0, 1.0, 1.5, 0.1, exacto_2xy),
    "GT — Problema y' = e^(x+y), y(1)=0": ("exp(x + y)", 1.0, 0.0, 1.2, 0.1, exacto_expxy),
}
EXACTAS = {v[0]: v[5] for v in EJEMPLOS.values()}

with tab_libre:
    sel = st.selectbox("Cargar ejemplo", list(EJEMPLOS))
    expr0, x00, y00, xf0, h0, _ = EJEMPLOS[sel]
    c1, c2, c3, c4, c5 = st.columns([3, 1, 1, 1, 1])
    expr = c1.text_input("f(x, y) =", value=expr0, key=f"e_{sel}")
    x0 = c2.number_input("x₀", value=x00, format="%.4f", key=f"x0_{sel}")
    y0 = c3.number_input("y₀", value=y00, format="%.4f", key=f"y0_{sel}")
    xf_l = c4.number_input("x final", value=xf0, format="%.4f", key=f"xf_{sel}")
    h_l = c5.number_input("h", value=h0, min_value=0.0001, format="%.4f", key=f"h_{sel}")
    metodos = st.multiselect("Métodos", list(METODOS), default=["Euler", "Euler mejorado (Heun)"])
    try:
        f = construir_funcion_xy(expr)
        exacto_fn = EXACTAS.get(expr)
        res = {m: resolver(f, x0, y0, xf_l, h_l, m) for m in metodos}
    except (PasoInvalidoError, ValueError, SyntaxError, ZeroDivisionError) as exc:
        st.error(f"Revisa los datos: {exc}")
        st.stop()
    if not metodos:
        st.info("Elige al menos un método.")
        st.stop()
    for m in metodos:
        with st.expander(f"{m} (orden {METODOS[m][1]})", expanded=len(res[m][0]) <= 12):
            if exacto_fn:
                st.dataframe(fmt(pd.DataFrame(tabla_comparacion(res[m][0], res[m][1], exacto_fn))),
                             width="stretch", hide_index=True)
            st.dataframe(fmt(pd.DataFrame(tabla_paso_a_paso(f, x0, y0, xf_l, h_l, m))), width="stretch", hide_index=True)
    if exacto_fn:
        st.dataframe(fmt(pd.DataFrame(tabla_metodos(f, x0, y0, xf_l, h_l, exacto_fn, tuple(metodos)))),
                     width="stretch", hide_index=True)
    series = {m: (res[m][0], res[m][1]) for m in metodos}
    if exacto_fn:
        xx = np.linspace(x0, xf_l, 100)
        series["exacto"] = (xx, [exacto_fn(v) for v in xx])
    st.altair_chart(grafica(series), width="stretch")

# =====================================================================
# 3. GP / GT
# =====================================================================
with tab_gp:
    st.markdown("**GP 1 — Euler:** y' = x + y, y(0) = 1, y(1) con h = 0.2 y h = 0.1 (GT: 2.97664 y 3.18748; exacto 3.43656).")
    filas = []
    for hh in (0.2, 0.1):
        v = resolver(f_lineal, 0, 1, 1, hh, "Euler")[1][-1]
        filas.append({"h": hh, "y(1) Euler": v, "exacto": exacto_lineal(1), "E_n (%)": 100 * abs(exacto_lineal(1) - v) / exacto_lineal(1)})
    st.dataframe(pd.DataFrame(filas).style.format({"y(1) Euler": "{:.5f}", "exacto": "{:.5f}", "E_n (%)": "{:.2f}"}),
                 width="stretch", hide_index=True)
    st.markdown("**GP 2 — Euler mejorado:** misma EDO, h = 0.2 (GT: 3.40542; error 0.91 % vs 13.4 % de Euler).")
    st.dataframe(fmt(pd.DataFrame(tabla_comparacion(*resolver(f_lineal, 0, 1, 1, 0.2, "Euler mejorado (Heun)")[:2], exacto_lineal))),
                 width="stretch", hide_index=True)
    st.markdown("**GT Ejercicio:** y' = 2xy, y(1) = 1, y(1.5), h = 0.1  ·  **GT Problema:** y' = eˣ⁺ʸ, y(1) = 0, y(1.2), h = 0.1")
    st.dataframe(fmt(pd.DataFrame(tabla_metodos(f_2xy, 1, 1, 1.5, 0.1, exacto_2xy, tuple(METODOS)))), width="stretch", hide_index=True)
    st.dataframe(fmt(pd.DataFrame(tabla_metodos(f_expxy, 1, 0, 1.2, 0.1, exacto_expxy, tuple(METODOS)))), width="stretch", hide_index=True)

st.markdown("---")
st.caption("Basado en la Guía Teórica (GT), la Guía de Práctica (GP) y la GAA, Sesión 14 — EDO, UPeU.")
