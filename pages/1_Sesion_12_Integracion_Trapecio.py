"""
1_Sesion_12_Integracion_Trapecio.py
-----------------------------------
Interfaz Streamlit de la Sesion 12 - Integracion Numerica (regla de los
trapecios simple y compuesta). Tres pestañas:

  1. GAA  - trafico de red R(t) en [0, 12] s: paso a paso con n = 4 y
            analisis de convergencia para n en {4, 10, 50, 100, 1000}.
  2. Modo libre - cualquier f(x), [a, b] y n, con los ejemplos de la GT
            y la GP precargados, cota de error y n minimo.
  3. Datos tabulados - trapecio sobre una tabla (x_i, y_i).

Parte de la suite MN-U3. Ejecutar con:
    streamlit run app.py
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from integracion_solver import (
    A_GAA,
    B_GAA,
    EXACTO_EJERCICIO,
    EXACTO_EXP,
    EXACTO_EXPCOS,
    EXACTO_GAA,
    EXACTO_GT_RECT,
    EXACTO_INV_X,
    MAXF2_EXPCOS,
    MAXF2_GAA,
    N_GAA,
    NS_GAA,
    TOL_EXPCOS,
    R_trafico,
    construir_funcion,
    cota_error_trapecio,
    max_abs_segunda_derivada,
    n_minimo_trapecio,
    rectangulos,
    tabla_convergencia,
    tabla_trapecio,
    trapecio_compuesto,
    trapecio_datos,
)

st.set_page_config(page_title="Integración Numérica - MN-U3", page_icon="📏", layout="wide")

st.title("∫ Integración Numérica — Regla de los Trapecios")
st.caption(
    "Métodos Numéricos · Sesión 12 · Trapecio simple I ≈ (h/2)[f(a)+f(b)], trapecio compuesto "
    "I ≈ (h/2)[f(x₀) + 2Σf(xᵢ) + f(xₙ)] y cota de error |E| ≤ n·h³/12 · max|f''|."
)


# ---------------------------------------------------------------------
# Grafica comun: f(x) + trapecios sombreados + nodos
# ---------------------------------------------------------------------
def grafica_trapecios(f, a, b, n, titulo_y="f(x)"):
    x_curva = np.linspace(a, b, 600)
    df_curva = pd.DataFrame({"x": x_curva, "y": f(x_curva) * np.ones_like(x_curva)})
    h, filas = tabla_trapecio(f, a, b, n)
    df_nodos = pd.DataFrame({"x": [r["x_i"] for r in filas], "y": [r["f(x_i)"] for r in filas]})
    esc_x = alt.Scale(domain=[float(a), float(b)], nice=False)
    capas = []
    if n <= 200:  # el poligono es ilegible con miles de trapecios
        area = alt.Chart(df_nodos).mark_area(opacity=0.25, color="#ff7f0e", line={"color": "#ff7f0e"}).encode(
            x=alt.X("x:Q", scale=esc_x), y="y:Q"
        )
        lados = alt.Chart(df_nodos).mark_rule(color="#ff7f0e", opacity=0.6).encode(x="x:Q", y="y:Q", y2=alt.datum(0))
        capas += [area, lados]
    curva = alt.Chart(df_curva).mark_line(strokeWidth=2.5, color="#1f77b4").encode(
        x=alt.X("x:Q", title="x", scale=esc_x), y=alt.Y("y:Q", title=titulo_y)
    )
    capas.append(curva)
    if n <= 200:
        capas.append(alt.Chart(df_nodos).mark_circle(size=70, color="#d62728").encode(
            x="x:Q", y="y:Q", tooltip=["x", "y"]))
    return alt.layer(*capas).properties(height=360)


def df_convergencia(filas):
    df = pd.DataFrame(filas)
    fmt = {"h": "{:.6g}", "I_trapecio": "{:.10f}", "error absoluto": "{:.3e}", "error %": "{:.6f}"}
    if "cota teórica" in df:
        fmt["cota teórica"] = "{:.3e}"
    return df, fmt


def grafica_convergencia(df):
    base = df.assign(serie="error absoluto real")[["n", "error absoluto", "serie"]].rename(columns={"error absoluto": "valor"})
    if "cota teórica" in df:
        cota = df.assign(serie="cota teórica")[["n", "cota teórica", "serie"]].rename(columns={"cota teórica": "valor"})
        base = pd.concat([base, cota], ignore_index=True)
    return alt.Chart(base).mark_line(point=True).encode(
        x=alt.X("n:Q", scale=alt.Scale(type="log"), title="n (escala log)"),
        y=alt.Y("valor:Q", scale=alt.Scale(type="log"), title="error (escala log)"),
        color=alt.Color("serie:N", title=None),
        tooltip=["n", "serie", alt.Tooltip("valor:Q", format=".3e")],
    ).properties(height=320)


tab_gaa, tab_libre, tab_datos = st.tabs(["📡 GAA — Tráfico de red", "🧪 Modo libre", "📋 Datos tabulados"])

# =====================================================================
# 1. GAA
# =====================================================================
with tab_gaa:
    st.subheader("Contexto: estimación del tráfico de red en un servidor web")
    st.latex(r"D=\int_0^{12} R(t)\,dt,\qquad R(t)=10+5\sin\!\left(\frac{\pi t}{4}\right)+e^{0.15t}\ \text{[Mbps]}")

    st.markdown("#### Parte 1 — Modelado teórico (n = 4)")
    h, filas = tabla_trapecio(R_trafico, A_GAA, B_GAA, N_GAA)
    c1, c2, c3 = st.columns(3)
    c1.metric("Ancho de paso h", f"{h:g} s")
    I4 = trapecio_compuesto(R_trafico, A_GAA, B_GAA, N_GAA)
    c2.metric("I_trapecio (n = 4)", f"{I4:.4f} Mb")
    c3.metric("Valor exacto D", f"{EXACTO_GAA:.4f} Mb", delta=f"error {abs(EXACTO_GAA - I4):.4f} Mb", delta_color="off")
    st.dataframe(pd.DataFrame(filas).style.format({"x_i": "{:g}", "f(x_i)": "{:.6f}", "c_i·f(x_i)": "{:.6f}"}),
                 width="stretch", hide_index=True)
    suma = sum(r["c_i·f(x_i)"] for r in filas)
    st.latex(
        rf"I_{{trapecio}}=\frac{{h}}{{2}}\Big[R(t_0)+2\big(R(t_1)+R(t_2)+R(t_3)\big)+R(t_4)\Big]"
        rf"=\frac{{{h:g}}}{{2}}({suma:.6f})={I4:.6f}\ \text{{Mb}}"
    )
    st.altair_chart(grafica_trapecios(R_trafico, A_GAA, B_GAA, N_GAA, "R(t) [Mbps]"), width="stretch")
    st.caption("Con n = 4 los trapecios no captan las oscilaciones del seno (período 8 s): de ahí el error de ~3.6 %.")

    st.markdown("#### Parte 2 — Análisis de convergencia")
    df, fmt = df_convergencia(tabla_convergencia(R_trafico, A_GAA, B_GAA, EXACTO_GAA, NS_GAA, MAXF2_GAA))
    st.dataframe(df.style.format(fmt), width="stretch", hide_index=True)
    st.caption(
        f"Valor exacto: D = 120 + 40/π + (e^1.8 − 1)/0.15 = {EXACTO_GAA:.6f} Mb · "
        f"max|R''(t)| ≈ {MAXF2_GAA:.4f} en [0, 12]. El error cae ~ 1/n² (orden O(h²)): "
        "al duplicar n el error se divide aprox. entre 4."
    )
    st.altair_chart(grafica_convergencia(df), width="stretch")
    st.download_button("⬇️ Descargar tabla (CSV)", df.to_csv(index=False).encode("utf-8"),
                       "convergencia_trapecio_gaa.csv", "text/csv")

# =====================================================================
# 2. MODO LIBRE
# =====================================================================
EJEMPLOS = {
    "(ninguno)": None,
    "GT - Rectángulos ∫₋₁¹ (1−x²) dx, n=4": {"f": "1 - x^2", "a": -1.0, "b": 1.0, "n": 4, "exacto": EXACTO_GT_RECT},
    "GT/GP - Trapecio simple ∫₃^3.6 1/x dx": {"f": "1/x", "a": 3.0, "b": 3.6, "n": 1, "exacto": EXACTO_INV_X},
    "GT/GP - Trapecio simple ∫₀¹ eˣ dx": {"f": "exp(x)", "a": 0.0, "b": 1.0, "n": 1, "exacto": EXACTO_EXP},
    "GT - Compuesto ∫₃^3.6 1/x dx, n=6": {"f": "1/x", "a": 3.0, "b": 3.6, "n": 6, "exacto": EXACTO_INV_X},
    "GT - Compuesto ∫₀¹ eˣ dx, n=10": {"f": "exp(x)", "a": 0.0, "b": 1.0, "n": 10, "exacto": EXACTO_EXP},
    "GP - Ejercicio ∫₀⁴ (1+x²)^(-1/2) dx (n=1, 2, 8)": {"f": "(1 + x^2)^(-1/2)", "a": 0.0, "b": 4.0, "n": 8, "exacto": EXACTO_EJERCICIO},
    "GP - Ejemplo ∫₀^1.2 eˣ cos x dx, error < 0.5e-3": {"f": "exp(x)*cos(x)", "a": 0.0, "b": 1.2, "n": 43, "exacto": EXACTO_EXPCOS,
                                                         "tol": TOL_EXPCOS, "maxf2": MAXF2_EXPCOS},
}

for _k, _v in {"f12": "exp(x)*cos(x)", "a12": 0.0, "b12": 1.2, "n12": 4, "ex12": "", "tol12": 0.0005, "mf12": 0.0}.items():
    st.session_state.setdefault(_k, _v)
pendiente = st.session_state.pop("_pending_ejemplo12", None)
if pendiente is not None:
    ej = EJEMPLOS[pendiente]
    st.session_state.update({
        "f12": ej["f"], "a12": ej["a"], "b12": ej["b"], "n12": ej["n"],
        "ex12": f"{ej['exacto']:.12g}", "tol12": ej.get("tol", 0.0005), "mf12": ej.get("maxf2", 0.0),
    })

with tab_libre:
    col_ej, col_bt = st.columns([4, 1])
    ejemplo_sel = col_ej.selectbox("Cargar ejemplo de la GT / GP", list(EJEMPLOS.keys()))
    col_bt.write("")
    if col_bt.button("Cargar", width="stretch") and EJEMPLOS[ejemplo_sel] is not None:
        st.session_state["_pending_ejemplo12"] = ejemplo_sel
        st.rerun()

    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
    expr = c1.text_input("f(x) =", key="f12", help="Ej.: exp(x)*cos(x), 1/x, (1+x^2)^(-1/2), sin(pi*x/4)")
    a = c2.number_input("a", format="%.4f", key="a12")
    b = c3.number_input("b", format="%.4f", key="b12")
    n = c4.number_input("n", min_value=1, max_value=100000, step=1, key="n12")
    exacto_txt = st.text_input("Valor exacto (opcional, para calcular el error real)", key="ex12")

    try:
        f = construir_funcion(expr)
        if a >= b:
            raise ValueError("Se requiere a < b.")
        exacto = float(exacto_txt) if exacto_txt.strip() else None
    except Exception as exc:  # noqa: BLE001 - se muestra al usuario
        st.error(f"Revisa los datos: {exc}")
        st.stop()

    h, filas = tabla_trapecio(f, a, b, int(n))
    I_simple = trapecio_compuesto(f, a, b, 1)
    I_comp = trapecio_compuesto(f, a, b, int(n))
    I_rect = rectangulos(f, a, b, int(n), "medio")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("h", f"{h:.6g}")
    m2.metric("Trapecio simple (n=1)", f"{I_simple:.6f}")
    m3.metric(f"Trapecio compuesto (n={int(n)})", f"{I_comp:.6f}")
    m4.metric(f"Rectángulos, punto medio (n={int(n)})", f"{I_rect:.6f}")

    if exacto is not None:
        df_cmp = pd.DataFrame([
            {"método": "Trapecio simple", "aproximación": I_simple},
            {"método": f"Trapecio compuesto n={int(n)}", "aproximación": I_comp},
            {"método": f"Rectángulos n={int(n)}", "aproximación": I_rect},
        ])
        df_cmp["error absoluto"] = (exacto - df_cmp["aproximación"]).abs()
        df_cmp["error relativo %"] = 100 * df_cmp["error absoluto"] / abs(exacto) if exacto else np.nan
        st.dataframe(df_cmp, width="stretch", hide_index=True)

    with st.expander("Tabla paso a paso (x_i, f(x_i), pesos)", expanded=int(n) <= 12):
        st.dataframe(pd.DataFrame(filas), width="stretch", hide_index=True)
    st.altair_chart(grafica_trapecios(f, a, b, int(n)), width="stretch")

    st.markdown("#### Cota del error y n mínimo")
    e1, e2 = st.columns(2)
    mf_estimado = max_abs_segunda_derivada(f, a, b)
    maxf2 = e1.number_input("max |f''(x)| en [a, b]  (0 = usar estimación numérica)", min_value=0.0,
                            format="%.6f", key="mf12")
    maxf2 = maxf2 if maxf2 > 0 else mf_estimado
    tol = e2.number_input("Tolerancia de error", min_value=1e-12, format="%.2e", key="tol12")
    n_min, h_max = n_minimo_trapecio(a, b, tol, maxf2)
    k1, k2, k3 = st.columns(3)
    k1.metric("max|f''| usado", f"{maxf2:.6f}", help=f"Estimación numérica: {mf_estimado:.6f}")
    k2.metric(f"Cota |E| con n={int(n)}", f"{cota_error_trapecio(a, b, int(n), maxf2):.3e}")
    k3.metric("n mínimo para la tolerancia", f"{n_min}", help=f"h ≤ {h_max:.6f}")
    st.caption("|E_TR| ≤ n·h³/12 · max|f''| = (b−a)·h²/12 · max|f''|  →  h ≤ √(12·tol / ((b−a)·max|f''|)).")

    if exacto is not None:
        st.markdown("#### Convergencia")
        ns = sorted({1, 2, 4, 8, int(n), 10, 50, 100, 1000, n_min})
        df, fmt = df_convergencia(tabla_convergencia(f, a, b, exacto, ns, maxf2))
        st.dataframe(df.style.format(fmt), width="stretch", hide_index=True)
        st.altair_chart(grafica_convergencia(df), width="stretch")

# =====================================================================
# 3. DATOS TABULADOS
# =====================================================================
with tab_datos:
    st.markdown("Cuando no se conoce f(x) analíticamente sino una **tabla de puntos** (GT, *Observaciones*). "
                "Admite espaciado no uniforme: suma de (x_{i+1} − x_i)·(y_i + y_{i+1})/2.")
    xs0 = np.linspace(3, 3.6, 7)
    df_ini = pd.DataFrame({"x": xs0, "y": np.round(1 / xs0, 4)})  # tabla de la GT (1/x, n = 6)
    df_tab = st.data_editor(df_ini, num_rows="dynamic", width="stretch", key="tabla12")
    try:
        d = df_tab.dropna().astype(float)
        I_tab = trapecio_datos(d["x"].to_numpy(), d["y"].to_numpy())
        st.metric("∫ ≈ (trapecio sobre la tabla)", f"{I_tab:.6f}")
        st.altair_chart(
            alt.Chart(d).mark_area(opacity=0.3, color="#ff7f0e", line=True, point=True).encode(x="x:Q", y="y:Q"),
            width="stretch",
        )
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))

st.markdown("---")
st.caption("Basado en la Guía Teórica (GT), la Guía de Práctica (GP) y la GAA, Sesión 12 — Integración, UPeU.")
