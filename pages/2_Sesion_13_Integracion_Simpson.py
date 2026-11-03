"""
2_Sesion_13_Integracion_Simpson.py
----------------------------------
Interfaz Streamlit de la Sesion 13 - Integracion Numerica (reglas de
Simpson 1/3 y 3/8). Tres pestañas:

  1. GAA  - consumo energetico P(t) de un centro de datos en [0, 6] h:
            Simpson 1/3 con n = 6, cota del error y convergencia para
            n en {2, 4, 6, 12, 24} (con validacion de n par).
  2. Modo libre - cualquier f(x), [a, b] y n, con los ejemplos de la GT
            y la GP precargados: trapecio vs Simpson 1/3 vs Simpson 3/8,
            cota de error y n minimo.
  3. Datos tabulados - Simpson sobre una tabla (problema del terreno).

Parte de la suite MN-U3. Ejecutar con:
    streamlit run app.py
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from integracion_solver import construir_funcion, trapecio_compuesto
from simpson_solver import (
    A_GAA13,
    B_GAA13,
    DIST_TERRENO,
    ELEV_TERRENO,
    EXACTO_EJERCICIO,
    EXACTO_EXP,
    EXACTO_EXPCOS,
    EXACTO_GAA13,
    EXACTO_INV_X,
    EXACTO_LN1,
    EXACTO_LNX,
    M_GUIA_GAA13,
    M_LN1,
    M_REAL_GAA13,
    N_GAA13,
    NS_GAA13,
    TOL_LN1,
    NNoMultiplo3Error,
    NNoParError,
    P_potencia,
    cota_error_simpson13,
    cota_error_simpson38,
    max_abs_cuarta_derivada,
    n_minimo_simpson13,
    simpson13,
    simpson13_datos,
    simpson38,
    simpson38_datos,
    tabla_convergencia_simpson,
    tabla_simpson13,
    validar_n_par,
)

st.set_page_config(page_title="Simpson - MN-U3", page_icon="📐", layout="wide")

st.title("∫ Integración Numérica — Reglas de Simpson")
st.caption(
    "Métodos Numéricos · Sesión 13 · Simpson 1/3: I ≈ (h/3)[f₀ + 4Σf_impar + 2Σf_par + fₙ] (n par) · "
    "Simpson 3/8: I ≈ (3h/8)[f₀ + 3f₁ + 3f₂ + 2f₃ + … + fₙ] (n múltiplo de 3) · |E| ≤ (b−a)h⁴/180 · max|f⁽⁴⁾|."
)


# ---------------------------------------------------------------------
# Grafica: f(x) + parabolas de Simpson sombreadas + nodos
# ---------------------------------------------------------------------
def grafica_simpson(f, a, b, n, titulo_y="f(x)"):
    x_curva = np.linspace(a, b, 600)
    df_curva = pd.DataFrame({"x": x_curva, "y": f(x_curva) * np.ones_like(x_curva)})
    h, filas = tabla_simpson13(f, a, b, n)
    xs = np.array([r["x_i"] for r in filas])
    ys = np.array([r["f(x_i)"] for r in filas])
    esc_x = alt.Scale(domain=[float(a), float(b)], nice=False)
    capas = []
    if n <= 100:
        trozos = []
        for k in range(0, n, 2):  # una parabola de Lagrange por cada par de subintervalos
            coef = np.polyfit(xs[k:k + 3], ys[k:k + 3], 2)
            xx = np.linspace(xs[k], xs[k + 2], 40)
            trozos.append(pd.DataFrame({"x": xx, "y": np.polyval(coef, xx), "bloque": k // 2}))
        df_par = pd.concat(trozos, ignore_index=True)
        capas.append(alt.Chart(df_par).mark_area(opacity=0.25, color="#2ca02c").encode(
            x=alt.X("x:Q", scale=esc_x), y="y:Q", detail="bloque:N"))
        capas.append(alt.Chart(df_par).mark_line(strokeDash=[5, 3], color="#2ca02c").encode(
            x="x:Q", y="y:Q", detail="bloque:N"))
    capas.append(alt.Chart(df_curva).mark_line(strokeWidth=2.5, color="#1f77b4").encode(
        x=alt.X("x:Q", title="x", scale=esc_x), y=alt.Y("y:Q", title=titulo_y)))
    if n <= 100:
        capas.append(alt.Chart(pd.DataFrame({"x": xs, "y": ys})).mark_circle(size=70, color="#d62728").encode(
            x="x:Q", y="y:Q", tooltip=["x", "y"]))
    return alt.layer(*capas).properties(height=360)


def grafica_convergencia(df):
    series = [df.assign(serie="cota teórica")[["n", "cota |E_t|", "serie"]].rename(columns={"cota |E_t|": "valor"})]
    if "error real" in df:
        series.append(df.assign(serie="error real")[["n", "error real", "serie"]].rename(columns={"error real": "valor"}))
    data = pd.concat(series, ignore_index=True)
    return alt.Chart(data).mark_line(point=True).encode(
        x=alt.X("n:Q", scale=alt.Scale(type="log"), title="n (escala log)"),
        y=alt.Y("valor:Q", scale=alt.Scale(type="log"), title="error (escala log)"),
        color=alt.Color("serie:N", title=None),
        tooltip=["n", "serie", alt.Tooltip("valor:Q", format=".3e")],
    ).properties(height=320)


FMT_CONV = {"h": "{:.6g}", "I_simpson": "{:.10f}", "cota |E_t|": "{:.4e}", "error real": "{:.4e}"}

tab_gaa, tab_libre, tab_datos = st.tabs(["⚡ GAA — Centro de datos", "🧪 Modo libre", "📋 Datos tabulados"])

# =====================================================================
# 1. GAA
# =====================================================================
with tab_gaa:
    st.subheader("Contexto: consumo energético de un centro de datos de microservicios")
    st.latex(r"E=\int_0^{6}P(t)\,dt,\qquad P(t)=10+5t\,\sin\!\left(\frac{\pi t}{6}\right)+e^{0.2t}\ \text{[kW]}")

    c1, c2, c3 = st.columns(3)
    a_g = c1.number_input("a (h)", value=float(A_GAA13), format="%.2f", key="a13g")
    b_g = c2.number_input("b (h)", value=float(B_GAA13), format="%.2f", key="b13g")
    n_txt = c3.text_input("n (entero par)", value=str(N_GAA13), key="n13g")
    try:
        if b_g <= a_g:
            raise ValueError("Se requiere a < b.")
        n_g = validar_n_par(n_txt)
    except NNoParError as exc:
        st.error(f"⛔ NNoParError: {exc}")
        st.stop()
    except ValueError as exc:
        st.error(f"⛔ {exc}")
        st.stop()

    es_caso = (a_g, b_g) == (A_GAA13, B_GAA13)
    st.markdown(f"#### Parte 1 — Simpson 1/3 compuesto (n = {n_g})")
    h, filas = tabla_simpson13(P_potencia, a_g, b_g, n_g)
    I = simpson13(P_potencia, a_g, b_g, n_g)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("h", f"{h:g} h")
    m2.metric(f"I_simpson (n = {n_g})", f"{I:.4f} kWh")
    m3.metric("Cota |E_t| (M = 0.45)", f"{cota_error_simpson13(a_g, b_g, n_g, M_GUIA_GAA13):.4e} kWh")
    if es_caso:
        m4.metric("Valor exacto E", f"{EXACTO_GAA13:.4f} kWh", delta=f"error {abs(EXACTO_GAA13 - I):.4f}",
                  delta_color="off")
    st.dataframe(pd.DataFrame(filas).style.format({"x_i": "{:g}", "f(x_i)": "{:.6f}", "c_i·f(x_i)": "{:.6f}"}),
                 width="stretch", hide_index=True)
    suma = sum(r["c_i·f(x_i)"] for r in filas)
    st.latex(rf"I\approx\frac{{h}}{{3}}\Big[P(t_0)+4\!\!\sum_{{i\ \text{{impar}}}}\!\!P(t_i)+2\!\!\sum_{{i\ \text{{par}}}}\!\!P(t_i)+P(t_n)\Big]"
             rf"=\frac{{{h:g}}}{{3}}({suma:.6f})={I:.6f}\ \text{{kWh}}")
    st.latex(r"|E_t|\le\frac{(b-a)\,h^4}{180}\,M")
    if es_caso:
        st.warning(
            f"⚠️ La guía da M ≈ 0.45 kW/h⁴, pero max|P⁽⁴⁾(t)| en [0, 6] es ≈ {M_REAL_GAA13:.4f} (en t ≈ 5.07). "
            f"Con M = 0.45 la cota (0.0150) es **menor** que el error real ({abs(EXACTO_GAA13 - I):.4f}), "
            f"así que no es una cota válida; con el M correcto la cota es "
            f"{cota_error_simpson13(a_g, b_g, n_g, M_REAL_GAA13):.4f} kWh."
        )
    st.altair_chart(grafica_simpson(P_potencia, a_g, b_g, n_g, "P(t) [kW]"), width="stretch")

    st.markdown("#### Parte 2 — Análisis de convergencia, n ∈ {2, 4, 6, 12, 24}")
    usar_real = st.toggle("Usar el M correcto (max|P⁽⁴⁾| numérico) en lugar de 0.45", value=False)
    M_usado = M_REAL_GAA13 if usar_real else M_GUIA_GAA13
    df = pd.DataFrame(tabla_convergencia_simpson(P_potencia, a_g, b_g, M_usado, NS_GAA13,
                                                 EXACTO_GAA13 if es_caso else None))
    st.dataframe(df.style.format({k: v for k, v in FMT_CONV.items() if k in df}), width="stretch", hide_index=True)
    st.caption(f"M usado = {M_usado:.4f}. Simpson 1/3 es de orden O(h⁴): al duplicar n el error se divide ≈ 16 "
               f"(de n = 12 a n = 24: {df['error real'].iloc[3] / df['error real'].iloc[4]:.1f}×)." if es_caso else
               f"M usado = {M_usado:.4f}.")
    st.altair_chart(grafica_convergencia(df), width="stretch")
    st.download_button("⬇️ Descargar tabla (CSV)", df.to_csv(index=False).encode("utf-8"),
                       "convergencia_simpson_gaa.csv", "text/csv")

# =====================================================================
# 2. MODO LIBRE
# =====================================================================
EJEMPLOS = {
    "(ninguno)": None,
    "GT - Simpson 1/3 ∫₃^3.6 1/x dx, n=2": {"f": "1/x", "a": 3.0, "b": 3.6, "n": 2, "exacto": EXACTO_INV_X},
    "GT - Simpson 1/3 ∫₀¹ eˣ dx, n=2": {"f": "exp(x)", "a": 0.0, "b": 1.0, "n": 2, "exacto": EXACTO_EXP},
    "GT - ∫₀^1.2 eˣ cos x dx, n=6 (1/3 y 3/8)": {"f": "exp(x)*cos(x)", "a": 0.0, "b": 1.2, "n": 6, "exacto": EXACTO_EXPCOS},
    "GT - ∫₁⁵ ln x dx, n=10": {"f": "log(x)", "a": 1.0, "b": 5.0, "n": 10, "exacto": EXACTO_LNX},
    "GT Ejercicio - ∫₁³ ln(x+1) dx, error < 1e-3": {"f": "log(x + 1)", "a": 1.0, "b": 3.0, "n": 4, "exacto": EXACTO_LN1,
                                                    "tol": TOL_LN1, "M": M_LN1},
    "GP Ejercicio - ∫₀⁴ (1+x²)^(-1/2) dx": {"f": "(1 + x^2)^(-1/2)", "a": 0.0, "b": 4.0, "n": 8, "exacto": EXACTO_EJERCICIO},
}

for _k, _v in {"f13": "exp(x)*cos(x)", "a13": 0.0, "b13": 1.2, "n13": 6, "ex13": "", "tol13": 0.001, "M13": 0.0}.items():
    st.session_state.setdefault(_k, _v)
pendiente = st.session_state.pop("_pending_ejemplo13", None)
if pendiente is not None:
    ej = EJEMPLOS[pendiente]
    st.session_state.update({
        "f13": ej["f"], "a13": ej["a"], "b13": ej["b"], "n13": ej["n"],
        "ex13": f"{ej['exacto']:.12g}", "tol13": ej.get("tol", 0.001), "M13": ej.get("M", 0.0),
    })

with tab_libre:
    col_ej, col_bt = st.columns([4, 1])
    ejemplo_sel = col_ej.selectbox("Cargar ejemplo de la GT / GP", list(EJEMPLOS.keys()))
    col_bt.write("")
    if col_bt.button("Cargar", width="stretch") and EJEMPLOS[ejemplo_sel] is not None:
        st.session_state["_pending_ejemplo13"] = ejemplo_sel
        st.rerun()

    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
    expr = c1.text_input("f(x) =", key="f13", help="Ej.: exp(x)*cos(x), log(x+1), 1/x, (1+x^2)^(-1/2)")
    a = c2.number_input("a", format="%.4f", key="a13")
    b = c3.number_input("b", format="%.4f", key="b13")
    n = c4.number_input("n", min_value=1, max_value=100000, step=1, key="n13")
    exacto_txt = st.text_input("Valor exacto (opcional)", key="ex13")

    try:
        f = construir_funcion(expr)
        if a >= b:
            raise ValueError("Se requiere a < b.")
        exacto = float(exacto_txt) if exacto_txt.strip() else None
    except Exception as exc:  # noqa: BLE001
        st.error(f"Revisa los datos: {exc}")
        st.stop()

    n = int(n)
    resultados = [{"método": f"Trapecio compuesto (n={n})", "aproximación": trapecio_compuesto(f, a, b, n)}]
    try:
        resultados.append({"método": f"Simpson 1/3 (n={n})", "aproximación": simpson13(f, a, b, n)})
    except NNoParError as exc:
        st.info(f"Simpson 1/3 no aplica: {exc}")
    try:
        resultados.append({"método": f"Simpson 3/8 (n={n})", "aproximación": simpson38(f, a, b, n)})
    except NNoMultiplo3Error as exc:
        st.info(f"Simpson 3/8 no aplica: {exc}")
    df_cmp = pd.DataFrame(resultados)
    if exacto is not None:
        df_cmp["error absoluto"] = (exacto - df_cmp["aproximación"]).abs()
        df_cmp["error relativo %"] = 100 * df_cmp["error absoluto"] / abs(exacto) if exacto else np.nan
    st.dataframe(df_cmp.style.format({"aproximación": "{:.8f}", "error absoluto": "{:.3e}", "error relativo %": "{:.5f}"}),
                 width="stretch", hide_index=True)

    if n % 2 == 0:
        with st.expander("Tabla paso a paso Simpson 1/3 (x_i, f(x_i), coeficientes)", expanded=n <= 12):
            st.dataframe(pd.DataFrame(tabla_simpson13(f, a, b, n)[1]), width="stretch", hide_index=True)
        st.altair_chart(grafica_simpson(f, a, b, n), width="stretch")

    st.markdown("#### Cota del error y n mínimo (Simpson 1/3)")
    e1, e2 = st.columns(2)
    M_est = max_abs_cuarta_derivada(f, a, b)
    M = e1.number_input("max |f⁽⁴⁾(x)| en [a, b]  (0 = estimación numérica)", min_value=0.0, format="%.6f", key="M13")
    M = M if M > 0 else M_est
    tol = e2.number_input("Tolerancia", min_value=1e-12, format="%.2e", key="tol13")
    n_min, h_max = n_minimo_simpson13(a, b, tol, M)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("M usado", f"{M:.6f}", help=f"Estimación numérica: {M_est:.6f}")
    k2.metric(f"Cota 1/3 (n={n})", f"{cota_error_simpson13(a, b, n, M):.3e}" if n % 2 == 0 else "—")
    k3.metric(f"Cota 3/8 (n={n})", f"{cota_error_simpson38(a, b, n, M):.3e}" if n % 3 == 0 else "—")
    k4.metric("n mínimo (par)", f"{n_min}", help=f"h ≤ {h_max:.6f}")
    st.caption("|E_S| ≤ n·h⁵/180·M = (b−a)·h⁴/180·M  →  h ≤ ⁴√(180·tol / ((b−a)·M)), y n se redondea al par siguiente.")

    if exacto is not None:
        st.markdown("#### Convergencia Simpson 1/3")
        ns = sorted({2, 4, 6, 12, 24, 48, n_min} | ({n} if n % 2 == 0 else set()))
        df = pd.DataFrame(tabla_convergencia_simpson(f, a, b, M, ns, exacto))
        st.dataframe(df.style.format(FMT_CONV), width="stretch", hide_index=True)
        st.altair_chart(grafica_convergencia(df), width="stretch")

# =====================================================================
# 3. DATOS TABULADOS
# =====================================================================
with tab_datos:
    st.markdown("**GT — Problema del terreno:** elevaciones cada 5 m en 30 m. Área bajo el perfil "
                "(datos equiespaciados; Simpson 1/3 necesita n par y 3/8 n múltiplo de 3 — aquí n = 6 sirve para ambos).")
    df_ini = pd.DataFrame({"x": DIST_TERRENO, "y": ELEV_TERRENO})
    df_tab = st.data_editor(df_ini, num_rows="dynamic", width="stretch", key="tabla13")
    try:
        d = df_tab.dropna().astype(float)
        xs, ys = d["x"].to_numpy(), d["y"].to_numpy()
        pasos = np.diff(xs)
        if len(xs) < 3 or np.any(pasos <= 0) or not np.allclose(pasos, pasos[0]):
            raise ValueError("Se requieren ≥ 3 abscisas crecientes y equiespaciadas.")
        h_t = float(pasos[0])
        n_t = len(xs) - 1
        r1, r2, r3 = st.columns(3)
        r1.metric("Trapecio", f"{float(np.sum(h_t * (ys[:-1] + ys[1:]) / 2)):.4f}")
        r2.metric("Simpson 1/3", f"{simpson13_datos(ys, h_t):.4f}" if n_t % 2 == 0 else "n impar")
        r3.metric("Simpson 3/8", f"{simpson38_datos(ys, h_t):.4f}" if n_t % 3 == 0 else "n no múltiplo de 3")
        st.altair_chart(alt.Chart(d).mark_area(opacity=0.3, color="#2ca02c", line=True, point=True).encode(
            x=alt.X("x:Q", title="distancia (m)"), y=alt.Y("y:Q", title="elevación (m)")), width="stretch")
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))

st.markdown("---")
st.caption("Basado en la Guía Teórica (GT), la Guía de Práctica (GP) y la GAA, Sesión 13 — Integración, UPeU.")
