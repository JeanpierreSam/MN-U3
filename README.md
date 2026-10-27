# MN-U3 — Métodos Numéricos, Unidad 3

Suite de Métodos Numéricos de la **Unidad 3** del curso (UPeU): una sola
app de **Streamlit multi-página**, con un `app.py` de inicio y una
página por sesión en `pages/`. Cada sesión aporta su propio módulo de
lógica numérica pura (sin interfaz), reutilizado por su página. Misma
arquitectura que [MN-U2](https://github.com/JeanpierreSam/MN-U2).

## Estructura

```
MN-U3/
├── .gitignore
├── README.md
├── requirements.txt
├── app.py                                   <- página de inicio (entry point de Streamlit)
├── integracion_solver.py                    <- lógica de la Sesión 12 (Trapecio / Rectángulos)
└── pages/
    └── 1_Sesion_12_Integracion_Trapecio.py  <- interfaz de la Sesión 12
```

## Cómo correr la app localmente

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run app.py
```

## Cómo desplegar en Streamlit Community Cloud

1. Sube el repo a GitHub.
2. En [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Repo `JeanpierreSam/MN-U3`, rama `main`, **Main file path** `app.py`.

## Cómo agregar una nueva sesión

1. Crea su módulo de lógica pura en la raíz (p. ej. `simpson_solver.py`).
2. Crea su página en `pages/` con el siguiente número (p. ej. `pages/2_Sesion_13_....py`).
3. Agrega el enlace en `app.py` con `st.page_link`.
4. Si necesita una dependencia nueva, agrégala a `requirements.txt`.
5. `git add`, `git commit`, `git push`.

## Sesión 12 — Integración numérica (regla de los trapecios)

- `integracion_solver.py` — `trapecio_compuesto(f, a, b, n)` (función
  pedida por la GAA), `trapecio_simple`, `rectangulos` (suma de Riemann),
  `trapecio_datos` (tabla de puntos), `tabla_trapecio` (paso a paso),
  `cota_error_trapecio`, `n_minimo_trapecio`, `tabla_convergencia` y
  `construir_funcion` (f(x) escrita por el usuario).
- **Caso GAA — Tráfico de red:** R(t) = 10 + 5 sen(πt/4) + e^(0.15t) Mbps
  en [0, 12] s. Con n = 4 (h = 3): **I ≈ 160.4437 Mb**; exacto
  D = 120 + 40/π + (e^1.8 − 1)/0.15 = **166.3967 Mb**. Tabla de
  convergencia para n ∈ {4, 10, 50, 100, 1000}.
- La página **"Sesión 12"** tiene tres pestañas: GAA, Modo libre (con los
  ejemplos de la GT y la GP precargados, cota de error y n mínimo) y
  Datos tabulados.

```bash
python integracion_solver.py   # autoprueba con todos los casos de las guías
```
