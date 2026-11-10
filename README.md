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
├── simpson_solver.py                        <- lógica de la Sesión 13 (Simpson 1/3 y 3/8)
├── gaa_s13_simpson.py                       <- programa de consola de la GAA 13 (input + excepción)
├── edo_solver.py                            <- lógica de la Sesión 14 (Euler, Heun, RK3, RK4)
├── gaa_s14_edo.py                           <- programa de consola de la GAA 14
└── pages/
    ├── 1_Sesion_12_Integracion_Trapecio.py  <- interfaz de la Sesión 12
    ├── 2_Sesion_13_Integracion_Simpson.py   <- interfaz de la Sesión 13
    └── 3_Sesion_14_EDO.py                   <- interfaz de la Sesión 14
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

## Sesión 13 — Integración numérica (reglas de Simpson)

- `simpson_solver.py` — `simpson13(f, a, b, n)` (Simpson 1/3 compuesto con
  el vector de coeficientes `[1, 4, 2, 4, …, 2, 4, 1]` y un producto
  punto), `simpson38`, versiones para datos tabulados, `NNoParError`
  (excepción si n no es entero par), `cota_error_simpson13`,
  `n_minimo_simpson13`, `max_abs_cuarta_derivada` y
  `tabla_convergencia_simpson`.
- `gaa_s13_simpson.py` — programa de consola de la GAA: pide a, b y n,
  detiene la ejecución si n es impar, y muestra la tabla de convergencia
  para n ∈ {2, 4, 6, 12, 24}.
- **Caso GAA — Centro de datos:** P(t) = 10 + 5t·sen(πt/6) + e^(0.2t) kW en
  [0, 6] h. Con n = 6 (h = 1): **E ≈ 128.9212 kWh**; exacto
  55 + 180/π + 5e^1.2 = **128.8964 kWh**. Cota con M = 0.45: 0.015 kWh
  (la guía subestima M; el máximo real de |P⁽⁴⁾| es ≈ 3.433).

```bash
python simpson_solver.py       # autoprueba con los casos de la GT, GP y GAA
python gaa_s13_simpson.py      # programa de la GAA (Enter = a=0, b=6, n=6)
```
````

## Sesión 14 — Ecuaciones diferenciales ordinarias (PVI)

- `edo_solver.py` — `euler`, `euler_mejorado` (Heun: predictor + corrector),
  `rk3`, `rk4`, `resolver(f, x0, y0, xf, h, metodo)`, `tabla_paso_a_paso`,
  `tabla_metodos` (error absoluto y relativo %), `tabla_convergencia_h`,
  `construir_funcion_xy` (f(x, y) como texto) y `PasoInvalidoError`
  (si h no divide al intervalo).
- `gaa_s14_edo.py` — programa de consola de la GAA: pide f(x, y), x0, y0,
  x final y h; muestra el paso a paso de Euler y Euler mejorado y compara
  los errores relativos con el valor verdadero.
- **Caso GAA:** (y² − x²)dx + xy dy = 0, y(1) = 2 → y' = (x² − y²)/(xy),
  exacta y = √((7 + x⁴)/(2x²)). Con h = 0.1: **Euler y(1.2) ≈ 1.741278
  (error 1.90 %)**, **Euler mejorado ≈ 1.775807 (error 0.0465 %)**,
  exacto 1.774980.

```bash
python edo_solver.py     # autoprueba con los casos de la GT, GP y GAA
python gaa_s14_edo.py    # programa de la GAA (Enter = valores de la guía)
```
