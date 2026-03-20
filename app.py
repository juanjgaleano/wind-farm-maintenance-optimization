import streamlit as st
import pyomo.environ as pyo

import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time, os, json

def build_model(data):

    model = pyo.ConcreteModel()
    model.constraints = pyo.ConstraintList()

    # Sets
    model.I = pyo.Set(initialize=data["I"])
    model.K = pyo.Set(initialize=data["K"])
    model.T = pyo.RangeSet(1, data["T"])
    model.W = pyo.RangeSet(1, (data["T"] - data["L"] + 1))

    # Params:
    W_t = {
        w: list(range(w, w + data["L"])) for w in model.W
    }

    cw = {
        i: sum(data["ck"][f"{k}"] * data["a"].get(f"{i},{k}", 0) for k in model.K) / data["r"][f"{i}"]
        for i in model.I
    }

    N_f = data["_meta"]['n_turbines']

    # Decision Variables
    model.s = pyo.Var(model.I, model.T, domain=pyo.Binary)
    model.m = pyo.Var(model.I, model.T, domain=pyo.Binary)
    model.e = pyo.Var(model.W, domain=pyo.Binary)
    model.p = pyo.Var(model.I, model.T, domain=pyo.NonNegativeReals)
    model.h = pyo.Var(model.I, model.T, domain=pyo.NonNegativeReals)

    # Objetive Function:
    def objetive_minimize_cost(model, data):
        total_cost = 0

        # Emergency
        for w in model.W:
            total_cost += data["ce"] * model.e[w]

        # Maintenance
        for i in model.I:
            for t in model.T:
                for k in model.K:
                    total_cost += data["ck"][f"{k}"] * data["a"].get(f"{i},{k}", 0) * model.m[i, t]

        # Failure
        for i in model.I:
            for t in model.T:
                total_cost += data["cf"] * model.s[i, t]
        
        # Loss
        for i in model.I:
            for t in model.T:
                total_cost += cw[i] * model.p[i, t]

        return total_cost

    model.objetive_minimize_cost = pyo.Objective(
        rule=lambda m: objetive_minimize_cost(m, data),
        sense=pyo.minimize
    )

    # Constraints:

    # C1: Resource Availability
    for k in model.K:
        for t in model.T:
            model.constraints.add(
                sum(data["a"].get(f"{i},{k}", 0) * model.m[i, t] for i in model.I) <= data["R"][f"{k},{t}"]
            )

    # C2, C3 y C4: Health
    for t in model.T:
        for i in model.I:
            
            # Initial
            if t == 1:
                h_prev = data["Smax"][f"{i}"]
            else:
                h_prev = model.h[i, t - 1]
                
            # Health Dynamics (Include Loss)
            model.constraints.add(
                model.h[i, t] + model.p[i, t] == h_prev + (data["r"][f"{i}"] * model.m[i, t]) - (data["delta"][f"{i}"] * (1 - model.m[i, t]))
            )
            
            # Exceed Maximum Health is Forbidden
            model.constraints.add(
                model.h[i, t] <= data["Smax"][f"{i}"]
            )
            
            # Under Threshold Declare Fail
            model.constraints.add(
                model.h[i, t] >= data["Smin"][f"{i}"] - (data["M"] * model.s[i, t])
            )

    # C5: Force emergency fleet activation if simultaneous failures reach the Fmax threshold
    for t in model.T:
        model.constraints.add(
            sum(model.s[i, t] for i in model.I) <= (data["Fmax"] - 1) + N_f * sum(model.e[w] for w in model.W if t in W_t[w])
        )

    # C6: Prevent overlapping deployments of the emergency fleet (maximum one active at a time)
    for t in model.T:
        model.constraints.add(
            sum(model.e[w] for w in model.W if t in W_t[w]) <= 1
        )

    return model

# Solver
def solve_instance(data, time_limit=None):
    model = build_model(data)
    solver = pyo.SolverFactory("appsi_highs")
    if time_limit is not None:
        solver.options["time_limit"] = time_limit
    results = solver.solve(model)

    # Termination status
    term = results.solver.termination_condition
    if term == pyo.TerminationCondition.optimal:
        status = "optimal"
    elif term == pyo.TerminationCondition.maxTimeLimit:
        status = "feasible"
    else:
        status = "infeasible"

    # Cost
    cost = pyo.value(model.objetive_minimize_cost)

    # Maintenance Plan
    maintenance = pd.DataFrame(index=model.I, columns=model.T)
    for i in model.I:
        for t in model.T:
            maintenance.loc[i, t] = "X" if pyo.value(model.m[i, t]) > 0.5 else ""

    # Emergency Plan
    emergency = pd.DataFrame(index=["Flota"], columns=model.T)
    for t in model.T:
        emergency.loc["Flota", t] = ""

    for w in model.W:
        if pyo.value(model.e[w]) > 0.5:
            for t in range(w, w + data["L"]):
                emergency.loc["Flota", t] = "E"

    # Turbine Health and Loss
    health = pd.DataFrame(index=model.I, columns=model.T)
    loss = pd.DataFrame(index=model.I, columns=model.T)
    
    for i in model.I:
        for t in model.T:
            health.loc[i, t] = round(pyo.value(model.h[i, t]), 2)
            loss.loc[i, t] = round(pyo.value(model.p[i, t]), 2)

    # Results
    return {
        "cost": cost,
        "maintenance": maintenance,
        "emergency": emergency,
        "health": health,
        "loss": loss,
        "status": status,
    }


st.set_page_config(page_title="Optimización Eólica", layout="wide")

# CSS Avanzado: Fuente, limpieza, ajuste de márgenes y expansión de pestañas
st.markdown("""
    <style>
        /* Limpiar interfaz nativa */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Ajustar espaciado superior */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 0rem !important;
        }
        
        /* Expandir las pestañas para llenar el espacio vacío a la derecha */
        div[data-baseweb="tab-list"] {
            display: flex;
            width: 100%;
        }
        div[data-baseweb="tab"] {
            flex-grow: 1;
            justify-content: center;
        }
    </style>
""", unsafe_allow_html=True)

# --- BARRA SUPERIOR: LOGO Y TÍTULO CENTRADOS ---
st.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 10px;">
    <img src="app/static/logo.png" style="height: 55px; width: auto;">
    <div style="flex: 1; text-align: center; font-size: 1.1rem; font-weight: bold; color: #333;">
        SISTEMA DE PLANEACIÓN DE MANTENIMIENTO DE ENERGÍA EÓLICA
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# --- PESTAÑAS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Contexto Problema", 
    "Descripción", 
    "Modelo Matemático", 
    "Código Python",
    "Solver",
    "Análisis Resultados",
])

# --- 1. CONTEXTO PROBLEMA ---
with tab1:
    col_img, col_txt = st.columns([1, 2])
    
    with col_img:
        st.image("assets/eolico_off.jpg", use_container_width=True)
    
    with col_txt:
        st.markdown("""
        Los parques eólicos marinos representan una parte considerable de la transición energética global. Al instalar aerogeneradores sobre plataformas fijas o flotantes en el océano, se logra una mayor eficiencia en la generación de electricidad a gran escala, minimizando al mismo tiempo el impacto visual y acústico en zonas habitadas. Esta tecnología plantea desafíos de ingeniería complejos, como la resistencia a la corrosión salina y la integración de sistemas de transmisión submarina para llevar la energía hasta la costa.

        A nivel mundial, la capacidad operativa de la eólica marina está por encima de los 83 GW, con proyecciones que apuntan a triplicar esta cifra para 2030. Actualmente, existen aproximadamente 1,555 proyectos eólicos marinos en diversas etapas (operativos, en construcción o en desarrollo avanzado) distribuidos principalmente en 44 países, lo que demuestra que la tecnología ha dejado de ser exclusiva del Mar del Norte para convertirse en un fenómeno global.

        Considere la operación de un parque eólico marino compuesto por múltiples aerogeneradores ubicados a varios kilómetros de la costa. Este parque representa una fracción significativa del suministro energético renovable de una región industrial. Debido a su localización en mar abierto, las operaciones de mantenimiento de las turbinas requieren embarcaciones especializadas y cuadrillas técnicas, cuya disponibilidad diaria es limitada y genera costos significativos. Cada intervención de mantenimiento consume ciertos recursos técnicos (barcos, personal, equipos), y la suma de los recursos utilizados en un día no puede exceder la disponibilidad correspondiente.

        Cada turbina experimenta un proceso de degradación progresiva de su estado de salud. Si no recibe mantenimiento en un día determinado, su nivel de salud disminuye según una tasa conocida; si es intervenida, su estado se recupera hasta un máximo permitido. Cuando el nivel de salud cae por debajo de un umbral mínimo, la turbina se considera fallada y deja de operar, generando un costo asociado.

        Además, existe un número máximo tolerable de turbinas falladas por día; si este umbral se supera, el sistema incurre en penalizaciones adicionales. Para enfrentar situaciones críticas, la empresa puede activar una flota de emergencia que incrementa la capacidad operativa disponible. Sin embargo, su activación implica un costo fijo muy alto y, una vez programada, debe mantenerse activa durante al menos un periodo mínimo de tiempo. Esta decisión es estratégica, pues permite evitar penalizaciones por exceso de fallas, pero incrementa significativamente los costos totales.

        Se requiere formular un modelo de optimización que determine, para cada día del horizonte de planeación, qué turbinas intervenir y si se activa la flota de emergencia, de manera que se minimice el costo total del sistema. El modelo debe considerar la dinámica temporal del estado de las turbinas, las restricciones de capacidad de recursos, la definición de falla y las condiciones de activación mínima de la flota de emergencia para un horizonte de planeación específico.
        """)


# --- 2. VERBALIZACIÓN DEL PROBLEMA ---
with tab2:
    st.markdown("""
    ### ***Conjuntos***
    El problema se enmarca en un sistema compuesto por:
    * **Horizonte de Planificación:** Un periodo de tiempo discreto (días) sobre el cual se toman las decisiones.
    * **Aerogeneradores:** La flota de turbinas individuales que componen el parque eólico.
    * **Recursos:** Los diferentes tipos de insumos, personal especializado o embarcaciones necesarias para realizar intervenciones.
    * **Ventanas de Emergencia:** Los subconjuntos de días posibles en los que se puede iniciar el despliegue de la flota de contingencia.

    ### ***Decisiones***
    * **Plan de Mantenimiento:** Qué turbina intervenir en qué día específico del horizonte.
    * **Estado de Falla:** Identificar lógicamente qué turbinas han cruzado el umbral de falla en un día dado.
    * **Despliegue de Emergencia:** En qué día exacto inicia sus operaciones la flota de emergencia.
    * **Nivel de Salud:** Rastrear el estado operativo exacto de cada turbina a lo largo del tiempo.
    * **Salud Desperdiciada:** Cuantificar la recuperación teórica que se pierde si una turbina recibe mantenimiento cuando ya está cerca de su límite máximo de salud.

    ### ***Objetivo*** 
    Minimizar el costo total operativo, compuesto por la activación de la flota de emergencia, la ejecución de mantenimientos, las penalizaciones por falla y la penalización por la salud desperdiciada.

    ### ***Restricciones***
    ***1. Disponibilidad de Recursos:*** El consumo total de cada recurso en cada día no puede exceder la cantidad disponible.

    ***2. Dinámica de Salud:*** El estado de salud actual más el desperdicio equivale a la salud del día anterior, más la recuperación en caso de mantenimiento, menos la degradación en caso de operación normal.

    ***3. Límite Máximo de Salud:*** La salud de un aerogenerador no puede superar su condición máxima de diseño.

    ***4. Declaración de Falla:*** Si la salud de un aerogenerador cae por debajo de su umbral mínimo de operación, se declara que el aerogenerador falló.

    ***5. Activación de la Flota de Emergencia:*** Si el número de fallas simultáneas supera el umbral permitido, se obliga al modelo a tener una flota de emergencia activa que cubra ese día.

    ***6. Activación No Solapada:*** Como máximo, solo puede haber un despliegue de la flota de emergencia activo en cualquier día.
    """)

# --- 3. MODELO MATEMÁTICO ---
with tab3:
    st.markdown(r"""
    ## ***Conjuntos***
    * $I$: Conjunto de aerogeneradores.
    * $K$: Conjunto de recursos.$T$: Conjunto de días en el horizonte de planificación.
    * $W$: Conjunto de días de inicio posibles para la flota de emergencia, donde $w \in \{1, 2, ..., |T| - L + 1\}$.
    * $W_t$: Subconjunto de días de inicio $w$ tales que la flota de emergencia esté activa en el día $t$.

    ## ***Parámetros***
    * $cf$: Costo de que un aerogenerador incurra en una falla.
    * $ce$: Costo de activar la flota de emergencia.
    * $F_{max}$: Número máximo de turbinas que pueden fallar simultáneamente sin decretar emergencia.
    * $L$: Número de días consecutivos que debe permanecer activa la flota de emergencia tras su activación.
    * $\delta_{i}$: Degradación del aerogenerador $i$ después de operar cada día sin mantenimiento.
    * $r_{i}$: Recuperación de salud teórica del aerogenerador $i$ después del mantenimiento.
    * $S_{min,i}$: Condición de salud mínima del aerogenerador $i$ para no ser declarado en falla.
    * $S_{max,i}$: Condición de salud máxima permitida del aerogenerador $i$.
    * $ck_{k}$: Costo unitario del recurso $k$.
    * $R_{kt}$: Cantidad disponible del recurso $k$ en el día $t$.
    * $a_{ik}$: Requerimiento del recurso $k$ para realizar el mantenimiento al aerogenerador $i$.
    * $cw_{i}$: Costo de penalización por salud desperdiciada en mantenimientos prematuros para el aerogenerador $i$, calculado como $\sum_{k \in K} (ck_k \cdot a_{ik}) / r_i$.
    * $M$: Un número suficientemente grande (Big-M) usado para la restricción de falla.
    * $N_{f}$: Número total de aerogeneradores en el parque ($|I|$), usado como Big-M para la restricción de emergencia.

    ## ***Variables de Decisión***
    ### ***Binarias:***
    * $s_{it} \in \{0, 1\}$: Toma el valor de 1 si el aerogenerador $i$ falla en el día $t$, 0 en caso contrario.
    * $m_{it} \in \{0, 1\}$: Toma el valor de 1 si se le realiza mantenimiento al aerogenerador $i$ en el día $t$, 0 en caso contrario.
    * $e_{w} \in \{0, 1\}$: Toma el valor de 1 si se activa la flota de emergencia a partir del día $w$, 0 en caso contrario.

    ### ***Continuas:***
    * $h_{it} \ge 0$: Nivel de salud (estado operativo) del aerogenerador $i$ al final del día $t$.
    * $p_{it} \ge 0$: Salud desperdiciada (pérdida/exceso de recuperación) del aerogenerador $i$ en el día $t$ debido al límite máximo.

    ## ***Función Objetivo***
    Minimizar el costo total operativo, compuesto por la activación de la flota de emergencia, la ejecución de mantenimientos, las penalizaciones por falla y la penalización por la salud desperdiciada:

    $$\min Z = \sum_{w \in W} ce \cdot e_{w} + \sum_{i \in I} \sum_{t \in T} \sum_{k \in K} (ck_{k} \cdot a_{ik}) \cdot m_{it} + \sum_{i \in I} \sum_{t \in T} cf \cdot s_{it} + \sum_{i \in I} \sum_{t \in T} cw_{i} \cdot p_{it}$$

    ## ***Restricciones***
    ***1. Disponibilidad de Recursos:***

    El consumo total de cada recurso $k$ en un día $t$ no puede exceder la cantidad disponible.

    $$\sum_{i \in I} a_{ik} \cdot m_{it} \le R_{kt} \quad \forall k \in K, \forall t \in T$$

    ***2. Dinámica de Salud:***

    El estado de salud actual más el desperdicio equivale a la salud del día anterior, más la recuperación en caso de mantenimiento, menos la degradación en caso de operación normal. (Nota: Para $t=1$, se asume que $h_{i, 0} = S_{max, i}$).

    $$h_{it} + p_{it} = h_{i, t-1} + r_{i} \cdot m_{it} - \delta_{i} \cdot (1 - m_{it}) \quad \forall i \in I, \forall t \in T$$

    ***3. Límite Máximo de Salud:***

    La salud de un aerogenerador no puede superar su condición máxima de diseño.

    $$h_{it} \le S_{max,i} \quad \forall i \in I, \forall t \in T$$

    ***4. Declaración de Falla:***

    Si la salud de un aerogenerador cae por debajo de su umbral mínimo de operación, la variable de falla ($s_{it}$) se ve forzada a tomar el valor de 1.

    $$h_{it} \ge S_{min,i} - M \cdot s_{it} \quad \forall i \in I, \forall t \in T$$

    ***5. Activación de la Flota de Emergencia:***

    Si el número de fallas simultáneas supera el umbral permitido ($F_{max}-1$), se obliga al modelo a tener una flota de emergencia activa que cubra el día $t$.

    $$\sum_{i \in I} s_{it} \le (F_{max} - 1) + M_{flota} \cdot \sum_{w \in W_t} e_{w} \quad \forall t \in T$$

    ***6. Activación No Solapada:***

    Como máximo, solo puede haber un despliegue de la flota de emergencia activo en cualquier día $t$.

    $$\sum_{w \in W_t} e_{w} \le 1 \quad \forall t \in T$$
    """)

# --- 4. CÓDIGO PYTHON ---
with tab4:
    st.markdown("### Construcción del Modelo (Pyomo)")
    
    # Aquí definimos el código en una cadena de texto sin procesar (triple comilla)
    pyomo_code = """
    import pyomo.environ as pyo
    import pandas as pd

    def build_model(data):

        model = pyo.ConcreteModel()
        model.constraints = pyo.ConstraintList()

        # Sets
        model.I = pyo.Set(initialize=data["I"])
        model.K = pyo.Set(initialize=data["K"])
        model.T = pyo.RangeSet(1, data["T"])
        model.W = pyo.RangeSet(1, (data["T"] - data["L"] + 1))

        # Params:
        W_t = {
            w: list(range(w, w + data["L"])) for w in model.W
        }

        cw = {
            i: sum(data["ck"][f"{k}"] * data["a"].get(f"{i},{k}", 0) for k in model.K) / data["r"][f"{i}"]
            for i in model.I
        }

        N_f = data["_meta"]['n_turbines']

        # Decision Variables
        model.s = pyo.Var(model.I, model.T, domain=pyo.Binary)
        model.m = pyo.Var(model.I, model.T, domain=pyo.Binary)
        model.e = pyo.Var(model.W, domain=pyo.Binary)
        model.p = pyo.Var(model.I, model.T, domain=pyo.NonNegativeReals)
        model.h = pyo.Var(model.I, model.T, domain=pyo.NonNegativeReals)

        # Objetive Function:
        def objetive_minimize_cost(model, data):
            total_cost = 0

            # Emergency
            for w in model.W:
                total_cost += data["ce"] * model.e[w]

            # Maintenance
            for i in model.I:
                for t in model.T:
                    for k in model.K:
                        total_cost += data["ck"][f"{k}"] * data["a"].get(f"{i},{k}", 0) * model.m[i, t]

            # Failure
            for i in model.I:
                for t in model.T:
                    total_cost += data["cf"] * model.s[i, t]
            
            # Loss
            for i in model.I:
                for t in model.T:
                    total_cost += cw[i] * model.p[i, t]

            return total_cost

        model.objetive_minimize_cost = pyo.Objective(
            rule=lambda m: objetive_minimize_cost(m, data),
            sense=pyo.minimize
        )

        # Constraints:

        # C1: Resource Availability
        for k in model.K:
            for t in model.T:
                model.constraints.add(
                    sum(data["a"].get(f"{i},{k}", 0) * model.m[i, t] for i in model.I) <= data["R"][f"{k},{t}"]
                )

        # C2, C3 y C4: Health
        for t in model.T:
            for i in model.I:
                
                # Initial
                if t == 1:
                    h_prev = data["Smax"][f"{i}"]
                else:
                    h_prev = model.h[i, t - 1]
                    
                # Health Dynamics (Include Loss)
                model.constraints.add(
                    model.h[i, t] + model.p[i, t] == h_prev + (data["r"][f"{i}"] * model.m[i, t]) - (data["delta"][f"{i}"] * (1 - model.m[i, t]))
                )
                
                # Exceed Maximum Health is Forbidden
                model.constraints.add(
                    model.h[i, t] <= data["Smax"][f"{i}"]
                )
                
                # Under Threshold Declare Fail
                model.constraints.add(
                    model.h[i, t] >= data["Smin"][f"{i}"] - (data["M"] * model.s[i, t])
                )

        # C5: Force emergency fleet activation if simultaneous failures reach the Fmax threshold
        for t in model.T:
            model.constraints.add(
                sum(model.s[i, t] for i in model.I) <= (data["Fmax"] - 1) + N_f * sum(model.e[w] for w in model.W if t in W_t[w])
            )

        # C6: Prevent overlapping deployments of the emergency fleet (maximum one active at a time)
        for t in model.T:
            model.constraints.add(
                sum(model.e[w] for w in model.W if t in W_t[w]) <= 1
            )

        return model
        """
    
    # Esta función de Streamlit renderiza la caja de código interactiva
    st.code(pyomo_code)

# --- 5. SOLVER ---
with tab5:
 
    # ── CSS minimalista ────────────────────────────────────────
    st.markdown("""
    <style>
        /* KPI cards */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 1px;
            background: #e0e0e0;
            border: 1px solid #e0e0e0;
            margin-bottom: 32px;
        }
        .kpi-cell {
            background: #ffffff;
            padding: 18px 16px 14px;
        }
        .kpi-label {
            font-size: 0.70rem;
            letter-spacing: 0.10em;
            text-transform: uppercase;
            color: #999;
            margin-bottom: 6px;
        }
        .kpi-value {
            font-size: 1.65rem;
            font-weight: 700;
            color: #1a1a1a;
            line-height: 1;
        }
        .kpi-sub {
            font-size: 0.72rem;
            color: #bbb;
            margin-top: 4px;
        }
 
        /* Status pill */
        .status-row {
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 24px;
        }
        .status-pill {
            font-size: 0.75rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            padding: 3px 14px;
            border-radius: 2px;
            font-weight: 600;
        }
        .status-meta {
            font-size: 0.78rem;
            color: #999;
        }
 
        /* Section headers */
        .section-title {
            font-size: 0.72rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #666;
            border-bottom: 1px solid #e8e8e8;
            padding-bottom: 6px;
            margin-bottom: 16px;
            margin-top: 32px;
        }
 
        /* Legend */
        .legend-row {
            display: flex;
            gap: 24px;
            font-size: 0.76rem;
            color: #777;
            margin-bottom: 12px;
        }
        .legend-dot {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 1px;
            vertical-align: middle;
            margin-right: 5px;
        }
    </style>
    """, unsafe_allow_html=True)
 
    # ── Selector + botón ──────────────────────────────────────
    st.markdown("Seleccione una instancia de datos (JSON) para ejecutar la optimización.")
 
    data_folder = "data"
 
    if not os.path.exists(data_folder):
        st.warning(f"No se encontró la carpeta '{data_folder}'.")
    else:
        json_files = [f for f in os.listdir(data_folder) if f.endswith('.json')]
 
        if not json_files:
            st.warning(f"La carpeta '{data_folder}' está vacía.")
        else:
            TIME_OPTIONS = {
                "5 minutos": 300,
                "10 minutos": 600,
                "15 minutos": 900,
                "Hasta el óptimo": None,
            }

            c1, c2, c3 = st.columns([3, 2, 1])
            with c1:
                selected_file = st.selectbox("Instancia", json_files, label_visibility="collapsed")
            with c2:
                selected_time_label = st.selectbox(
                    "Tiempo límite",
                    list(TIME_OPTIONS.keys()),
                    index=3,
                    label_visibility="collapsed",
                )
            with c3:
                run = st.button("Ejecutar", type="primary", use_container_width=True)
 
            if run:
                file_path = os.path.join(data_folder, selected_file)
                with open(file_path) as f:
                    instance_data = json.load(f)

                selected_time = TIME_OPTIONS[selected_time_label]
                spinner_msg = (
                    f"Resolviendo con HiGHS (límite: {selected_time_label})..."
                    if selected_time is not None
                    else "Resolviendo con HiGHS hasta el óptimo..."
                )
 
                with st.spinner(spinner_msg):
                    t0  = time.time()
                    res = solve_instance(instance_data, time_limit=selected_time)
                    elapsed = time.time() - t0

                    st.session_state["res"] = res
                    st.session_state["instance_data"] = instance_data
                    st.session_state["elapsed"] = elapsed

            
            if "res" in st.session_state:
                res = st.session_state["res"]
                instance_data = st.session_state["instance_data"]
                elapsed = st.session_state["elapsed"]
                    
                # Resultados base
                df_maint  = res['maintenance']
                df_emerg  = res['emergency']
                df_health = res['health']
                df_loss   = res['loss']
 
                turbines  = df_health.index.tolist()
                days      = [int(c) for c in df_health.columns]
 
                # ── STATUS ────────────────────────────────────
                status = res.get("status", "optimal")
                pill_styles = {
                    "optimal":    "background:#1a1a1a; color:#fff;",
                    "feasible":   "background:#e8e0d0; color:#5a4a2a;",
                    "infeasible": "background:#f0e0e0; color:#7a2020;",
                }
                pill_labels = {
                    "optimal":    "Optimo global",
                    "feasible":   "Solución factible",
                    "infeasible": "Infactible",
                }
                st.markdown(f"""
                <div class="status-row">
                    <span class="status-pill" style="{pill_styles.get(status, '')}">
                        {pill_labels.get(status, status)}
                    </span>
                    <span class="status-meta">
                        {selected_file} &nbsp;&nbsp; {elapsed:.2f} s
                    </span>
                </div>
                """, unsafe_allow_html=True)
 
                # ── KPIs ──────────────────────────────────────
                n_maint    = (df_maint == "X").sum().sum()
                n_emerg    = (df_emerg == "E").sum().sum()
                n_fail     = int(res.get("n_failures", 0))
                health_avg = df_health.iloc[:, -1].mean() if not df_health.empty else 0
                total_loss = float(df_loss.values.sum()) if not df_loss.empty else 0.0
 
                st.markdown(f"""
                <div class="kpi-grid">
                    <div class="kpi-cell">
                        <div class="kpi-label">Costo total</div>
                        <div class="kpi-value">${res['cost']:,.0f}</div>
                    </div>
                    <div class="kpi-cell">
                        <div class="kpi-label">Mantenimientos</div>
                        <div class="kpi-value">{n_maint}</div>
                        <div class="kpi-sub">intervenciones</div>
                    </div>
                    <div class="kpi-cell">
                        <div class="kpi-label">Emergencias</div>
                        <div class="kpi-value">{n_emerg}</div>
                        <div class="kpi-sub">días activos</div>
                    </div>
                    <div class="kpi-cell">
                        <div class="kpi-label">Salud final prom.</div>
                        <div class="kpi-value">{health_avg:.1f}</div>
                    </div>
                    <div class="kpi-cell">
                        <div class="kpi-label">Salud desperdiciada</div>
                        <div class="kpi-value">{total_loss:.1f}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
 
                # ── GRÁFICO DE BARRAS POR TURBINA ─────────────
                st.markdown('<div class="section-title">Salud por aerogenerador</div>', unsafe_allow_html=True)
 
                selected_turbine = st.selectbox(
                    "Aerogenerador",
                    turbines,
                    format_func=lambda x: f"Turbina {x}",
                    label_visibility="collapsed",
                    key="turbine_selector",
                )
 
                smin_vals = instance_data.get("Smin", {})
                smax_vals = instance_data.get("Smax", {})
                smin = smin_vals.get(str(selected_turbine), None)
                smax = smax_vals.get(str(selected_turbine), None)
 
                health_vals = df_health.loc[selected_turbine].tolist()
 
                # Colorear barras según estado del día
                bar_colors = []
                for idx_d, day in enumerate(days):
                    key = str(day) if str(day) in df_maint.columns else day
                    is_maint = (key in df_maint.columns and df_maint.loc[selected_turbine, key] == "X")
                    is_fail  = (smin is not None and health_vals[idx_d] < smin)
                    if is_fail:
                        bar_colors.append("#c9a99a")   # terracota — falla
                    elif is_maint:
                        bar_colors.append("#8aa8c0")   # azul pizarra — mantenimiento
                    else:
                        bar_colors.append("#c8cfd6")   # gris neutro — normal
 
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=[str(d) for d in days],
                    y=health_vals,
                    marker_color=bar_colors,
                    marker_line_width=0,
                    hovertemplate="Día %{x}<br>Salud: %{y:.2f}<extra></extra>",
                ))
 
                if smin is not None:
                    fig_bar.add_hline(
                        y=smin, line_dash="dot",
                        line_color="#b05a5a", line_width=1.2,
                        annotation_text=f"Smin {smin}",
                        annotation_font_size=10,
                        annotation_font_color="#b05a5a",
                        annotation_position="bottom right",
                    )
                if smax is not None:
                    fig_bar.add_hline(
                        y=smax, line_dash="dot",
                        line_color="#4a6a8a", line_width=1.2,
                        annotation_text=f"Smax {smax}",
                        annotation_font_size=10,
                        annotation_font_color="#4a6a8a",
                        annotation_position="top right",
                    )
 
                fig_bar.update_layout(
                    height=300,
                    margin=dict(l=0, r=0, t=16, b=0),
                    plot_bgcolor="#fafafa",
                    paper_bgcolor="#fafafa",
                    xaxis=dict(
                        title=None,
                        tickfont=dict(size=11, color="#888"),
                        gridcolor="#efefef",
                        linecolor="#e0e0e0",
                    ),
                    yaxis=dict(
                        title=None,
                        tickfont=dict(size=11, color="#888"),
                        gridcolor="#efefef",
                        linecolor="#e0e0e0",
                    ),
                    bargap=0.25,
                    showlegend=False,
                )
 
                st.markdown("""
                <div class="legend-row">
                    <span><span class="legend-dot" style="background:#c8cfd6;"></span>Operación normal</span>
                    <span><span class="legend-dot" style="background:#8aa8c0;"></span>Mantenimiento</span>
                    <span><span class="legend-dot" style="background:#c9a99a;"></span>Falla</span>
                </div>
                """, unsafe_allow_html=True)
                st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
 
                # ── GANTT (HEATMAP MINIMALISTA) ───────────────
                st.markdown('<div class="section-title">Calendario de acciones</div>', unsafe_allow_html=True)
 
                maint_cols = [int(c) for c in df_maint.columns]
                gantt_z    = np.zeros((len(turbines), len(maint_cols)))
 
                for ri, turb in enumerate(turbines):
                    for ci, day in enumerate(maint_cols):
                        key = str(day) if str(day) in df_maint.columns else day
                        val = df_maint.loc[turb, key] if key in df_maint.columns else ""
                        if val == "X":
                            gantt_z[ri, ci] = 1
                        elif val == "F":
                            gantt_z[ri, ci] = 2
 
                emerg_days = []
                if not df_emerg.empty:
                    for day_col in df_emerg.columns:
                        if df_emerg.loc["Flota", day_col] == "E":
                            emerg_days.append(str(day_col))
 
                colorscale = [
                    [0.00, "#efefef"],
                    [0.49, "#efefef"],
                    [0.50, "#8aa8c0"],
                    [0.99, "#8aa8c0"],
                    [1.00, "#c9a99a"],
                ]
 
                fig_gantt = go.Figure(go.Heatmap(
                    z=gantt_z,
                    x=[str(d) for d in maint_cols],
                    y=[f"T{t}" for t in turbines],
                    colorscale=colorscale,
                    zmin=0, zmax=2,
                    showscale=False,
                    xgap=2, ygap=2,
                    hovertemplate="<b>Turbina %{y}</b><br>Día %{x}<br>%{customdata}<extra></extra>",
                    customdata=np.where(
                        gantt_z == 0, "Operando",
                        np.where(gantt_z == 1, "Mantenimiento", "Falla")
                    ),
                ))
 
                for ed in emerg_days:
                    fig_gantt.add_vrect(
                        x0=ed, x1=ed,
                        fillcolor="rgba(176,90,90,0.12)",
                        line_color="rgba(176,90,90,0.35)",
                        line_width=1,
                    )
 
                fig_gantt.update_layout(
                    height=max(160, len(turbines) * 38 + 60),
                    margin=dict(l=0, r=0, t=8, b=0),
                    plot_bgcolor="#fafafa",
                    paper_bgcolor="#fafafa",
                    xaxis=dict(
                        side="top",
                        tickfont=dict(size=10, color="#888"),
                        linecolor="#e0e0e0",
                        title=None,
                    ),
                    yaxis=dict(
                        tickfont=dict(size=10, color="#888"),
                        linecolor="#e0e0e0",
                        title=None,
                        autorange="reversed",
                    ),
                )
 
                st.markdown("""
                <div class="legend-row">
                    <span><span class="legend-dot" style="background:#efefef; border:1px solid #d0d0d0;"></span>Operación normal</span>
                    <span><span class="legend-dot" style="background:#8aa8c0;"></span>Mantenimiento</span>
                    <span><span class="legend-dot" style="background:#c9a99a;"></span>Falla</span>
                    <span><span class="legend-dot" style="background:rgba(176,90,90,0.18); border:1px solid rgba(176,90,90,0.4);"></span>Emergencia activa</span>
                </div>
                """, unsafe_allow_html=True)
                st.plotly_chart(fig_gantt, use_container_width=True, config={"displayModeBar": False})
 
                # ── TABLAS DETALLADAS ─────────────────────────
                st.markdown('<div class="section-title">Tablas detalladas</div>', unsafe_allow_html=True)

                def _style(val):
                    if val == "X":
                        return "background-color:#dde8f0; color:#2a4a6a;"
                    elif val == "E":
                        return "background-color:#f0dedd; color:#6a2a2a;"
                    return "color:#444;"

                st.markdown('<div class="section-title" style="margin-top:8px;">Mantenimiento</div>', unsafe_allow_html=True)
                st.dataframe(res['maintenance'].style.map(_style), use_container_width=True)

                st.markdown('<div class="section-title">Despliegue de emergencia</div>', unsafe_allow_html=True)
                st.dataframe(res['emergency'].style.map(_style), use_container_width=True)

                st.markdown('<div class="section-title">Evolución de salud</div>', unsafe_allow_html=True)
                st.dataframe(
                    res['health'].style
                        .format("{:.2f}")
                        .background_gradient(cmap="Blues", axis=None),
                    use_container_width=True,
                )

                st.markdown('<div class="section-title">Salud desperdiciada</div>', unsafe_allow_html=True)
                st.dataframe(
                    res['loss'].style
                        .format("{:.2f}")
                        .background_gradient(cmap="Oranges", axis=None),
                    use_container_width=True,
                )
with tab6:
    st.markdown("""
    ### ***Resultados Computacionales***
    
    Se ejecutaron todas las instancias en Google Colab (GPU T4) durante 800 segundos. Los resultados se muestran en la tabla, para 8 de las 10 instancias se encontró la solución óptima, mientras que para las 2 instacias restantes el GAP máximo fue de 4.3%.
    Tambien es relevante señalar que en ninguna de las soluciones encontradas por el solver se utilizó la flota de emergencia, es decir, no se superó el número máximo de aerogenedares fallados en ningun día del horizonte de tiempo de cada una de las instancias. 
    """)
 
    results_table = {
        'Instancia': [
            'small_01.json', 'small_02.json',
            'medium_01.json', 'medium_02.json', 'medium_03.json',
            'large_01.json', 'large_02.json', 'large_03.json', 'large_04.json',
            'xlarge_01.json',
        ],
        'GAP': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0001, 0.0001, 0.0314, 0.0430],
        'Mejor Cota': [
            1954.52, 9794.25, 37411.499048, 69428.868824, 52663.231015,
            81498.893049, 216995.925725, 269704.936656, 650463.1762, 572647.311807,
        ],
        'Mejor Solución': [
            1954.52, 9794.25, 37411.499048, 69428.868824, 52663.231015,
            81498.893049, 217016.570839, 269722.8572, 671552.806249, 598388.120402,
        ],
    }
 
    df_results = pd.DataFrame(results_table)
 
    def _style_gap(val):
        if isinstance(val, float):
            if val == 0.0:
                return "color:#2a6a2a; font-weight:600;"
            elif val < 0.01:
                return "color:#5a4a2a;"
            else:
                return "color:#7a2020;"
        return ""
 
    st.dataframe(
        df_results.style
            .format({"GAP": "{:.4f}", "Mejor Cota": "{:,.2f}", "Mejor Solución": "{:,.2f}"})
            .applymap(_style_gap, subset=["GAP"]),
        use_container_width=True,
        hide_index=True,
    )
