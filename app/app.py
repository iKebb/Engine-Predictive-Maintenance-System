import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from engine_logic import engine_analyzer, COLORS

# Configuración de la página
st.set_page_config(
    page_title="Engine Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Cargar modelo al inicio con ruta correcta
try:
    # Intentar diferentes rutas
    engine_analyzer.load_model("../models/trained_model.pkl")
except Exception as e:
    st.warning(f"Model could not be loaded: {e}. Using heuristic estimation.")

# Estilos CSS
st.markdown(f"""
<style>
    .main-header {{
        font-size: 2.5rem;
        font-weight: 700;
        color: {COLORS['dark']};
        text-align: center;
        margin-bottom: 1rem;
        padding-top: 1rem;
        font-family: 'Inter', sans-serif;
    }}
    .sub-header {{
        font-size: 1.2rem;
        color: {COLORS['gray']};
        text-align: center;
        margin-bottom: 2rem;
        font-family: 'Inter', sans-serif;
    }}
    .upload-section {{
        background-color: {COLORS['card']};
        border-radius: 12px;
        padding: 2rem;
        border: 2px dashed {COLORS['primary']};
        margin-bottom: 2rem;
    }}
    .status-card {{
        background: linear-gradient(135deg, {COLORS['primary']}15, {COLORS['secondary']}15);
        border-radius: 12px;
        padding: 1.5rem;
        color: {COLORS['dark']};
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border-left: 4px solid {COLORS['primary']};
    }}
    .metric-card {{
        background-color: {COLORS['background']};
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        border: 1px solid {COLORS['light']};
    }}
    .alert-high {{
        background: linear-gradient(135deg, {COLORS['danger']}15, {COLORS['warning']}15);
        color: {COLORS['danger']};
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid {COLORS['danger']};
    }}
    .alert-moderate {{
        background: linear-gradient(135deg, {COLORS['warning']}15, #ff9e0015);
        color: {COLORS['warning']};
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid {COLORS['warning']};
    }}
    .alert-low {{
        background: linear-gradient(135deg, {COLORS['success']}15, {COLORS['primary']}15);
        color: {COLORS['success']};
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid {COLORS['success']};
    }}
    .sensor-preview {{
        background: {COLORS['card']};
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem;
        text-align: center;
        border: 1px solid {COLORS['light']};
        transition: all 0.3s ease;
    }}
    .sensor-preview:hover {{
        background: {COLORS['light']};
        border-color: {COLORS['primary']};
    }}
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['secondary']});
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.8rem;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }}
    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 12px {COLORS['primary']}60;
    }}
</style>
""", unsafe_allow_html=True)

# ===== FUNCIONES PRINCIPALES =====

def validate_dataframe(df: pd.DataFrame) -> bool:
    """Valida que el dataframe tenga el formato correcto"""
    if df is None or df.empty:
        return False
    
    required_columns = ['unit', 'cycle']
    if not all(col in df.columns for col in required_columns):
        return False
    
    # Verificar que haya sensores
    sensor_cols = [col for col in df.columns if 'sensor_measure' in col]
    return len(sensor_cols) >= 5

def show_sensor_theory():
    """Muestra información teórica de los sensores"""
    st.markdown("### Sensor Theory and Reference Information")
    
    with st.expander("View Sensor Details"):
        tab1, tab2 = st.tabs(["Critical Sensors", "All Sensors"])
        
        with tab1:
            critical_sensors = [info for info in engine_analyzer.sensor_info.values() if info["critical"]]
            for sensor in critical_sensors:
                st.markdown(f"""
                **{sensor['name']}**  
                *{sensor['description']}*  
                Unit: {sensor['unit']}  
                """)
        
        with tab2:
            sensor_data = []
            for i in range(1, 22):
                info = engine_analyzer.sensor_info[i]
                sensor_data.append({
                    "ID": i,
                    "Name": info["name"],
                    "Description": info["description"],
                    "Unit": info["unit"],
                    "Critical": "Yes" if info["critical"] else "No"
                })
            
            sensor_df = pd.DataFrame(sensor_data)
            st.dataframe(sensor_df, use_container_width=True, hide_index=True)

def show_failure_example():
    """Muestra ejemplo de motor fallado"""
    st.markdown("### Engine Failure Progression Example")
    
    # Datos de ejemplo
    cycles = list(range(1, 101))
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Temperature T30", "Temperature T50", "Pressure Ps30", "Vibration"),
        vertical_spacing=0.15
    )
    
    # Simular datos
    t30_data = [518 + i*0.1 for i in cycles[:70]] + [525 + (i-70)*3.96 for i in cycles[70:]]
    t50_data = [1430 + i*0.14 for i in cycles[:70]] + [1440 + (i-70)*5.65 for i in cycles[70:]]
    ps30_data = [21.61 - i*0.016 for i in cycles[:70]] + [20.5 - (i-70)*0.196 for i in cycles[70:]]
    vibration_data = np.concatenate([
        np.random.normal(0.1, 0.02, 70),
        np.random.normal(0.3, 0.1, 30)
    ])
    
    # Añadir trazas
    fig.add_trace(go.Scatter(x=cycles, y=t30_data, mode='lines', name='T30', 
                           line=dict(color=COLORS['warning'])), row=1, col=1)
    fig.add_trace(go.Scatter(x=cycles, y=t50_data, mode='lines', name='T50', 
                           line=dict(color=COLORS['danger'])), row=1, col=2)
    fig.add_trace(go.Scatter(x=cycles, y=ps30_data, mode='lines', name='Ps30', 
                           line=dict(color=COLORS['primary'])), row=2, col=1)
    fig.add_trace(go.Scatter(x=cycles, y=vibration_data, mode='lines', name='Vibration', 
                           line=dict(color=COLORS['accent'])), row=2, col=2)
    
    # Añadir línea de degradación
    for row in [1, 2]:
        for col in [1, 2]:
            fig.add_vline(x=70, line_dash="dash", line_color=COLORS['gray'], 
                        annotation_text="Degradation Start", row=row, col=col)
    
    fig.update_layout(height=500, showlegend=False, 
                     title_text="Engine Failure Pattern - Unit 68 (Failed at Cycle 199)")
    st.plotly_chart(fig, use_container_width=True)

def create_sensor_analysis(df_original, df_processed, unit_id):
    """Crea análisis de sensores"""
    if unit_id not in df_original["unit"].values:
        st.warning(f"Engine unit {unit_id} not found in data")
        return
    
    df_unit = df_original[df_original["unit"] == unit_id]
    sensor_cols = engine_analyzer.get_sensor_columns(df_original)
    
    st.markdown("### Sensor Analysis")
    
    # Vista previa de sensores
    cols_per_row = 6
    sensor_cols_filtered = [col for col in sensor_cols if col in df_unit.columns]
    
    # Crear mini gráficos para cada sensor
    for i in range(0, len(sensor_cols_filtered), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, sensor_col in enumerate(sensor_cols_filtered[i:i+cols_per_row]):
            sensor_id = int(sensor_col.split("_")[-1])
            sensor_info = engine_analyzer.sensor_info.get(sensor_id, {})
            
            with cols[j]:
                if sensor_col in df_unit.columns:
                    sensor_data = df_unit[sensor_col]
                    
                    # Mini gráfico
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=sensor_data.values[:20] if len(sensor_data) > 20 else sensor_data.values,
                        mode='lines',
                        line=dict(color=COLORS['primary'], width=1)
                    ))
                    fig.update_layout(
                        height=60,
                        margin=dict(l=5, r=5, t=5, b=5),
                        showlegend=False,
                        xaxis_visible=False,
                        yaxis_visible=False,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                    
                    # Información del sensor
                    mean_val = sensor_data.mean() if len(sensor_data) > 0 else 0
                    st.markdown(f"""
                    <div style="text-align: center;">
                        <div style="font-size: 0.8rem; color: {COLORS['gray']};">S{sensor_id}</div>
                        <div style="font-size: 0.9rem; font-weight: 600; color: {COLORS['dark']};">{sensor_info.get('name', f'S{sensor_id}')}</div>
                        <div style="font-size: 0.7rem; color: {COLORS['primary']};">{mean_val:.1f}</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Análisis detallado de un sensor seleccionado
    st.markdown("---")
    st.markdown("#### Detailed Sensor Analysis")
    
    selected_sensor = st.selectbox(
        "Select a sensor for detailed analysis",
        options=[f"Sensor {i}" for i in range(1, 22)],
        format_func=lambda x: f"{x} - {engine_analyzer.sensor_info.get(int(x.split()[1]), {}).get('name', 'Unknown')}"
    )
    
    if selected_sensor:
        sensor_id = int(selected_sensor.split()[1])
        sensor_data = engine_analyzer.get_sensor_data(df_original, sensor_id, unit_id)
        
        if sensor_data is not None and len(sensor_data) > 0:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_unit.index,
                    y=sensor_data.values,
                    mode="lines",
                    name=f"Sensor {sensor_id}",
                    line=dict(color=COLORS["primary"], width=2)
                ))
                
                fig.update_layout(
                    title=f"Sensor {sensor_id} - Time Series",
                    xaxis_title="Observation",
                    yaxis_title="Value",
                    height=300,
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Estadísticas
                stats = {
                    "Current": f"{sensor_data.iloc[-1]:.2f}",
                    "Mean": f"{sensor_data.mean():.2f}",
                    "Std Dev": f"{sensor_data.std():.2f}",
                    "Min": f"{sensor_data.min():.2f}",
                    "Max": f"{sensor_data.max():.2f}"
                }
                
                for key, value in stats.items():
                    st.metric(label=key, value=value)

def create_engine_health_indicators(df_processed, unit_id):
    """Crea indicadores de salud del motor"""
    st.markdown("### Engine Health Indicators")
    
    if unit_id not in df_processed["unit"].values:
        st.warning(f"No processed data for engine unit {unit_id}")
        return
    
    df_unit = df_processed[df_processed["unit"] == unit_id]
    
    # Crear gráficos de salud
    col1, col2 = st.columns(2)
    
    with col1:
        if "thermal_stress" in df_unit.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_unit.index,
                y=df_unit["thermal_stress"],
                mode="lines+markers",
                name="Thermal Stress",
                line=dict(color=COLORS["warning"], width=2)
            ))
            
            fig.update_layout(
                title="Thermal Stress",
                xaxis_title="Observation",
                yaxis_title="Δ Temperature (°R)",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if "pressure_ratio" in df_unit.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_unit.index,
                y=df_unit["pressure_ratio"],
                mode="lines+markers",
                name="Pressure Ratio",
                line=dict(color=COLORS["primary"], width=2)
            ))
            
            fig.update_layout(
                title="Pressure Ratio",
                xaxis_title="Observation",
                yaxis_title="Ratio",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Gráficos adicionales si existen
    health_plots = []
    for col in ["T30_norm", "Ps30_norm", "T50_norm"]:
        if col in df_unit.columns:
            health_plots.append(col)
    
    if health_plots:
        st.markdown("#### Normalized Indicators")
        fig = go.Figure()
        
        colors = [COLORS["primary"], COLORS["success"], COLORS["accent"]]
        for i, col in enumerate(health_plots[:3]):
            fig.add_trace(go.Scatter(
                x=df_unit.index,
                y=df_unit[col],
                mode="lines",
                name=col,
                line=dict(color=colors[i], width=1.5)
            ))
        
        fig.update_layout(
            title="Normalized Health Indicators",
            xaxis_title="Observation",
            yaxis_title="Normalized Value",
            height=300,
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

def single_engine_dashboard(df_original, df_processed, unit_id):
    """Dashboard para análisis de motor individual"""
    
    # Evaluar salud del motor
    try:
        rul_estimate, risk_level, risk_color, thermal_stress, pressure_ratio = engine_analyzer.assess_engine_health(df_processed, unit_id)
    except Exception as e:
        st.error(f"Error assessing engine health: {e}")
        rul_estimate, risk_level, risk_color, thermal_stress, pressure_ratio = 150, "LOW", COLORS["success"], 0, 0
    
    # Header
    st.markdown(f'<div class="main-header">Engine Analysis - Unit {unit_id}</div>', unsafe_allow_html=True)
    
    # ===== SECCIÓN 1: RISK LEVEL & ESTIMATED RUL =====
    st.markdown("### Risk Assessment & RUL Estimation")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="status-card">
            <h3 style="margin:0; font-size:1rem; color:#666;">RISK LEVEL</h3>
            <h1 style="margin:0; font-size:2rem; color:{risk_color};">{risk_level}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="status-card">
            <h3 style="margin:0; font-size:1rem; color:#666;">ESTIMATED RUL</h3>
            <h1 style="margin:0; font-size:2rem; color:{COLORS['dark']};">{int(rul_estimate)}</h1>
            <p style="margin:0; color:#999; font-size:0.8rem;">cycles remaining</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="status-card">
            <h3 style="margin:0; font-size:1rem; color:#666;">THERMAL STRESS</h3>
            <h1 style="margin:0; font-size:2rem; color:{COLORS['dark']};">{thermal_stress:.1f}</h1>
            <p style="margin:0; color:#999; font-size:0.8rem;">°R difference</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="status-card">
            <h3 style="margin:0; font-size:1rem; color:#666;">PRESSURE RATIO</h3>
            <h1 style="margin:0; font-size:2rem; color:{COLORS['dark']};">{pressure_ratio:.2f}</h1>
            <p style="margin:0; color:#999; font-size:0.8rem;">performance index</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Alerta de riesgo
    alert_class = "alert-low" if risk_level == "LOW" else "alert-moderate" if risk_level == "MODERATE" else "alert-high"
    alert_messages = {
        "LOW": "✅ Engine operating within normal parameters. Regular maintenance schedule sufficient.",
        "MODERATE": "⚠️ Engine requires increased monitoring. Schedule maintenance within next 50-100 cycles.",
        "HIGH": "🚨 Immediate attention required! Critical degradation detected. Maintenance needed within 30 cycles."
    }
    
    st.markdown(f"""
    <div class="{alert_class}">
        <h4 style="margin:0;">Status Assessment</h4>
        <p style="margin:0.5rem 0 0 0;">{alert_messages.get(risk_level, 'Status unknown')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ===== SECCIÓN 2: SENSOR ANALYSIS =====
    create_sensor_analysis(df_original, df_processed, unit_id)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ===== SECCIÓN 3: ENGINE HEALTH INDICATORS =====
    create_engine_health_indicators(df_processed, unit_id)
    
    # Advanced Analysis
    st.markdown("### Advanced Analysis")
    with st.expander("View Advanced Analysis Tools"):
        st.markdown("""
        **Available Analytical Tools:**
        
        1. **Trend Analysis** - Identify long-term degradation patterns
        2. **Anomaly Detection** - Spot abnormal sensor behavior
        3. **Correlation Analysis** - Understand sensor relationships
        4. **Predictive Forecasting** - Estimate future engine state
        
        *Note: Some features require historical data spanning multiple operational cycles.*
        """)
        
        # Opción para análisis de correlación
        if st.button("Run Correlation Analysis"):
            st.info("Correlation analysis would show relationships between different sensors. This requires multiple data points for accurate results.")

def multi_engine_dashboard(df_original, df_processed):
    """Dashboard para análisis de múltiples motores"""
    
    unique_units = df_original["unit"].unique()
    num_engines = len(unique_units)
    
    st.markdown(f'<div class="main-header">Fleet Analysis - {num_engines} Engines</div>', unsafe_allow_html=True)
    
    # Engine selection
    st.markdown("### Engine Selection")
    st.markdown(f"**{num_engines} engines detected in the dataset.**")
    
    max_engines_to_show = min(10, num_engines)
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_to_display = st.number_input(
            "Number of engines to display",
            min_value=1,
            max_value=num_engines,
            value=max_engines_to_show
        )
    
    with col2:
        specific_engines = st.multiselect(
            "Select specific engines",
            options=unique_units,
            default=unique_units[:min(3, num_engines)]
        )
    
    # Determine which engines to display
    if specific_engines:
        engines_to_display = specific_engines[:num_to_display]
    else:
        engines_to_display = unique_units[:num_to_display]
    
    # Fleet overview
    st.markdown("### Fleet Overview")
    
    fleet_data = []
    for unit in engines_to_display:
        try:
            rul, risk_level, _, thermal_stress, pressure_ratio = engine_analyzer.assess_engine_health(df_processed, unit)
            fleet_data.append({
                "Engine": f"Unit {unit}",
                "RUL": int(rul),
                "Risk": risk_level,
                "Thermal Stress": f"{thermal_stress:.1f}",
                "Pressure Ratio": f"{pressure_ratio:.2f}"
            })
        except Exception as e:
            print(f"Error processing engine {unit}: {e}")
    
    if fleet_data:
        fleet_df = pd.DataFrame(fleet_data)
        
        # Métricas de la flota
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_rul = np.mean([d["RUL"] for d in fleet_data])
            st.metric("Average RUL", f"{int(avg_rul)}")
        
        with col2:
            high_risk = sum(1 for d in fleet_data if d["Risk"] == "HIGH")
            st.metric("High Risk", high_risk)
        
        with col3:
            moderate_risk = sum(1 for d in fleet_data if d["Risk"] == "MODERATE")
            st.metric("Moderate Risk", moderate_risk)
        
        with col4:
            low_risk = sum(1 for d in fleet_data if d["Risk"] == "LOW")
            st.metric("Low Risk", low_risk)
        
        # Tabla de comparación
        st.markdown("### Engine Comparison")
        st.dataframe(fleet_df, use_container_width=True)
        
        # Gráfico de distribución de RUL
        st.markdown("### RUL Distribution")
        fig = px.histogram(
            x=[d["RUL"] for d in fleet_data],
            nbins=10,
            title="Remaining Useful Life Distribution",
            labels={'x': 'RUL (cycles)', 'y': 'Number of Engines'},
            color_discrete_sequence=[COLORS["primary"]]
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Análisis individual
        st.markdown("### Individual Engine Analysis")
        selected_engine = st.selectbox(
            "Select an engine for detailed analysis",
            options=engines_to_display,
            format_func=lambda x: f"Unit {x}"
        )
        
        if selected_engine:
            single_engine_dashboard(df_original, df_processed, selected_engine)
    else:
        st.warning("No engine data available for analysis")

def main():
    """Función principal de la aplicación"""
    # Header
    st.markdown('<div class="main-header">Engine Predictive Maintenance System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload sensor data for predictive analysis and risk assessment</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Data upload section
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload CSV file with engine sensor data",
            type=["csv", "txt"],
            help="Upload a CSV file with FD001 dataset format"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validar datos
            if not validate_dataframe(df):
                st.error("Invalid file format. Please upload a CSV with 'unit', 'cycle', and sensor columns.")
                return
            
            # Mostrar vista previa
            with st.expander("Preview Uploaded Data"):
                st.dataframe(df.head(10))
                st.write(f"Data shape: {df.shape}")
                st.write(f"Engines: {df['unit'].nunique()}")
                st.write(f"Cycles per engine: {df.groupby('unit')['cycle'].count().to_dict()}")
            
            # Procesar datos
            with st.spinner('Processing data and applying feature engineering...'):
                df_processed = engine_analyzer.feature_engineering(df.copy())
                st.session_state.data_loaded = True
            
            st.success("✅ Data processed successfully!")
            
            # Determinar número de motores
            unique_units = df["unit"].unique()
            num_engines = len(unique_units)
            
            # Mostrar información del modelo
            if engine_analyzer.model_loaded:
                st.info(f"✅ Predictive model loaded. Using AI-powered RUL estimation.")
            else:
                st.warning("⚠️ Using heuristic RUL estimation (no model loaded).")
            
            if num_engines == 1:
                single_engine_dashboard(df, df_processed, unique_units[0])
            else:
                multi_engine_dashboard(df, df_processed)
            
            # Mostrar secciones adicionales
            st.markdown("---")
            show_sensor_theory()
            
            st.markdown("---")
            show_failure_example()
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("""
            **Common Issues:**
            1. Ensure your CSV has columns: 'unit', 'cycle', and sensor_measure_1 to sensor_measure_21
            2. Check for missing values or incorrect data types
            3. Verify the file is not corrupted
            """)
    
    else:
        # Display instructions
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background-color: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
            <h3 style="color: #4a5568;">📁 Data Upload Instructions</h3>
            <p style="color: #718096; max-width: 600px; margin: 1rem auto; line-height: 1.6;">
                1. Prepare your CSV file with FD001 dataset format<br>
                2. Required columns: <code>unit</code>, <code>cycle</code>, sensor_measure_1 to sensor_measure_21<br>
                3. Optional columns: <code>op_setting_1</code>, <code>op_setting_2</code>, <code>op_setting_3</code><br>
                4. Click "Browse files" to upload your dataset
            </p>
            <div style="margin-top: 2rem; color: #a0aec0; font-size: 0.9rem;">
                <strong>Expected format:</strong> Each row represents one cycle of one engine unit
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"""
        <div style="text-align: center; color: #718096; font-size: 0.9rem; padding: 1rem;">
            Engine Predictive Maintenance System v2.2 • 
            <span style="color: {COLORS['primary']};">AI-Powered RUL Prediction</span>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()