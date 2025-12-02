import pandas as pd
import numpy as np
import pickle
import warnings
from typing import Tuple, Optional, Dict, Any
import os
warnings.filterwarnings('ignore')

# Diccionario de información de sensores
SENSOR_INFO = {
    1: {"name": "T2", "description": "Total temperature at fan inlet", "unit": "°R", "critical": True},
    2: {"name": "T24", "description": "Total temperature at LPC outlet", "unit": "°R", "critical": True},
    3: {"name": "T30", "description": "Total temperature at HPC outlet", "unit": "°R", "critical": True},
    4: {"name": "T50", "description": "Total temperature at LPT outlet", "unit": "°R", "critical": True},
    5: {"name": "P2", "description": "Static pressure at fan inlet", "unit": "psia", "critical": False},
    6: {"name": "P15", "description": "Total pressure in bypass duct", "unit": "psia", "critical": False},
    7: {"name": "P30", "description": "Total pressure at HPC outlet", "unit": "psia", "critical": True},
    8: {"name": "Nf", "description": "Physical fan speed", "unit": "rpm", "critical": True},
    9: {"name": "Nc", "description": "Physical core speed", "unit": "rpm", "critical": True},
    10: {"name": "epr", "description": "Engine pressure ratio (P50/P2)", "unit": "—", "critical": True},
    11: {"name": "Ps30", "description": "Static pressure at HPC outlet", "unit": "psia", "critical": True},
    12: {"name": "phi", "description": "Ratio of fuel flow to Ps30", "unit": "pps/psi", "critical": False},
    13: {"name": "NRf", "description": "Corrected fan speed", "unit": "rpm", "critical": False},
    14: {"name": "NRc", "description": "Corrected core speed", "unit": "rpm", "critical": False},
    15: {"name": "BPR", "description": "Bypass ratio", "unit": "—", "critical": False},
    16: {"name": "farB", "description": "Burner fuel-air ratio", "unit": "—", "critical": False},
    17: {"name": "htBleed", "description": "Bleed enthalpy", "unit": "—", "critical": False},
    18: {"name": "Nf_dmd", "description": "Required fan speed", "unit": "rpm", "critical": False},
    19: {"name": "PCNfR_dmd", "description": "Required corrected fan speed", "unit": "rpm", "critical": False},
    20: {"name": "W31", "description": "High-pressure turbine coolant bleed", "unit": "lbm/s", "critical": True},
    21: {"name": "W32", "description": "Low-pressure turbine coolant bleed", "unit": "lbm/s", "critical": True}
}

# Paleta de colores moderna
COLORS = {
    "primary": "#4361ee",
    "secondary": "#3a0ca3",
    "accent": "#7209b7",
    "success": "#4cc9f0",
    "warning": "#f72585",
    "danger": "#b5179e",
    "dark": "#1a1a2e",
    "light": "#f8f9fa",
    "gray": "#6c757d",
    "background": "#ffffff",
    "card": "#f1f3f9",
}

class EngineAnalyzer:
    def __init__(self):
        self.model = None
        self.sensor_info = SENSOR_INFO
        self.colors = COLORS
        self.model_loaded = False
        self.expected_features = None
        
    def load_model(self, model_path="../models/trained_model.pkl"):
        """Carga el modelo entrenado desde la ruta especificada"""
        try:
            # Intentar diferentes rutas relativas
            possible_paths = [
                model_path,
                "models/trained_model.pkl",
                "../models/trained_model.pkl",
                os.path.join(os.path.dirname(__file__), "../models/trained_model.pkl")
            ]
            
            loaded = False
            for path in possible_paths:
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        self.model = pickle.load(f)
                    self.model_loaded = True
                    print(f"Model loaded successfully from {path}")
                    
                    # Intentar obtener las características esperadas del modelo
                    self._extract_expected_features()
                    loaded = True
                    break
            
            if not loaded:
                print("Model file not found. Using heuristic estimation.")
                self.model = None
                self.model_loaded = False
            
            return self.model_loaded
            
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            self.model = None
            self.model_loaded = False
            return False
    
    def _extract_expected_features(self):
        """Extrae las características esperadas del modelo"""
        try:
            # Para modelos scikit-learn
            if hasattr(self.model, 'feature_names_in_'):
                self.expected_features = list(self.model.feature_names_in_)
            elif hasattr(self.model, 'get_booster'):
                # Para XGBoost
                self.expected_features = self.model.get_booster().feature_names
            elif hasattr(self.model, 'feature_name_'):
                # Para LightGBM
                self.expected_features = self.model.feature_name_
            else:
                # Usar características por defecto basadas en FD001_enhanced
                self.expected_features = [
                    'sensor_measure_2', 'sensor_measure_3', 'sensor_measure_4', 
                    'sensor_measure_6', 'sensor_measure_7', 'sensor_measure_8', 
                    'sensor_measure_9', 'sensor_measure_11', 'sensor_measure_12', 
                    'sensor_measure_13', 'sensor_measure_14', 'sensor_measure_15', 
                    'sensor_measure_17', 'sensor_measure_20', 'sensor_measure_21',
                    'T30_norm', 'd_T30', 'Ps30_norm', 'd_Ps30', 'T50_norm', 'd_T50',
                    'thermal_stress', 'pressure_ratio'
                ]
            print(f"Expected features: {len(self.expected_features)} features")
            
        except Exception as e:
            print(f"Could not extract expected features: {e}")
            self.expected_features = None
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica ingeniería de características al dataframe siguiendo el formato del modelo"""
        X = df.copy()
        
        print(f"Original columns: {list(X.columns)}")
        
        # Normalize features by unit - EN EL ORDEN ESPECÍFICO
        for unit in X["unit"].unique():
            unit_mask = X["unit"] == unit
            
            # T30 features - primero la normalización, luego la derivada
            if "sensor_measure_3" in X.columns:
                initial_T30 = X.loc[unit_mask, "sensor_measure_3"].iloc[0]
                if initial_T30 != 0:
                    X.loc[unit_mask, "T30_norm"] = X.loc[unit_mask, "sensor_measure_3"] / initial_T30
                    X.loc[unit_mask, "d_T30"] = X.loc[unit_mask, "T30_norm"].diff().fillna(0)
            
            # Ps30 features - orden específico: normalización luego derivada
            if "sensor_measure_11" in X.columns:
                initial_Ps30 = X.loc[unit_mask, "sensor_measure_11"].iloc[0]
                if initial_Ps30 != 0:
                    X.loc[unit_mask, "Ps30_norm"] = X.loc[unit_mask, "sensor_measure_11"] / initial_Ps30
                    X.loc[unit_mask, "d_Ps30"] = X.loc[unit_mask, "Ps30_norm"].diff().fillna(0)
            
            # T50 features - orden específico: normalización luego derivada
            if "sensor_measure_4" in X.columns:
                initial_T50 = X.loc[unit_mask, "sensor_measure_4"].iloc[0]
                if initial_T50 != 0:
                    X.loc[unit_mask, "T50_norm"] = X.loc[unit_mask, "sensor_measure_4"] / initial_T50
                    X.loc[unit_mask, "d_T50"] = X.loc[unit_mask, "T50_norm"].diff().fillna(0)
        
        # Physical health indicators
        if "sensor_measure_4" in X.columns and "sensor_measure_3" in X.columns:
            X["thermal_stress"] = X["sensor_measure_4"] - X["sensor_measure_3"]
        
        if "sensor_measure_11" in X.columns and "sensor_measure_7" in X.columns:
            # Evitar división por cero
            valid_mask = X["sensor_measure_7"] != 0
            X.loc[valid_mask, "pressure_ratio"] = X.loc[valid_mask, "sensor_measure_11"] / X.loc[valid_mask, "sensor_measure_7"]
            X["pressure_ratio"] = X["pressure_ratio"].fillna(0)
        
        # Mantener solo las columnas necesarias basadas en el formato FD001_enhanced
        # Este es el orden CORRECTO basado en el error que vimos
        enhanced_format_columns = [
            'unit',
            'sensor_measure_2', 'sensor_measure_3', 'sensor_measure_4',
            'sensor_measure_6', 'sensor_measure_7', 'sensor_measure_8',
            'sensor_measure_9', 'sensor_measure_11', 'sensor_measure_12',
            'sensor_measure_13', 'sensor_measure_14', 'sensor_measure_15',
            'sensor_measure_17', 'sensor_measure_20', 'sensor_measure_21',
            'T30_norm', 'd_T30', 'Ps30_norm', 'd_Ps30', 'T50_norm', 'd_T50',
            'thermal_stress', 'pressure_ratio'
        ]
        
        # Filtrar solo las columnas que existen en X
        existing_columns = [col for col in enhanced_format_columns if col in X.columns]
        
        # Asegurarse de que tenemos todas las columnas necesarias
        missing_columns = [col for col in enhanced_format_columns if col not in X.columns]
        if missing_columns:
            print(f"Warning: Missing columns in processed data: {missing_columns}")
            # Crear columnas faltantes con valores por defecto
            for col in missing_columns:
                if col not in ['unit', 'RUL']:  # No crear unit o RUL
                    X[col] = 0
        
        # Ordenar las columnas en el orden correcto
        final_columns = [col for col in enhanced_format_columns if col in X.columns]
        X = X[final_columns]
        
        print(f"Processed columns ({len(X.columns)}): {list(X.columns)}")
        return X
    
    def _align_features_with_model(self, features: pd.DataFrame) -> pd.DataFrame:
        """Alinea las características con lo que espera el modelo"""
        if self.expected_features is None:
            return features
        
        # Crear un nuevo DataFrame con las características en el orden correcto
        aligned_features = pd.DataFrame()
        
        for feature in self.expected_features:
            if feature in features.columns:
                aligned_features[feature] = features[feature]
            else:
                # Si falta una característica, usar 0
                print(f"Warning: Missing expected feature: {feature}")
                aligned_features[feature] = 0
        
        # Asegurarse de que tenemos todas las columnas en el orden correcto
        aligned_features = aligned_features[self.expected_features]
        
        return aligned_features
    
    def predict_rul(self, df_processed: pd.DataFrame) -> float:
        """Predice RUL usando el modelo cargado"""
        if not self.model_loaded or self.model is None:
            # Si no hay modelo, usar estimación heurística
            return self._estimate_rul_heuristic(df_processed)[0]
        
        try:
            # Preparar datos para el modelo
            features_for_model = self._prepare_for_prediction(df_processed)
            
            if len(features_for_model) == 0:
                print("No features for prediction, using heuristic")
                return self._estimate_rul_heuristic(df_processed)[0]
            
            # Verificar que tenemos las columnas correctas
            print(f"Features for prediction: {list(features_for_model.columns)}")
            print(f"Features shape: {features_for_model.shape}")
            
            # Realizar predicción
            if hasattr(self.model, 'predict'):
                predictions = self.model.predict(features_for_model)
            else:
                # Para otros tipos de modelos
                predictions = np.array([100])  # Valor por defecto
            
            # Asegurar que sea un array 1D
            if hasattr(predictions, 'flatten'):
                predictions = predictions.flatten()
            
            # Promediar predicciones para el motor
            rul = float(np.mean(predictions))
            print(f"Predicted RUL: {rul}")
            
            return rul
            
        except Exception as e:
            print(f"Error in model prediction: {e}")
            import traceback
            traceback.print_exc()
            return self._estimate_rul_heuristic(df_processed)[0]
    
    def _prepare_for_prediction(self, df_processed: pd.DataFrame) -> pd.DataFrame:
        """Prepara características para predicción"""
        try:
            # Eliminar columnas que no son características
            features = df_processed.copy()
            
            # Eliminar columnas no numéricas o no características
            columns_to_drop = ['unit', 'RUL', 'cycle', 'max_cycle']
            for col in columns_to_drop:
                if col in features.columns:
                    features = features.drop(columns=[col])
            
            # Manejar valores NaN
            features = features.fillna(0)
            
            # Asegurar que todas las columnas sean numéricas
            for col in features.columns:
                if not pd.api.types.is_numeric_dtype(features[col]):
                    features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0)
            
            # Si tenemos características esperadas, alinear
            if self.expected_features is not None:
                features = self._align_features_with_model(features)
            
            return features
            
        except Exception as e:
            print(f"Error preparing features: {e}")
            return pd.DataFrame()
    
    def _estimate_rul_heuristic(self, df_processed: pd.DataFrame) -> Tuple[float, str, str, float, float]:
        """Estimación heurística de RUL si no hay modelo disponible"""
        if len(df_processed) == 0:
            return 100.0, "LOW", self.colors["success"], 0.0, 0.0
        
        try:
            # Para motores nuevos (primer ciclo), asumir bajo riesgo
            if 'cycle' in df_processed.columns:
                min_cycle = df_processed['cycle'].min() if len(df_processed) > 0 else 1
                if min_cycle <= 5:  # Primeros 5 ciclos
                    return 180.0, "LOW", self.colors["success"], 0.0, 0.0
            
            # Calcular métricas clave con manejo seguro
            thermal_stress_mean = 0.0
            if "thermal_stress" in df_processed.columns:
                thermal_stress_mean = float(df_processed["thermal_stress"].mean())
            
            pressure_ratio_mean = 1.0
            if "pressure_ratio" in df_processed.columns:
                pressure_ratio_mean = float(df_processed["pressure_ratio"].mean())
            
            # Risk assessment logic más realista
            risk_score = 0
            
            # Umbrales más realistas para motores
            if thermal_stress_mean > 100:  # Alto estrés térmico
                risk_score += 2
            elif thermal_stress_mean > 50:  # Estrés térmico moderado
                risk_score += 1
            
            if pressure_ratio_mean > 3.0:  # Ratio de presión muy alto
                risk_score += 2
            elif pressure_ratio_mean > 2.0:  # Ratio de presión alto
                risk_score += 1
            
            # Determinar nivel de riesgo basado en score
            if risk_score == 0:
                rul_estimate = float(np.random.randint(180, 250))  # Vida larga
                risk_level = "LOW"
                risk_color = self.colors["success"]
            elif risk_score <= 2:
                rul_estimate = float(np.random.randint(100, 180))  # Vida media
                risk_level = "MODERATE"
                risk_color = self.colors["warning"]
            else:
                rul_estimate = float(np.random.randint(30, 100))   # Vida corta
                risk_level = "HIGH"
                risk_color = self.colors["danger"]
            
            print(f"Heuristic: RUL={rul_estimate}, Risk={risk_level}, Thermal={thermal_stress_mean}, Pressure={pressure_ratio_mean}")
            
            return rul_estimate, risk_level, risk_color, thermal_stress_mean, pressure_ratio_mean
            
        except Exception as e:
            print(f"Heuristic estimation error: {e}")
            # Por defecto, asumir bajo riesgo
            return 150.0, "LOW", self.colors["success"], 0.0, 0.0
    
    def assess_engine_health(self, df_processed: pd.DataFrame, unit_id: Optional[int] = None) -> Tuple[float, str, str, float, float]:
        """Evalúa la salud del motor"""
        try:
            if unit_id is not None:
                df_unit = df_processed[df_processed["unit"] == unit_id].copy()
            else:
                df_unit = df_processed.copy()
            
            if len(df_unit) == 0:
                return 150.0, "LOW", self.colors["success"], 0.0, 0.0
            
            print(f"Assessing engine {unit_id}, data points: {len(df_unit)}")
            
            # Usar modelo si está disponible y funciona
            if self.model_loaded and self.model is not None:
                try:
                    rul_estimate = self.predict_rul(df_unit)
                    
                    # Estimación de riesgo basada en RUL con umbrales realistas
                    if rul_estimate > 180:
                        risk_level = "LOW"
                        risk_color = self.colors["success"]
                    elif rul_estimate > 100:
                        risk_level = "MODERATE"
                        risk_color = self.colors["warning"]
                    else:
                        risk_level = "HIGH"
                        risk_color = self.colors["danger"]
                    
                    thermal_stress = 0.0
                    if "thermal_stress" in df_unit.columns:
                        thermal_stress = float(df_unit["thermal_stress"].mean())
                    
                    pressure_ratio = 1.0
                    if "pressure_ratio" in df_unit.columns:
                        pressure_ratio = float(df_unit["pressure_ratio"].mean())
                    
                    print(f"Model assessment: RUL={rul_estimate}, Risk={risk_level}")
                    return rul_estimate, risk_level, risk_color, thermal_stress, pressure_ratio
                    
                except Exception as model_error:
                    print(f"Model prediction failed, using heuristic: {model_error}")
                    # Si falla el modelo, usar heurística
                    return self._estimate_rul_heuristic(df_unit)
            else:
                # Usar heurística si no hay modelo
                return self._estimate_rul_heuristic(df_unit)
                
        except Exception as e:
            print(f"Engine health assessment error: {e}")
            # Por defecto, asumir bajo riesgo
            return 150.0, "LOW", self.colors["success"], 0.0, 0.0
    
    def get_sensor_columns(self, df: pd.DataFrame) -> list:
        """Obtiene las columnas de sensores del dataframe"""
        return [col for col in df.columns if "sensor_measure" in col and col in df.columns]
    
    def get_sensor_data(self, df: pd.DataFrame, sensor_id: int, unit_id: Optional[int] = None) -> Optional[pd.Series]:
        """Obtiene datos de un sensor específico"""
        sensor_col = f"sensor_measure_{sensor_id}"
        if sensor_col not in df.columns:
            return None
        
        if unit_id is not None:
            df_unit = df[df["unit"] == unit_id]
        else:
            df_unit = df
        
        return df_unit[sensor_col]
    
    def get_correlated_sensors(self, df: pd.DataFrame, sensor_id: int, unit_id: Optional[int] = None, top_n: int = 3) -> list:
        """Encuentra sensores correlacionados con el sensor dado"""
        try:
            sensor_col = f"sensor_measure_{sensor_id}"
            
            if unit_id is not None:
                df_unit = df[df["unit"] == unit_id]
            else:
                df_unit = df
            
            sensor_cols = self.get_sensor_columns(df_unit)
            
            if sensor_col not in sensor_cols or len(sensor_cols) < 2:
                return []
            
            corr_matrix = df_unit[sensor_cols].corr()
            sensor_correlations = corr_matrix[sensor_col].abs()
            sensor_correlations = sensor_correlations.drop(sensor_col).sort_values(ascending=False)
            
            return sensor_correlations.head(top_n).index.tolist()
            
        except Exception as e:
            print(f"Correlation analysis error: {e}")
            return []

# Instancia global del analizador
engine_analyzer = EngineAnalyzer()