from functionalities.requests import fetch_vbox_data_as_dataframe, validate_reference_code, create_table_if_dont_exist, insert_df_to_table, fetch_all_vbox_reference_records
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import KernelDensity


import h3

def request_processor(reference_code: str):

    vbox_df = fetch_vbox_data_as_dataframe(reference_code)
    vbox_df.to_csv("fetched_data.csv", index=False)
    vbox_df_iso_forest = isolation_forest_preprocess(vbox_df)
    vbox_df_with_slope =  add_slope_percent_per_100m(vbox_df_iso_forest)
    vbox_df_with_slope_and_status = add_empty_loaded_labels(vbox_df_with_slope)
    vbox_df_with_slope_status_h3 = add_h3_cell_and_centroid(vbox_df_with_slope_and_status)

    vbox_df_with_slope_status_h3.to_csv("slopes_status_h3.csv", index=False)

    vbox_final_simplified_df = simplify_by_h3_cell(vbox_df_with_slope_status_h3)
    vbox_final_simplified_df.to_csv("simplified_vbox_data.csv", index=False)

    # Procesar DataFrame original
    result = process_dataframe_with_clustering(vbox_final_simplified_df)
    # 
    # # Obtener DataFrame con nuevas columnas
    df_final = result['dataframe']
    df_final.to_csv("vbox_data_processed.csv", index=False)

    # write_df_to_table(df_final, reference_code, "datavvh_simplificada")
    print("Datos de Referencia: " + reference_code + " procesados de manera satisfactoria")

def add_slope_percent_per_100m(data: pd.DataFrame,
                               dist_col: str = "v_distancem",
                               height_col: str = "v_height",
                               order_col: str = "v_elapsedt",
                               out_col: str = "slope_percent") -> pd.DataFrame:
    """
    Agrega % Slope por tramos de 100 m.
    - Ordena por `order_col`
    - Bins: [0-100], [100-200], ...
    - slope = (altura_final - altura_inicial) / 100
    """
    dfc = data.copy()
    for c in (dist_col, height_col, order_col):
        dfc[c] = pd.to_numeric(dfc[c], errors="coerce")

    dfc = dfc.sort_values(order_col).reset_index(drop=True)
    dfc["_bin_100m"] = (dfc[dist_col] // 100).astype("Int64")

    first_h = dfc.groupby("_bin_100m")[height_col].first()
    last_h  = dfc.groupby("_bin_100m")[height_col].last()

    first_x = dfc.groupby("_bin_100m")[dist_col].first()
    last_x  = dfc.groupby("_bin_100m")[dist_col].last()

    slope_by_bin = round( (last_h - first_h) / (last_x - first_x), 2)

    dfc[out_col] = dfc["_bin_100m"].map(slope_by_bin)
    return dfc.drop(columns=["_bin_100m"])

def add_empty_loaded_labels(
    df: pd.DataFrame,
    weight_col: str = "v_vehiclew",
    out_col: str = "loading_status",
    p_low: float = 0.10,
    p_high: float = 0.90,
    min_gap: float = 5.0   # diferencia mínima entre p10 y p90 en toneladas
) -> pd.DataFrame:
    """
    Etiqueta filas como Empty / Partial Loaded / Loaded según percentiles del peso del vehículo.
    - <= p_low -> 'Empty'
    - >= p_high -> 'Loaded'
    - entre -> 'Partial Loaded'
    Casos ambiguos (p10≈p90 o poca variación) -> 'Not Apply'
    """
    df2 = df.copy()
    df2[weight_col] = pd.to_numeric(df2[weight_col], errors="coerce")

    s = df2[weight_col].dropna().astype(float)

    if len(s) == 0:
        df2[out_col] = "Not Apply"
        return df2

    # percentiles
    p10 = np.nanpercentile(s, p_low * 100)
    p90 = np.nanpercentile(s, p_high * 100)

    # chequeos
    if not np.isfinite(p10) or not np.isfinite(p90) or (p90 - p10) < min_gap:
        df2[out_col] = "Not Apply"
        return df2

    df2[out_col] = df2[weight_col].apply(
        lambda v: "Empty" if v <= p10 else ("Loaded" if v >= p90 else "Partial Loaded")
    )

    return df2

def isolation_forest_preprocess(df: object, outliers_percent: float = 0.05):
    X = df[["v_speed", "v_height"]].values  # Convert to NumPy array

    # Apply Isolation Forest
    iso_forest = IsolationForest(contamination=outliers_percent, random_state=0)
    outlier_predictions = iso_forest.fit_predict(X)

    # Filter out the outliers (-1 are outliers, 1 are inliers)
    df_filtered = df[outlier_predictions == 1].copy()

    return df_filtered  # Return cleaned dataset

def add_h3_cell_and_centroid(df: object, resolution: int = 12):
    df_h3 = df.copy()
    df_h3["latitudet"]  = pd.to_numeric(df_h3["latitudet"],  errors="coerce")
    df_h3["longitudet"] = pd.to_numeric(df_h3["longitudet"], errors="coerce")

    # Add h3_r12 and h3_r11 columns in the same loop
    df_h3["h3_cell"] = [
        h3.latlng_to_cell(lat, lng, resolution) 
        for lat, lng in zip(df_h3["latitudet"], df_h3["longitudet"])
    ]

    df_h3[["centroid_lat", "centroid_lng"]] = pd.DataFrame(
        df_h3["h3_cell"].apply(h3.cell_to_latlng).tolist(),
        index=df_h3.index
    )

    return df_h3

def _circular_mean_deg(deg_series: pd.Series) -> float:
    r = np.radians(pd.to_numeric(deg_series, errors="coerce").dropna().values)
    if r.size == 0:
        return np.nan
    s, c = np.sin(r).mean(), np.cos(r).mean()
    ang = np.degrees(np.arctan2(s, c))
    return (ang + 360) % 360

def simplify_by_h3_cell(
    df: pd.DataFrame,
    group_by_loadstate: bool = True,
    alpha: float = 0.5  # blend for weight_transparency: 0=only flow, 1=only count
) -> pd.DataFrame:
    """
    Aggregate dataset by h3_cell. Expects:
      - h3_cell, centroid_lat, centroid_lng
      - v_speed, v_heading
      - optional: loading_status (if group_by_loadstate=True)
    Returns one row per h3_cell (or per h3_cell+loading_status).
    """
    d = df.copy()
    numeric_columns = ["v_speed", "v_heading", "centroid_lat", "centroid_lng", "v_vertical", "v_height", "v_longitudinala", "v_laterala", "v_radiusot", "v_vehiclew", "v_txphf", "v_txphr", "slope_percent"]
    # ensure numeric
    for c in numeric_columns:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # each row = 1s → flow_km = speed(kmh) * 1/3600 h
    d["flow_km"] = d["v_speed"] / 3600.0
    # group keys
    keys = ["h3_cell"] + (["loading_status"] if group_by_loadstate and "loading_status" in d.columns else [])

    agg = (
        d.groupby(keys, dropna=False)
         .agg(
             n_points=("h3_cell","size"),
             centroid_lat=("centroid_lat","first"),
             centroid_lng=("centroid_lng","first"),
             avg_heading=("v_heading", _circular_mean_deg),
             avg_speed=("v_speed","mean"),
             avg_vertical_speed=("v_vertical", "mean"),
             avg_height=("v_height", "mean"),
             avg_longitudinal_a=("v_longitudinala", "mean"),
             avg_lateral_a=("v_laterala", "mean"),
             avg_ratius_ot=("v_radiusot", "mean"),
             avg_vehicle_w=("v_vehiclew", "mean"),
             avg_tkph_f=("v_txphf", "mean"),
             avg_tkph_r=("v_txphr", "mean"),
             avg_slope_percent=("slope_percent", "mean"),
             sum_flow_km=("flow_km","sum")
         )
         .reset_index()
    )

    # normalize weights
    n_norm = agg["n_points"] / (agg["n_points"].max() or 1)
    f_norm = agg["sum_flow_km"] / (agg["sum_flow_km"].max() or 1.0)
    agg["transparency"] = (alpha * n_norm + (1 - alpha) * f_norm).clip(0, 1)

    agg = agg.drop(columns=["sum_flow_km"])

    # tidy rounding
    agg = agg.round({
        "avg_heading": 2,
        "avg_speed": 2,
        "avg_vertical_speed": 2,
        "avg_height": 2,
        "avg_longitudinal_a": 2,
        "avg_lateral_a": 2,
        "avg_ratius_ot": 2,
        "avg_vehicle_w": 2,
        "avg_tkph_f": 2,
        "avg_tkph_r": 2,
        "avg_slope_percent": 4,
        "transparency": 4
    })

    return agg

def write_df_to_table(df, reference_code, table_name):

    # Validate that the reference exist, in that case you CAN'T insert the data
    if validate_reference_code(reference_code):
        print(f"🚫 Skipping {reference_code}: Record already exists in datavvh_simplificada.")
        return
    
    # Create table if it doesn't exist (adjust column types as needed)
    create_table_if_dont_exist(table_name, df.columns)

    # Prepare INSERT query dynamically based on column names
    insert_df_to_table(df, table_name)

    print(f"✅ Data from reference {reference_code} written to table 'datavvh_simplificada' successfully!")

#----------------
# First Time Load
#----------------

def first_time_load():
    result = fetch_all_vbox_reference_records()
    
    # Extract refe codes as a list
    refelist = [row[0] for row in result]

    for reference in refelist:
        try:
            request_processor(reference)
        except Exception as e:
            print(f"🚫 Skipping {reference}: Record has an error in DataTypes → {e}")



def add_clustering_columns_to_dataframe(df):
    """
    Toma el DataFrame original y agrega columnas con resultados de clustering y KDE
    
    Parámetros:
    df: DataFrame original con todas las filas
    
    Retorna:
    df: Mismo DataFrame pero con columnas nuevas:
        - cluster_r1, kde_r1 (velocidad alta)
        - cluster_r2, kde_r2 (aceleración lateral) 
        - cluster_r3, kde_r3 (aceleración longitudinal)
        - cluster_r4, kde_r4 (velocidad vertical)
        - cluster_r5, kde_r5 (ondulaciones rápidas)
        - cluster_r6, kde_r6 (curvas rápidas)
    """
    
    print("🚀 AGREGANDO COLUMNAS DE CLUSTERING AL DATAFRAME ORIGINAL")
    print("=" * 60)
    
    # Crear copia del DataFrame para no modificar el original
    df_result = df.copy()
    
    # Verificar columnas necesarias
    required_columns = ['centroid_lat', 'centroid_lng', 'avg_speed', 'avg_lateral_a', 
                       'avg_longitudinal_a', 'avg_vertical_speed']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ ERROR: Columnas faltantes: {missing_columns}")
        return df_result
    
    # Definir las reglas
    rules_definitions = {
        'r1': {
            'name': 'velocidad_alta',
            'description': 'Velocidades > 50 km/h',
            'condition': lambda df: df['avg_speed'] > 50
        },
        'r2': {
            'name': 'aceleracion_lateral', 
            'description': 'Aceleración lateral > ±0.15g',
            'condition': lambda df: np.abs(df['avg_lateral_a']) > 0.15
        },
        'r3': {
            'name': 'aceleracion_longitudinal',
            'description': 'Aceleración longitudinal > ±0.1g', 
            'condition': lambda df: np.abs(df['avg_longitudinal_a']) > 0.1
        },
        'r4': {
            'name': 'velocidad_vertical',
            'description': 'Velocidad vertical > ±3 m/s',
            'condition': lambda df: np.abs(df['avg_vertical_speed']) > 3
        }
    }
    
    # Regla 5: Ondulaciones rápidas 
    if 'avg_vertical_speed' in df.columns:
        ondulacion_score = (df['avg_speed'] > 50) & (df['avg_vertical_speed'] > 3)
        threshold_ondulacion = (df['avg_speed'] > 50) & (df['avg_vertical_speed'] > 3)
        rules_definitions['r5'] = {
            'name': 'ondulaciones_rapidas',
            'description': f'Ondulaciones rápidas (Sver×Speed) avg_speed>50 & avg_vertical_speed',
            'condition': lambda df: (df['avg_speed'] > 50) & (df['avg_vertical_speed'] > 3)
        }
    
    # Regla 6: Curvas rápidas (si existe columna avg_radius_ot)
    if 'avg_ratius_ot' in df.columns:
        rules_definitions['r6'] = {
            'name': 'curvas_rapidas',
            'description': 'Curvas rápidas (velocidad excede la máxima permitida para el radio)',
            'condition': lambda df: df.apply(
                lambda row: row['avg_speed'] > get_max_speed_from_radius(row['avg_ratius_ot']),
                axis=1
            )
        }
    
    # Procesar cada regla
    for rule_id, rule_info in rules_definitions.items():
        
        print(f"\n🎯 Procesando {rule_id.upper()}: {rule_info['description']}")
        
        try:
            # Aplicar condición para identificar eventos críticos
            critical_mask = rule_info['condition'](df_result)
            critical_indices = df_result[critical_mask].index
            
            print(f"   📍 Eventos críticos encontrados: {len(critical_indices)}")
            
            # Inicializar columnas con -999 (no aplica) y NaN para KDE
            cluster_col = f'cluster_{rule_id}'
            kde_col = f'kde_{rule_id}'
            
            df_result[cluster_col] = -999  # -999 = no es evento crítico para esta regla
            df_result[kde_col] = np.nan   # NaN = no tiene densidad KDE
            
            if len(critical_indices) < 5:
                print(f"   ⚠️ Muy pocos eventos críticos para clustering (mínimo 5)")
                # Las columnas quedan con valores por defecto
                continue
            
            # Extraer coordenadas de eventos críticos
            critical_coords = df_result.loc[critical_indices, ['centroid_lat', 'centroid_lng']].dropna()
            valid_indices = critical_coords.index
            coords_array = critical_coords.values
            
            if len(coords_array) < 5:
                print(f"   ⚠️ Muy pocos eventos con coordenadas válidas")
                continue
            
            # Aplicar HDBSCAN
            n_points = len(coords_array)
            min_cluster_size = max(3, n_points // 30)
            min_samples = max(2, min_cluster_size // 2)
            
            hdbscan = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=0.0005,
                metric='euclidean'
            )
            
            clusters = hdbscan.fit_predict(coords_array)
            
            # Asignar resultados de clustering
            df_result.loc[valid_indices, cluster_col] = clusters
            
            # Estadísticas de clustering
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            n_noise = list(clusters).count(-1)
            n_in_clusters = len(clusters) - n_noise
            
            print(f"   🎯 Hotspots identificados: {n_clusters}")
            print(f"   ✅ Eventos en hotspots: {n_in_clusters}")
            print(f"   🔀 Eventos de ruido: {n_noise}")
            
            # Aplicar KDE si hay suficientes puntos en clusters
            if n_clusters > 0 and n_in_clusters >= 5:
                
                # KDE solo en puntos que están en clusters (no ruido)
                cluster_mask = clusters != -1
                clustered_coords = coords_array[cluster_mask]
                clustered_indices = valid_indices[cluster_mask]
                
                try:
                    kde = KernelDensity(bandwidth=0.003, kernel='epanechnikov')
                    kde.fit(clustered_coords)
                    
                    # Calcular densidad KDE para todos los eventos críticos
                    kde_scores = kde.score_samples(coords_array)
                    kde_density = np.exp(kde_scores)
                    probs = kde_density / kde_density.sum()
                    
                    # Asignar densidades KDE
                    df_result.loc[valid_indices, kde_col] = probs
                    
                    print(f"   📊 KDE calculado para {len(valid_indices)} puntos")
                    print(f"   📈 Densidad máxima: {kde_density.max():.6f}")
                    
                except Exception as kde_error:
                    print(f"   ⚠️ Error en KDE: {kde_error}")
            
            else:
                print(f"   ⚠️ No se puede calcular KDE (pocos clusters o puntos)")
        
        except Exception as rule_error:
            print(f"   ❌ Error procesando regla {rule_id}: {rule_error}")
            continue
    
    # Mostrar resumen final
    print(f"\n{'📊 RESUMEN FINAL':=^60}")
    
    cluster_columns = [col for col in df_result.columns if col.startswith('cluster_')]
    kde_columns = [col for col in df_result.columns if col.startswith('kde_')]
    
    print(f"✅ Columnas de clustering agregadas: {len(cluster_columns)}")
    for col in cluster_columns:
        n_clusters = len(df_result[df_result[col] >= 0][col].unique())
        n_in_clusters = len(df_result[df_result[col] >= 0])
        n_noise = len(df_result[df_result[col] == -1])
        print(f"   {col}: {n_clusters} hotspots, {n_in_clusters} eventos agrupados, {n_noise} ruido")
    
    print(f"✅ Columnas de KDE agregadas: {len(kde_columns)}")
    for col in kde_columns:
        n_with_kde = len(df_result[df_result[col].notna()])
        print(f"   {col}: {n_with_kde} eventos con densidad KDE")
    
    print(f"\n📏 DataFrame final: {len(df_result)} filas × {len(df_result.columns)} columnas")
    print(f"📏 Nuevas columnas agregadas: {len(cluster_columns + kde_columns)}")
    
    return df_result

def get_clustering_summary(df_with_clusters):
    """
    Función auxiliar para obtener resumen de los resultados de clustering
    """
    
    print("📊 RESUMEN DETALLADO DE CLUSTERING")
    print("=" * 50)
    
    cluster_columns = [col for col in df_with_clusters.columns if col.startswith('cluster_')]
    kde_columns = [col for col in df_with_clusters.columns if col.startswith('kde_')]
    
    summary_data = []
    
    for col in cluster_columns:
        rule_id = col.split('_')[1]  # Extraer r1, r2, etc.
        
        # Estadísticas de clustering
        all_clusters = df_with_clusters[col]
        critical_events = all_clusters[all_clusters != -999]  # Eventos que aplican a esta regla
        
        if len(critical_events) == 0:
            continue
            
        n_total_critical = len(critical_events)
        n_in_clusters = len(critical_events[critical_events >= 0])
        n_noise = len(critical_events[critical_events == -1])
        n_unique_clusters = len(critical_events[critical_events >= 0].unique())
        
        clustering_rate = (n_in_clusters / n_total_critical * 100) if n_total_critical > 0 else 0
        
        # Estadísticas de KDE
        kde_col = f'kde_{rule_id}'
        n_with_kde = 0
        max_kde = 0
        if kde_col in df_with_clusters.columns:
            kde_values = df_with_clusters[kde_col].dropna()
            n_with_kde = len(kde_values)
            max_kde = kde_values.max() if len(kde_values) > 0 else 0
        
        summary_data.append({
            'Regla': rule_id.upper(),
            'Eventos_Críticos': n_total_critical,
            'Hotspots': n_unique_clusters,
            'En_Clusters': n_in_clusters,
            'Ruido': n_noise,
            'Tasa_Agrupamiento_%': f"{clustering_rate:.1f}%",
            'Con_KDE': n_with_kde,
            'Max_Densidad_KDE': f"{max_kde:.6f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    return summary_df


# ===============================
# FUNCIÓN PRINCIPAL PARA USAR
# ===============================

def process_dataframe_with_clustering(df, export_coordinates=True):
    """
    Función principal que procesa todo el DataFrame
    
    Parámetros:
    df: DataFrame original
    export_coordinates: Si exportar coordenadas de hotspots
    
    Retorna:
    dict con:
    - 'dataframe': DataFrame con columnas nuevas
    - 'summary': Resumen de clustering
    - 'hotspots': Coordenadas de hotspots (si export_coordinates=True)
    """
    
    print("🚀 PROCESAMIENTO COMPLETO DEL DATAFRAME")
    print("=" * 60)
    
    # Agregar columnas de clustering
    df_with_clusters = add_clustering_columns_to_dataframe(df)
    
    # Generar resumen
    summary = get_clustering_summary(df_with_clusters)
       
    return {
        'dataframe': df_with_clusters,
        'summary': summary, 
    }

# Rango de radios (m) → Velocidad máxima (kph)
radius_speed_ranges = [
    (0, 15, 8),
    (15, 20, 9),
    (20, 25, 10),
    (25, 32, 11),
    (32, 37, 12),
    (37, 42, 13),
    (42, 49, 14),
    (49, 53, 15),
    (53, 58, 16),
    (58, 63, 17),
    (63, 68, 18),
    (68, 73, 19),
    (73, 79, 20),
    (79, 84, 21),
    (84, 90, 22),
    (90, 96, 23),
    (96, 106, 24),
    (106, 113, 25),
    (113, 120, 26),
    (120, 132, 27),
    (132, 141, 28),
    (141, 150, 29),
    (150, 160, 30),
    (160, 170, 31),
    (170, 198, 32)
]

def get_max_speed_from_radius(radius):
    for r_min, r_max, vmax in radius_speed_ranges:
        if r_min <= radius <= r_max:
            return vmax
    return np.inf  # si está fuera de los rangos, no marcar como crítico


