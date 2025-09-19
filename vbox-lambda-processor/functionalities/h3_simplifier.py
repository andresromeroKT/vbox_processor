# functionalities/h3_simplifier.py (MODIFICADO PARA LAMBDA)
from functionalities.requests import fetch_vbox_data_as_dataframe, validate_reference_code, create_table_if_dont_exist, insert_df_to_table, fetch_all_vbox_reference_records
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import KernelDensity
import h3
import logging
import os
import tempfile

# Configure logging
logger = logging.getLogger()

def request_processor(reference_code: str):
    """
    Process a single reference code - optimized for Lambda
    """
    try:
        logger.info(f"Starting processing for reference: {reference_code}")
        
        # Step 1: Fetch data
        logger.info("Fetching VBox data...")
        vbox_df = fetch_vbox_data_as_dataframe(reference_code)
        
        if vbox_df.empty:
            raise ValueError(f"No data found for reference: {reference_code}")
        
        logger.info(f"Fetched {len(vbox_df)} records")
        
        # Step 2: Process data pipeline
        logger.info("Applying isolation forest...")
        vbox_df_iso_forest = isolation_forest_preprocess(vbox_df)
        
        logger.info("Adding slope calculations...")
        vbox_df_with_slope = add_slope_percent_per_100m(vbox_df_iso_forest)
        
        logger.info("Adding loading status labels...")
        vbox_df_with_slope_and_status = add_empty_loaded_labels(vbox_df_with_slope)
        
        logger.info("Adding H3 cells and centroids...")
        vbox_df_with_slope_status_h3 = add_h3_cell_and_centroid(vbox_df_with_slope_and_status)
        
        logger.info("Simplifying by H3 cells...")
        vbox_final_simplified_df = simplify_by_h3_cell(vbox_df_with_slope_status_h3)
        
        logger.info("Processing clustering analysis...")
        result = process_dataframe_with_clustering(vbox_final_simplified_df)
        df_final = result['dataframe']
        
        # Step 3: Write to database
        logger.info("Writing processed data to database...")
        write_df_to_table(df_final, reference_code, "datavvh_simplificada_v2")
        
        logger.info(f"Successfully processed reference: {reference_code}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing reference {reference_code}: {str(e)}")
        raise

def write_df_to_table(df, reference_code, table_name):
    """
    Write DataFrame to database table - with better error handling for Lambda
    """
    try:
        # Validate that the reference doesn't exist
        if validate_reference_code(reference_code):
            logger.warning(f"Skipping {reference_code}: Record already exists in {table_name}")
            return False
        
        # Create table if it doesn't exist
        create_table_if_dont_exist(table_name, df.columns)

        df['refe'] = reference_code

        # Insert data
        insert_df_to_table(df, table_name)
        
        logger.info(f"Successfully wrote {len(df)} records for reference {reference_code}")
        return True
        
    except Exception as e:
        logger.error(f"Error writing to database: {str(e)}")
        raise

# Keep all your existing processing functions exactly the same
def add_slope_percent_per_100m(data: pd.DataFrame,
                               dist_col: str = "v_distancem",
                               height_col: str = "v_height",
                               order_col: str = "v_elapsedt",
                               out_col: str = "slope_percent") -> pd.DataFrame:
    """
    Agrega % Slope por tramos de 100 m.
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
    min_gap: float = 5.0
) -> pd.DataFrame:
    """
    Etiqueta filas como Empty / Partial Loaded / Loaded según percentiles del peso del vehículo.
    """
    df2 = df.copy()
    df2[weight_col] = pd.to_numeric(df2[weight_col], errors="coerce")

    s = df2[weight_col].dropna().astype(float)

    if len(s) == 0:
        df2[out_col] = "Not Apply"
        return df2

    p10 = np.nanpercentile(s, p_low * 100)
    p90 = np.nanpercentile(s, p_high * 100)

    if not np.isfinite(p10) or not np.isfinite(p90) or (p90 - p10) < min_gap:
        df2[out_col] = "Not Apply"
        return df2

    df2[out_col] = df2[weight_col].apply(
        lambda v: "Empty" if v <= p10 else ("Loaded" if v >= p90 else "Partial Loaded")
    )

    return df2

def isolation_forest_preprocess(df: object, outliers_percent: float = 0.05):
    """
    Apply Isolation Forest to remove outliers
    """
    X = df[["v_speed", "v_height"]].values

    iso_forest = IsolationForest(contamination=outliers_percent, random_state=0)
    outlier_predictions = iso_forest.fit_predict(X)

    df_filtered = df[outlier_predictions == 1].copy()
    return df_filtered

def add_h3_cell_and_centroid(df: object, resolution: int = 12):
    """
    Add H3 cells and centroids
    """
    df_h3 = df.copy()
    df_h3["latitudet"]  = pd.to_numeric(df_h3["latitudet"],  errors="coerce")
    df_h3["longitudet"] = pd.to_numeric(df_h3["longitudet"], errors="coerce")

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
    """
    Calculate circular mean for heading degrees
    """
    r = np.radians(pd.to_numeric(deg_series, errors="coerce").dropna().values)
    if r.size == 0:
        return np.nan
    s, c = np.sin(r).mean(), np.cos(r).mean()
    ang = np.degrees(np.arctan2(s, c))
    return (ang + 360) % 360

def simplify_by_h3_cell(
    df: pd.DataFrame,
    group_by_loadstate: bool = True,
    alpha: float = 0.5
) -> pd.DataFrame:
    """
    Aggregate dataset by h3_cell
    """
    d = df.copy()
    numeric_columns = ["v_speed", "v_heading", "centroid_lat", "centroid_lng", "v_vertical", "v_height", "v_longitudinala", "v_laterala", "v_radiusot", "v_vehiclew", "v_txphf", "v_txphr", "slope_percent"]
    
    for c in numeric_columns:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    d["flow_km"] = d["v_speed"] / 3600.0
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

    n_norm = agg["n_points"] / (agg["n_points"].max() or 1)
    f_norm = agg["sum_flow_km"] / (agg["sum_flow_km"].max() or 1.0)
    agg["transparency"] = (alpha * n_norm + (1 - alpha) * f_norm).clip(0, 1)

    agg = agg.drop(columns=["sum_flow_km"])

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

def add_clustering_columns_to_dataframe(df):
    """
    Add clustering and KDE columns to DataFrame
    """
    df_result = df.copy()
    logger.info("Adding clustering columns to dataframe")
    if 'avg_ratius_ot' in df_result.columns:
        df_result['avg_ratius_ot'] = df_result['avg_ratius_ot'].fillna(1000)
    
    # Después llenar todos los demás NaN con 0
    df_result = df_result.fillna(0)    
    
    required_columns = ['centroid_lat', 'centroid_lng', 'avg_speed', 'avg_lateral_a', 
                       'avg_longitudinal_a', 'avg_vertical_speed']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.warning(f"Missing columns for clustering: {missing_columns}")
        return df_result
    
    # Define clustering rules
    rules_definitions = {
        'r1': {
            'name': 'speed',
            'description': 'Velocidades > 50 km/h',
            'condition': lambda df: df['avg_speed'] > 50
        },
        'r2': {
            'name': 'lateral_ac', 
            'description': 'Aceleración lateral > ±0.15g',
            'condition': lambda df: np.abs(df['avg_lateral_a']) > 0.15
        },
        'r3': {
            'name': 'longitudinal_ac',
            'description': 'Aceleración longitudinal > ±0.1g', 
            'condition': lambda df: np.abs(df['avg_longitudinal_a']) > 0.1
        },
        'r4': {
            'name': 'vertical_speed',
            'description': 'Velocidad vertical > ±3 m/s',
            'condition': lambda df: np.abs(df['avg_vertical_speed']) > 3
        }
    }
    
    # Add rules 5 and 6 if columns exist
    if 'avg_vertical_speed' in df.columns:
        rules_definitions['r5'] = {
            'name': 'speed_waviness',
            'description': 'Ondulaciones rápidas',
            'condition': lambda df: (df['avg_speed'] > 50) & (df['avg_vertical_speed'] > 3)
        }
    
    if 'avg_ratius_ot' in df.columns:
        rules_definitions['r6'] = {
            'name': 'speed_turns',
            'description': 'Curvas rápidas',
            'condition': lambda df: df.apply(
                lambda row: row['avg_speed'] > get_max_speed_from_radius(row['avg_ratius_ot']),
                axis=1
            )
        }
    
    # Process each rule
    for rule_id, rule_info in rules_definitions.items():
        logger.info(f"Processing rule {rule_id}: {rule_info['description']}")
        
        try:
            critical_mask = rule_info['condition'](df_result)
            critical_indices = df_result[critical_mask].index
            
            cluster_col = f'cluster_{rule_info["name"]}'
            kde_col = f'kde_{rule_info["name"]}'
            
            df_result[cluster_col] = -999
            df_result[kde_col] = 0.0
            
            if len(critical_indices) < 5:
                logger.warning(f"Too few critical events for rule {rule_id}")
                continue
            
            critical_coords = df_result.loc[critical_indices, ['centroid_lat', 'centroid_lng']].dropna()
            valid_indices = critical_coords.index
            coords_array = critical_coords.values
            
            if len(coords_array) < 5:
                continue
            
            # Apply HDBSCAN
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
            df_result.loc[valid_indices, cluster_col] = clusters
            
            # Apply KDE
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            n_in_clusters = len(clusters) - list(clusters).count(-1)
            
            if n_clusters > 0 and n_in_clusters >= 5:
                cluster_mask = clusters != -1
                clustered_coords = coords_array[cluster_mask]
                
                try:
                    kde = KernelDensity(bandwidth=0.003, kernel='epanechnikov')
                    kde.fit(clustered_coords)
                    
                    all_coords = df_result[['centroid_lat', 'centroid_lng']].dropna()
                    all_valid_indices = all_coords.index
                    all_coords_array = all_coords.values

                    kde_scores = kde.score_samples(all_coords_array)
                    kde_density = np.exp(kde_scores)
                    probs = kde_density / kde_density.sum()
                    
                    df_result.loc[all_valid_indices, kde_col] = probs
                    
                except Exception as kde_error:
                    logger.warning(f"KDE error for rule {rule_id}: {kde_error}")
        
        except Exception as rule_error:
            logger.error(f"Error processing rule {rule_id}: {rule_error}")
            continue
    
    return df_result

def get_clustering_summary(df_with_clusters):
    """Generate clustering summary"""
    cluster_columns = [col for col in df_with_clusters.columns if col.startswith('cluster_')]
    kde_columns = [col for col in df_with_clusters.columns if col.startswith('kde_')]
    
    summary_data = []
    
    for col in cluster_columns:
        rule_id = col.split('_')[1]
        all_clusters = df_with_clusters[col]
        critical_events = all_clusters[all_clusters != -999]
        
        if len(critical_events) == 0:
            continue
            
        n_total_critical = len(critical_events)
        n_in_clusters = len(critical_events[critical_events >= 0])
        n_noise = len(critical_events[critical_events == -1])
        n_unique_clusters = len(critical_events[critical_events >= 0].unique())
        
        clustering_rate = (n_in_clusters / n_total_critical * 100) if n_total_critical > 0 else 0
        
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
    
    return pd.DataFrame(summary_data)

def process_dataframe_with_clustering(df, export_coordinates=True):
    """Main function to process DataFrame with clustering"""
    logger.info("Processing dataframe with clustering")
    
    df_with_clusters = add_clustering_columns_to_dataframe(df)
    summary = get_clustering_summary(df_with_clusters)
       
    return {
        'dataframe': df_with_clusters,
        'summary': summary, 
    }

# Radius-speed mapping
radius_speed_ranges = [
    (0, 15, 8), (15, 20, 9), (20, 25, 10), (25, 32, 11), (32, 37, 12),
    (37, 42, 13), (42, 49, 14), (49, 53, 15), (53, 58, 16), (58, 63, 17),
    (63, 68, 18), (68, 73, 19), (73, 79, 20), (79, 84, 21), (84, 90, 22),
    (90, 96, 23), (96, 106, 24), (106, 113, 25), (113, 120, 26), (120, 132, 27),
    (132, 141, 28), (141, 150, 29), (150, 160, 30), (160, 170, 31), (170, 198, 32)
]

def get_max_speed_from_radius(radius):
    """Get maximum speed for given radius"""
    for r_min, r_max, vmax in radius_speed_ranges:
        if r_min <= radius <= r_max:
            return vmax
    return np.inf
