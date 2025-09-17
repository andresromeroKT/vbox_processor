from functionalities.requests import fetch_vbox_data_as_dataframe, validate_reference_code, create_table_if_dont_exist, insert_df_to_table, fetch_all_vbox_reference_records
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

import h3

def request_processor(reference_code: str):

    vbox_df = fetch_vbox_data_as_dataframe(reference_code)
    vbox_df.to_csv("fetched_data.csv", index=False)
    # vbox_df_iso_forest = isolation_forest_preprocess(vbox_df)
    vbox_df_with_slope =  add_slope_percent_per_100m(vbox_df)
    vbox_df_with_slope_and_status = add_empty_loaded_labels(vbox_df_with_slope)
    vbox_df_with_slope_status_h3 = add_h3_cell_and_centroid(vbox_df_with_slope_and_status)

    vbox_df_with_slope_status_h3.to_csv("slopes_status_h3.csv", index=False)

    vbox_final_simplified_df = simplify_by_h3_cell(vbox_df_with_slope_status_h3)
    vbox_final_simplified_df.to_csv("simplified_vbox_data.csv", index=False)

    # write_df_to_table(vbox_final_simplified_df, reference_code, "datavvh_simplificada")
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

