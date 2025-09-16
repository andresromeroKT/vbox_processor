from functionalities.requests import fetch_vbox_data_as_dataframe, validate_reference_code, create_table_if_dont_exist, insert_df_to_table, fetch_all_vbox_reference_records
import pandas as pd
from sklearn.ensemble import IsolationForest

import h3

def request_processor(reference_code: str):

    vbox_df = fetch_vbox_data_as_dataframe(reference_code)
    vbox_df_iso_forest = isolation_forest_preprocess(vbox_df)
    vbox_df_iso_forest_h3_simplification = h3_simplification(vbox_df_iso_forest)

    write_df_to_table(vbox_df_iso_forest_h3_simplification, reference_code, "datavvh_simplificada")

    vbox_df_iso_forest_h3_simplification.to_csv("grouped_data.csv", index=False)
    
def isolation_forest_preprocess(df: object, outliers_percent: float = 0.05):
    X = df[["v_speed", "v_height"]].values  # Convert to NumPy array

    # Apply Isolation Forest
    iso_forest = IsolationForest(contamination=outliers_percent, random_state=0)
    outlier_predictions = iso_forest.fit_predict(X)

    # Filter out the outliers (-1 are outliers, 1 are inliers)
    df_filtered = df[outlier_predictions == 1].copy()

    return df_filtered  # Return cleaned dataset

def h3_simplification(df: object):
    # Add h3_r12 and h3_r11 columns in the same loop
    df[["h3_r12", "h3_r11"]] = df.apply(
    lambda row: pd.Series({
        "h3_r12": h3.latlng_to_cell(float(row["latitudet"]), float(row["longitudet"]), 12),
        "h3_r11": h3.latlng_to_cell(float(row["latitudet"]), float(row["longitudet"]), 11)
    }), 
    axis=1
)

    # Separate numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include="number").columns
    non_numeric_cols = df.select_dtypes(exclude="number").columns

    # Group by h3_r12
    df_grouped = df.groupby("h3_r12", as_index=False).agg(
        {col: "mean" for col in numeric_cols} | {col: "max" for col in non_numeric_cols}
    )

    # Round all numeric columns to 2 decimal places
    df_grouped[numeric_cols] = df_grouped[numeric_cols].round(2)

    return df_grouped

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

