import mysql.connector
import pandas as pd
import time
from dotenv import load_dotenv
import os

# Carga las variables desde el archivo .env
load_dotenv()

def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )

def fetch_vbox_refes() -> pd.DataFrame:
    mydb = get_db_connection()
    mycursor = mydb.cursor()

    query = f'''
        SELECT 
            refe,
            count(*)
        FROM data_vbox 
        WHERE v_utc <> 'UTC time'
        GROUP BY refe
    '''

    mycursor.execute(query)
    myresult = mycursor.fetchall()
    columns = [col[0] for col in mycursor.description]
    
    mycursor.close()
    mydb.close()
    
    return pd.DataFrame(myresult, columns=columns)

def fetch_vbox_data_as_dataframe(reference_code: str) -> pd.DataFrame:
    mydb = get_db_connection()
    mycursor = mydb.cursor()
    
    query = '''
        SELECT 
            v_utc,
            v_satellites,
            v_speed,
            v_heading,
            v_height,
            v_vertical,
            v_longitudinala,
            v_laterala,
            v_longitudinalb,
            v_lateralb,
            v_elapsedt,
            v_distancem,
            v_pordistance,
            v_date,
            v_day,
            v_radiusot,
            v_movingf,
            v_movingr,
            v_vehiclew,
            latitudet,
            longitudet,
            refe
        FROM data_vbox 
        WHERE refe = %s
        AND v_utc <> 'UTC time'
    '''
    
    mycursor.execute(query, (reference_code,))
    myresult = mycursor.fetchall()
    columns = [col[0] for col in mycursor.description]
    
    mycursor.close()
    mydb.close()
    
    df = pd.DataFrame(myresult, columns=columns)

    # Define invalid/error markers
    error_markers = ["#¡VALOR!", "#VALUE!", "#N/A", "NaN", "NULL"]

    # Replace commas with dots and filter errors
    def clean_value(x):
        if isinstance(x, str):
            if any(err in x for err in error_markers):
                return None   # mark for removal
            return x.replace(",", ".")
        return x

    # Apply cleaning
    df = df.applymap(clean_value)

    # Drop rows with invalid values
    df = df.dropna(how="any")

    return df

def validate_reference_code(reference_code: str) -> bool:
    mydb = get_db_connection()
    mycursor = mydb.cursor()
    
    validation_query = f"SELECT COUNT(*) FROM datavvh_simplificada WHERE refe = '{reference_code}'"
    mycursor.execute(validation_query)
    count_result = mycursor.fetchone()
    
    mycursor.close()
    mydb.close()
    
    return count_result[0] >= 1 if count_result else False

def create_table_if_dont_exist(table_name: str, columns: list):
    mydb = get_db_connection()
    mycursor = mydb.cursor()
    
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        {', '.join([f'`{col}` VARCHAR(255)' for col in columns])}
    )
    """
    
    mycursor.execute(create_table_query)
    mydb.commit()
    
    mycursor.close()
    mydb.close()

def insert_df_to_table(df, table_name: str):
    mydb = get_db_connection()
    mycursor = mydb.cursor()
    
    placeholders = ', '.join(['%s'] * len(df.columns))
    columns = ', '.join([f'`{col}`' for col in df.columns])
    insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
    
    values = [tuple(row) for row in df.itertuples(index=False, name=None)]
    mycursor.executemany(insert_query, values)
    mydb.commit()
    
    mycursor.close()
    mydb.close()
    
    time.sleep(5)

def fetch_all_vbox_reference_records():
    mydb = get_db_connection()
    mycursor = mydb.cursor()
    
    mycursor.execute("SELECT DISTINCT refe FROM data_vbox")
    result = mycursor.fetchall()
    
    mycursor.close()
    mydb.close()
    
    return result
