import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_table
import pandas as pd
import folium
from folium.plugins import HeatMap
import base64
import io
import plotly.express as px
import os
import xlsxwriter
from fpdf import FPDF
import matplotlib.pyplot as plt
import numpy as np

# Inicializar la aplicación Dash
app = dash.Dash(__name__)

# Directorio para almacenar análisis previos
STORAGE_DIR = "stored_analysis"
os.makedirs(STORAGE_DIR, exist_ok=True)

app.layout = html.Div([
    html.H1("Análisis de VBox - Kal Tire", style={'textAlign': 'center', 'color': '#FF5733'}),
    
    # Carga de archivos
    dcc.Upload(
        id='upload-data',
        children=html.Button("Subir Archivo VBox"),
        multiple=False
    ),
    
    html.Div(id='output-data-upload'),
    
    # Controles de filtro
    html.H3("Filtros de Análisis"),
    dcc.RangeSlider(
        id='speed-filter',
        min=0,
        max=100,
        step=1,
        marks={i: str(i) for i in range(0, 101, 10)},
        value=[0, 100]
    ),
    dcc.RangeSlider(
        id='criticality-filter',
        min=0,
        max=1,
        step=0.05,
        marks={i/10: str(i/10) for i in range(0, 11)},
        value=[0, 1]
    ),
    
    # Análisis de variables críticas
    html.H3("Análisis de Variables Críticas"),
    html.Div(id='speed-analysis'),
    html.Div(id='lateral-effort-analysis'),
    html.Div(id='longitudinal-effort-analysis'),
    html.Div(id='map-analysis'),
    
    # Sección de análisis previos
    html.H3("Análisis Previos"),
    dcc.Dropdown(id='previous-analysis-dropdown', options=[], placeholder="Selecciona un análisis previo"),
    html.Button("Cargar Análisis Previo", id='load-previous-analysis', n_clicks=0),
    
    # Exportar reportes
    html.Button("Exportar a Excel", id='export-excel', n_clicks=0),
    html.Button("Exportar a PDF", id="export-pdf", n_clicks=0),
    dcc.Download(id='download-dataframe-xlsx'),
    dcc.Download(id='download-dataframe-pdf')
])

# Función para calcular criticidad
def calculate_criticality(df):
    df = df.copy()
    
    if "Latitude.1" in df.columns and "Longitude.1" in df.columns:
        df["Latitude"] = df["Latitude.1"]
        df["Longitude"] = df["Longitude.1"]
    
    df["Speed (km/h)"] = pd.to_numeric(df["Speed (km/h)"], errors='coerce').fillna(0)
    df["Radius of turn (m)"] = pd.to_numeric(df["Radius of turn (m)"], errors='coerce').replace(0, 0.1).fillna(df["Radius of turn (m)"].median())
    
    df["SpeedNormalized"] = df["Speed (km/h)"].clip(lower=0) / df["Speed (km/h)"].max()
    df["LateralEffort"] = 1 / (df["Radius of turn (m)"].fillna(1) + 1)
    df["LongitudinalEffort"] = df["Speed (km/h)"].diff().abs().fillna(0)
    df["LongitudinalEffort"] /= df["LongitudinalEffort"].max() or 1
    
    df["Criticality"] = (
        df["SpeedNormalized"] * 0.3 + df["LateralEffort"] * 0.4 + df["LongitudinalEffort"] * 0.3
    )
    
    return df

# Callback para procesar el archivo subido y guardar análisis previos
@app.callback(
    [Output('output-data-upload', 'children'),
     Output('speed-analysis', 'children'),
     Output('lateral-effort-analysis', 'children'),
     Output('longitudinal-effort-analysis', 'children'),
     Output('map-analysis', 'children')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    if contents is None:
        return "Sube un archivo para iniciar el análisis.", None, None, None, None
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    except UnicodeDecodeError:
        df = pd.read_csv(io.StringIO(decoded.decode('latin1')))
    
    df_processed = calculate_criticality(df)
    df_processed = df_processed.dropna(subset=["Latitude", "Longitude"])  # Eliminar NaNs en coordenadas
    file_path = os.path.join(STORAGE_DIR, f"processed_{filename}.csv")
    df_processed.to_csv(file_path, index=False)
    
    speed_fig = px.histogram(df_processed, x="Speed (km/h)", nbins=30, title="Distribución de Velocidades")
    lateral_fig = px.histogram(df_processed, x="LateralEffort", nbins=30, title="Distribución de Esfuerzos Laterales")
    longitudinal_fig = px.histogram(df_processed, x="LongitudinalEffort", nbins=30, title="Distribución de Esfuerzos Longitudinales")
    
    map_fig = folium.Map(location=[df_processed["Latitude"].mean(), df_processed["Longitude"].mean()], zoom_start=12)
    heat_data = df_processed[['Latitude', 'Longitude', 'Criticality']].values.tolist()
    HeatMap(heat_data).add_to(map_fig)
    map_html = map_fig._repr_html_()
    
    return html.H4(f"Archivo cargado y guardado: {filename}"), dcc.Graph(figure=speed_fig), dcc.Graph(figure=lateral_fig), dcc.Graph(figure=longitudinal_fig), html.Iframe(srcDoc=map_html, width="100%", height="500")

if __name__ == '__main__':
    app.run_server(debug=True, host="0.0.0.0", port=8050)