from functionalities.requests import insert_df_to_table, create_table_if_dont_exist
import os
import pandas as pd
import exifread
from PIL import Image
from datetime import datetime
from fractions import Fraction
from ftplib import FTP
from dotenv import load_dotenv
import uuid

# --- Carga .env (estilo limpio) ---
load_dotenv()

# Vars de entorno
INPUT_FOLDER    = os.getenv("INPUT_FOLDER", "image_data")
CLIENTE_DEFAULT = os.getenv("CLIENTE_DEFAULT", "SOUTH32")

FTP_HOST        = os.getenv("FTP_HOST")
FTP_USER        = os.getenv("FTP_USER")
FTP_PASSWORD    = os.getenv("FTP_PASSWORD")
FTP_PORT        = int(os.getenv("FTP_PORT", "21"))
FTP_REMOTE_DIR  = os.getenv("FTP_REMOTE_DIR", "fotosvbox")

TARGET_SIZE_KB  = int(os.getenv("TARGET_SIZE_KB", "3000"))
IMAGE_FORMAT    = os.getenv("IMAGE_FORMAT", "JPEG")

TABLE_NAME = "poligonos_fotos"
TABLE_COLUMNS = [
    "filename", "width", "height", "upload_date",
    "latitude", "longitude", "date_taken", "camera_make", "camera_model",
    "exposure_time", "f_number", "iso", "cliente"
]

# ---------- utilidades ----------
def upload_to_ftp(local_path: str, remote_dir: str = FTP_REMOTE_DIR) -> str:
    """Sube un archivo al FTP y devuelve ruta remota (remote_dir/filename)."""
    filename = os.path.basename(local_path)
    remote_path = f"{filename}"

    ftp = FTP()
    ftp.connect(FTP_HOST, FTP_PORT, timeout=30)
    ftp.login(FTP_USER, FTP_PASSWORD)

    with open(local_path, "rb") as f:
        ftp.storbinary(f"STOR {filename}", f)

    ftp.quit()
    return F"{remote_dir}/{remote_path}"

def get_decimal_coords(tags):
    """Convierte EXIF a (lat, lon) decimales o (None, None)."""
    try:
        def convert_to_decimal(value):
            return float(value[0]) + float(value[1]) / 60 + float(value[2]) / 3600

        gps_latitude = tags["GPS GPSLatitude"].values
        gps_latitude_ref = tags["GPS GPSLatitudeRef"].values
        gps_longitude = tags["GPS GPSLongitude"].values
        gps_longitude_ref = tags["GPS GPSLongitudeRef"].values

        from fractions import Fraction
        lat = convert_to_decimal([Fraction(x) for x in gps_latitude])
        lon = convert_to_decimal([Fraction(x) for x in gps_longitude])

        if gps_latitude_ref != "N":
            lat = -lat
        if gps_longitude_ref != "E":
            lon = -lon

        return round(lat, 8), round(lon, 8)
    except KeyError:
        return None, None

def get_exif_data(image_path: str):
    with open(image_path, "rb") as f:
        tags = exifread.process_file(f, details=False)

    lat, lon = get_decimal_coords(tags)

    raw_date_taken = str(tags.get("EXIF DateTimeOriginal", "Unknown"))
    try:
        date_taken = datetime.strptime(raw_date_taken, "%Y:%m:%d %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        date_taken = "Unknown"

    camera_model = str(tags.get("Image Model", "Unknown"))
    camera_make = str(tags.get("Image Make", "Unknown"))
    exposure_time = str(tags.get("EXIF ExposureTime", "Unknown"))
    f_number = str(tags.get("EXIF FNumber", "Unknown"))
    iso = str(tags.get("EXIF ISOSpeedRatings", "Unknown"))

    return lat, lon, date_taken, camera_make, camera_model, exposure_time, f_number, iso

def compress_image(image_path: str, unique_id: str,
                   target_size_kb: int = TARGET_SIZE_KB,
                   format: str = IMAGE_FORMAT) -> str:
    """Comprime a tamaño objetivo sin cambiar dimensiones. Retorna ruta nueva."""
    img = Image.open(image_path)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    quality = 95
    temp_path = f"{unique_id}_{os.path.basename(image_path)}"

    while True:
        img.save(
            temp_path, format=format, quality=quality, optimize=True,
            progressive=(format.upper() == "JPEG")
        )
        file_size_kb = os.path.getsize(temp_path) / 1024
        if file_size_kb <= target_size_kb or quality < 50:
            break
        quality -= 5

    return temp_path

# ---------- flujo principal (inserta por imagen) ----------
def upload_images():
    if not os.path.isdir(INPUT_FOLDER):
        print(f"Carpeta no encontrada: {INPUT_FOLDER}")
        return

    # Crear tabla una sola vez
    create_table_if_dont_exist(TABLE_NAME, TABLE_COLUMNS)

    processed = 0
    errors = 0

    for filename in os.listdir(INPUT_FOLDER):
        if not filename.lower().endswith((".jpg", ".jpeg")):
            continue

        try:
            file_path = os.path.join(INPUT_FOLDER, filename)
            unique_id = str(uuid.uuid1())[:8]

            # 1) Comprimir
            compressed_path = compress_image(file_path, unique_id)

            # 2) Renombrar para remoto (id + nombre original)
            remote_basename = f"{unique_id}_{filename}"
            final_local_path = remote_basename
            os.replace(compressed_path, final_local_path)

            # 3) Subir al FTP
            remote_path = upload_to_ftp(final_local_path)

            # 4) Limpiar archivo temporal local
            try:
                os.remove(final_local_path)
            except Exception:
                pass

            # 5) EXIF + dimensiones
            lat, lon, date_taken, camera_make, camera_model, exposure_time, f_number, iso = get_exif_data(file_path)
            width, height = Image.open(file_path).size
            upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 6) Insertar inmediatamente esa fila en DB
            row = [[
                remote_path, width, height, upload_time, lat, lon, date_taken,
                camera_make, camera_model, exposure_time, f_number, iso, CLIENTE_DEFAULT
            ]]
            df_row = pd.DataFrame(row, columns=TABLE_COLUMNS)
            insert_df_to_table(df_row, TABLE_NAME)

            processed += 1
            print(f"✓ Guardado en DB y FTP: {remote_path}")

        except Exception as e:
            errors += 1
            print(f"✗ Error procesando {filename}: {e}")

    print(f"Finalizado. OK: {processed} | Errores: {errors}")
