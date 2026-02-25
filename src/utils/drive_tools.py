import io
import re
import logging
import unicodedata
from pathlib import Path
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from googleapiclient.errors import HttpError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from src.core.google_client import google_manager

logger = logging.getLogger(__name__)

# --- NUEVA FUNCIÓN AGREGADA ---
def get_id_from_url(url: str) -> str:
    """
    Extrae el ID de archivo de una URL de Google Drive o Docs.
    Soporta formatos estándar y IDs directos.
    """
    if not url:
        return None
    
    # Patrón común para docs, sheets, drive file links
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    
    # Patrón para carpetas de drive
    match_folder = re.search(r'folders/([a-zA-Z0-9_-]+)', url)
    if match_folder:
        return match_folder.group(1)

    # Si parece un ID directo (largo y sin barras), lo devolvemos tal cual
    if len(url) > 20 and '/' not in url:
        return url
        
    return None
# ------------------------------

def normalize_name(name):
    """Elimina acentos y convierte a minúsculas para búsquedas flexibles."""
    if not name: return ""
    n = unicodedata.normalize('NFD', name).encode('ascii', 'ignore').decode("utf-8")
    return n.lower().strip()
    
@retry(
    retry=retry_if_exception_type(HttpError),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5)
)
def find_subfolder(parent_id: str, target_names: list) -> tuple[str, str]:
    """Busca una carpeta hija con soporte para Drives Compartidos."""
    drive_service = google_manager.get_drive_service()
    query = f"'{parent_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    
    try:
        results = []
        page_token = None
        while True:
            response = drive_service.files().list(
                q=query, 
                fields='nextPageToken, files(id, name)', 
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True
            ).execute()
            results.extend(response.get('files', []))
            page_token = response.get('nextPageToken', None)
            if not page_token: break
        
        targets_norm = [normalize_name(t) for t in target_names]
        for folder in results:
            folder_norm = normalize_name(folder['name'])
            if any(t in folder_norm for t in targets_norm):
                logger.info(f"Carpeta encontrada: '{folder['name']}' (ID: {folder['id']})")
                return folder['id'], folder['name']
                
    except HttpError as e:
        if e.resp.status in [403, 429]:
            logger.warning(f"Límite de cuota Drive (find_subfolder). Reintentando... {e}")
            raise e 
        else:
            logger.error(f"Error buscando subcarpeta: {e}")
    except Exception as e:
        logger.error(f"Error genérico buscando carpeta: {e}")
    
    return None, None

@retry(
    retry=retry_if_exception_type(HttpError),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5)
)
def get_google_doc_content(doc_url: str) -> str:
    """Extrae el texto de un Google Doc."""
    file_id = get_id_from_url(doc_url) # Ahora usamos la función helper
    if not file_id:
        logger.warning(f"URL inválida o ID no encontrado: {doc_url}")
        return ""
    
    drive_service = google_manager.get_drive_service()
    
    try:
        request = drive_service.files().export_media(fileId=file_id, mimeType='text/plain')
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done: _, done = downloader.next_chunk()
        
        return fh.getvalue().decode('utf-8-sig')
        
    except Exception as e:
        logger.error(f"Error leyendo Doc ({file_id}): {e}")
        return ""

@retry(
    retry=retry_if_exception_type(HttpError),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3)
)
def list_files_in_folder(folder_id: str) -> list:
    """Retorna lista de archivos."""
    drive_service = google_manager.get_drive_service()
    try:
        res = drive_service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="files(id, name, mimeType)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            pageSize=1000
        ).execute()
        return res.get('files', [])
    except HttpError as e:
        logger.warning(f"Error listando archivos en {folder_id}: {e}")
        raise e
    except Exception as e:
        logger.error(f"Error genérico listando archivos: {e}")
        return []

@retry(
    retry=retry_if_exception_type(HttpError),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(3)
)
def upload_file_to_drive(local_path: str, parent_folder_id: str) -> str:
    """Sube archivo a Drive."""
    drive_service = google_manager.get_drive_service()
    path_obj = Path(local_path)
    
    if not path_obj.exists():
        logger.error(f"No se encuentra el archivo local: {local_path}")
        return None

    file_metadata = {
        'name': path_obj.name,
        'parents': [parent_folder_id]
    }
    
    media = MediaFileUpload(str(path_obj), resumable=True)

    try:
        logger.info(f"Subiendo {path_obj.name} a Drive...")
        file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, webViewLink',
            supportsAllDrives=True
        ).execute()
        return file.get('webViewLink')

    except Exception as e:
        logger.error(f"Error subiendo archivo: {e}")
        return None

@retry(
    retry=retry_if_exception_type(HttpError),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(3)
)
def create_folder_in_drive(folder_name: str, parent_folder_id: str) -> tuple[str, str]:
    """Crea carpeta en Drive."""
    drive_service = google_manager.get_drive_service()
    file_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [parent_folder_id]
    }
    try:
        folder = drive_service.files().create(
            body=file_metadata,
            fields='id, webViewLink',
            supportsAllDrives=True
        ).execute()
        return folder.get('id'), folder.get('webViewLink')
    except Exception as e:
        logger.error(f"Error creando carpeta: {e}")
        raise e

def download_file_from_drive(file_id: str, local_path: Path) -> bool:
    """Descarga archivo simple."""
    drive_service = google_manager.get_drive_service()
    try:
        request = drive_service.files().get_media(fileId=file_id)
        with open(local_path, 'wb') as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done: _, done = downloader.next_chunk()
        return True
    except Exception as e:
        logger.error(f"Error descargando {file_id}: {e}")
        return False