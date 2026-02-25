import io
import logging
import docx
from googleapiclient.http import MediaIoBaseDownload  # <-- CORREGIDO AQUÍ
from googleapiclient.errors import HttpError
from fpdf import FPDF
from src.core.google_client import google_manager
from src.config import Config
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
from google.api_core.exceptions import (
    GoogleAPIError,
    InternalServerError,
    ServiceUnavailable
)

logger = logging.getLogger(__name__)

class DriveService:
    """
    Servicio encargado de descargar y normalizar archivos de Drive a PDF.
    Cumple la regla: Todo input debe convertirse a PDF para el contexto de Vertex.
    """
    
    def __init__(self):
        self.service = google_manager.get_drive_service()

    @retry(
        stop=stop_after_attempt(Config.MAX_RETRIES),
        wait=wait_exponential(
            multiplier=1,
            min=Config.RETRY_MIN_WAIT,
            max=Config.RETRY_MAX_WAIT
        ),
        retry=retry_if_exception_type((
            HttpError,
            GoogleAPIError,
            TimeoutError,
            ConnectionError
        )),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def get_file_metadata(self, file_id):
        """Obtiene el nombre y tipo MIME de un archivo."""
        try:
            file = self.service.files().get(
                fileId=file_id, 
                fields="id, name, mimeType"
            ).execute()
            return file
        except HttpError as e:
            logger.error(f"Error obteniendo metadata para {file_id}: {e}")
            raise

    @retry(
        stop=stop_after_attempt(Config.MAX_RETRIES),
        wait=wait_exponential(
            multiplier=1,
            min=Config.RETRY_MIN_WAIT,
            max=Config.RETRY_MAX_WAIT
        ),
        retry=retry_if_exception_type((
            HttpError,
            GoogleAPIError,
            TimeoutError,
            ConnectionError
        )),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def download_as_pdf(self, file_id, output_path):
        """
        Descarga un archivo y lo guarda como PDF en output_path.
        Maneja conversión automática según el tipo de archivo.
        """
        try:
            meta = self.get_file_metadata(file_id)
            mime_type = meta.get('mimeType')
            name = meta.get('name')
            
            logger.info(f"Procesando archivo: {name} ({mime_type})")

            # CASO 1: Google Docs (Nativos) -> Exportar a PDF
            if mime_type == 'application/vnd.google-apps.document':
                request = self.service.files().export_media(
                    fileId=file_id,
                    mimeType='application/pdf'
                )
                self._execute_download(request, output_path)

            # CASO 2: PDF Real -> Descargar directo
            elif mime_type == 'application/pdf':
                request = self.service.files().get_media(fileId=file_id)
                self._execute_download(request, output_path)

            # CASO 3: Word (.docx) -> Descargar binario -> Convertir a PDF
            elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                temp_docx = io.BytesIO()
                request = self.service.files().get_media(fileId=file_id)
                downloader = MediaIoBaseDownload(temp_docx, request)
                
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                
                temp_docx.seek(0)
                text_content = self._extract_text_from_docx(temp_docx)
                self._create_pdf_from_text(text_content, output_path)

            # CASO 4: Texto Plano -> Descargar -> Convertir a PDF
            elif mime_type == 'text/plain':
                request = self.service.files().get_media(fileId=file_id)
                temp_txt = io.BytesIO()
                downloader = MediaIoBaseDownload(temp_txt, request)
                
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                
                temp_txt.seek(0)
                text_content = temp_txt.read().decode('utf-8', errors='replace')
                self._create_pdf_from_text(text_content, output_path)

            else:
                logger.warning(f"Tipo no soportado nativamente: {mime_type}. Intentando descarga binaria...")
                # Fallback: intentar descargar tal cual
                request = self.service.files().get_media(fileId=file_id)
                self._execute_download(request, output_path)

            logger.info(f"Archivo guardado exitosamente: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Fallo al procesar {file_id}: {e}")
            raise

    def _execute_download(self, request, output_path):
        """Ejecuta la descarga estándar de Drive API."""
        with open(output_path, 'wb') as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()

    def _extract_text_from_docx(self, docx_io):
        """Extrae texto plano de un archivo Word en memoria."""
        doc = docx.Document(docx_io)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)

    def _create_pdf_from_text(self, text, output_path):
        """Genera un PDF simple a partir de texto plano."""
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=11)
        
        # Saneamiento básico para FPDF (Latin-1)
        sanitized_text = text.encode('latin-1', 'replace').decode('latin-1')
        
        pdf.multi_cell(0, 10, sanitized_text)
        pdf.output(output_path)

# Instancia global
drive_service = DriveService()