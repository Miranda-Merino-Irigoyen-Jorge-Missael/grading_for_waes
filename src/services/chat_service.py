import logging
import concurrent.futures
import sys
import traceback
from vertexai.generative_models import Part
from src.config import Config
from src.core.google_client import google_manager
from src.core.vertex_wrapper import vertex_client
from src.utils.drive_tools import get_id_from_url
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

logger = logging.getLogger(__name__)

class ChatService:
    """
    Gestiona la sesión de chat con Vertex AI y la ejecución secuencial de prompts.
    """

    def __init__(self):
        self.drive_service = google_manager.get_drive_service()
        self.model = None
        self.chat_session = None

    def _fetch_doc_text(self, url):
        """Descarga el contenido de texto de un Google Doc dado su URL."""
        try:
            file_id = get_id_from_url(url)
            response = self.drive_service.files().export(
                fileId=file_id,
                mimeType="text/plain"
            ).execute()
            return response.decode('utf-8')
        except Exception as e:
            logger.error(f"Error leyendo prompt desde {url}: {e}")
            raise

    def initialize_session(self, cache_obj=None):
        """Inicia el modelo para el caso actual."""
        try:
            system_instr = self._fetch_doc_text(Config.URL_SYSTEM_INSTRUCTIONS)
            logger.info("Instrucciones del sistema cargadas.")

            if cache_obj:
                self.model = vertex_client.get_model_from_cache(cache_obj)
                logger.info("Modelo inicializado con Context Caching.")
            else:
                from vertexai.generative_models import GenerativeModel
                self.model = GenerativeModel(
                    "gemini-2.0-flash-001",
                    system_instruction=system_instr
                )
                logger.info("Modelo inicializado SIN caché.")

            self.chat_session = True  
            
        except Exception as e:
            logger.error(f"Error inicializando modelo: {e}")
            raise

    def execute_grading_flow(self, patient_files_tuple):
        """Ejecuta el flujo de Grading con reintentos configurados."""
        if not self.chat_session:
            raise ValueError("La sesión de chat no ha sido inicializada.")
        
        return self._send_with_retry(patient_files_tuple)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        retry=retry_if_exception_type((Exception,)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def _send_with_retry(self, patient_files_tuple):
        """Lógica de envío que será reintentada por Tenacity si hay error o timeout."""
        prompt_text = self._fetch_doc_text(Config.URL_PROMPT_WAES)
        doc_intro = "A continuación te envío los documentos del caso:\n\n"
        message_parts = []
        
        for doc_type, path in patient_files_tuple:
            doc_intro += f"- El archivo adjunto '{doc_type}.pdf'\n"
            with open(path, "rb") as f:
                pdf_data = f.read()
            message_parts.append(Part.from_data(data=pdf_data, mime_type="application/pdf"))

        formatting_rules = (
            "\n\n--- INSTRUCCIONES DE FORMATO OBLIGATORIAS ---\n"
            "1. ANÁLISIS EXHAUSTIVO Y PROFUNDO.\n"
            "2. TABLAS REQUERIDAS.\n"
        )

        final_prompt_text = f"{doc_intro}\n\nInstrucciones:\n{prompt_text}{formatting_rules}"
        message_parts.insert(0, final_prompt_text)

        return self._raw_send_to_vertex(message_parts)

    def _raw_send_to_vertex(self, content):
        """Envío directo con timeout estricto y log de error técnico."""
        timeout_val = Config.API_TIMEOUT_SECONDS
        logger.info(f"📤 Enviando payload a Vertex AI (Timeout: {timeout_val}s)...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.model.generate_content, content, stream=False)
            try:
                response = future.result(timeout=timeout_val)
                
                if not response or not response.text:
                    raise ValueError("Vertex AI devolvió una respuesta vacía o bloqueada.")

                logger.info("✅ Respuesta recibida exitosamente.")
                
                final_markdown = response.text
                usage = response.usage_metadata
                token_counts = {
                    "input": usage.prompt_token_count,
                    "output": usage.candidates_token_count
                }
                
                return final_markdown, token_counts, "gemini-2.0-flash-001"

            except concurrent.futures.TimeoutError:
                logger.error(f"⏰ ¡TIMEOUT! No hubo respuesta en {timeout_val}s.")
                raise Exception("TIMEOUT_INTERNAL_ERROR")
            except Exception as e:
                # CAPTURA TÉCNICA DEL ERROR REAL
                print("\n" + "!"*60)
                print("DETALLE TÉCNICO DEL ERROR (STACK TRACE):")
                traceback.print_exc()
                print("!"*60 + "\n")
                raise e

# INSTANCIA GLOBAL (Vital para evitar ImportError)
chat_service = ChatService()