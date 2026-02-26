import logging
import concurrent.futures
import sys
import traceback
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
    Gestiona la sesión de chat y la ejecución secuencial de prompts.
    Adaptado para soportar tanto Vertex AI como Gemini API Directa.
    """

    def __init__(self):
        self.drive_service = google_manager.get_drive_service()
        self.model = None
        self.chat_session = None
        self.uploaded_files = [] # Rastrear archivos subidos a Gemini API para borrarlos al terminar
        self.safety_settings = None # Guardaremos los ajustes de seguridad según el entorno

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
        """Inicia el modelo para el caso actual y apaga filtros de seguridad."""
        try:
            system_instr = self._fetch_doc_text(Config.URL_SYSTEM_INSTRUCTIONS)
            logger.info("Instrucciones del sistema cargadas.")

            if Config.USE_VERTEX_AI:
                from vertexai.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold
                self.safety_settings = {
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                }
                
                if cache_obj:
                    self.model = vertex_client.get_model_from_cache(cache_obj)
                    logger.info("Modelo Vertex inicializado con Context Caching.")
                else:
                    self.model = GenerativeModel(
                        "gemini-2.5-flash",
                        system_instruction=system_instr
                    )
                    logger.info("Modelo Vertex inicializado SIN caché.")
            else:
                import google.generativeai as genai
                self.safety_settings = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]
                
                if cache_obj:
                    self.model = vertex_client.get_model_from_cache(cache_obj)
                    logger.info("Modelo Gemini API inicializado con Context Caching.")
                else:
                    self.model = genai.GenerativeModel(
                        model_name="gemini-2.5-flash",
                        system_instruction=system_instr
                    )
                    logger.info("Modelo Gemini API inicializado SIN caché.")

            self.chat_session = True  
            
        except Exception as e:
            logger.error(f"Error inicializando modelo: {e}")
            raise

    def execute_grading_flow(self, patient_files_tuple):
        """Ejecuta el flujo de Grading con manejo de archivos y reintentos."""
        if not self.chat_session:
            raise ValueError("La sesión de chat no ha sido inicializada.")
        
        uploaded_parts = []
        
        try:
            # Preparar los documentos según la plataforma elegida
            if Config.USE_VERTEX_AI:
                from vertexai.generative_models import Part
                for doc_type, path in patient_files_tuple:
                    with open(path, "rb") as f:
                        data = f.read()
                    uploaded_parts.append((doc_type, Part.from_data(data=data, mime_type="application/pdf")))
            else:
                import google.generativeai as genai
                for doc_type, path in patient_files_tuple:
                    logger.info(f"Subiendo {doc_type} a Gemini File API...")
                    gemini_file = genai.upload_file(path=path, mime_type="application/pdf")
                    uploaded_parts.append((doc_type, gemini_file))
                    self.uploaded_files.append(gemini_file)

            # Ejecutar el chat con reintentos
            return self._send_with_retry(uploaded_parts)
            
        finally:
            # Siempre limpiar los archivos subidos a la API directa para no consumir cuota de almacenamiento
            self._cleanup_gemini_files()

    def _cleanup_gemini_files(self):
        if not Config.USE_VERTEX_AI and self.uploaded_files:
            import google.generativeai as genai
            for f in self.uploaded_files:
                try:
                    genai.delete_file(f.name)
                    logger.info(f"Archivo temporal {f.name} eliminado de la nube de Gemini.")
                except Exception as e:
                    logger.warning(f"No se pudo limpiar archivo en Gemini API: {e}")
            self.uploaded_files = []

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        retry=retry_if_exception_type((Exception,)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def _send_with_retry(self, uploaded_parts):
        """Lógica de envío que será reintentada por Tenacity si hay error o timeout."""
        prompt_text = self._fetch_doc_text(Config.URL_PROMPT_WAES)
        doc_intro = "A continuación te envío los documentos del caso:\n\n"
        message_parts = []
        
        for doc_type, part_obj in uploaded_parts:
            doc_intro += f"- El archivo adjunto '{doc_type}.pdf'\n"
            message_parts.append(part_obj)

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
        logger.info(f"📤 Enviando payload a la IA (Timeout: {timeout_val}s)...")
        
        # --- CONFIGURACIÓN DE GENERACIÓN ---
        # Temperatura de 0.5 y máximo de salida permitido por el modelo (8192 tokens)
        gen_config = {
            "temperature": 0.5,
            "max_output_tokens": 30000,
        }
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self.model.generate_content, 
                content, 
                stream=False,
                generation_config=gen_config,
                safety_settings=self.safety_settings
            )
            try:
                response = future.result(timeout=timeout_val)
                
                if not response or not response.text:
                    raise ValueError("La IA devolvió una respuesta vacía o bloqueada.")

                logger.info("✅ Respuesta recibida exitosamente.")
                
                final_markdown = response.text
                usage = response.usage_metadata
                
                # Acceso seguro a los tokens
                token_counts = {
                    "input": getattr(usage, 'prompt_token_count', 0),
                    "output": getattr(usage, 'candidates_token_count', 0)
                }
                
                return final_markdown, token_counts, "gemini-2.5-flash"

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

# INSTANCIA GLOBAL
chat_service = ChatService()