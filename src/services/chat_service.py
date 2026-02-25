import logging
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
from google.api_core.exceptions import (
    GoogleAPIError,
    InternalServerError,
    ServiceUnavailable,
    TooManyRequests,
    DeadlineExceeded
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

    @retry(
        stop=stop_after_attempt(Config.MAX_RETRIES),
        wait=wait_exponential(
            multiplier=1,
            min=Config.RETRY_MIN_WAIT,
            max=Config.RETRY_MAX_WAIT
        ),
        retry=retry_if_exception_type((
            GoogleAPIError,
            TimeoutError,
            ConnectionError,
            Exception
        )),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
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
            logger.info("Instrucciones del sistema cargadas desde Drive.")

            if cache_obj:
                self.model = vertex_client.get_model_from_cache(cache_obj)
                logger.info("Modelo inicializado con Context Caching.")
            else:
                from vertexai.generative_models import GenerativeModel
                self.model = GenerativeModel(
                    "gemini-2.0-flash-001",
                    system_instruction=system_instr
                )
                logger.info("Modelo inicializado SIN caché (Modelo estándar).")

            self.chat_session = True  
            
        except Exception as e:
            logger.error(f"Error inicializando modelo: {e}")
            raise

    def execute_grading_flow(self, patient_files_tuple):
        """Ejecuta el flujo de Grading para WAEs usando un único prompt."""
        if not self.chat_session:
            raise ValueError("La sesión de chat no ha sido inicializada.")

        try:
            logger.info("Ejecutando prompt único de WAEs...")
            
            prompt_text = self._fetch_doc_text(Config.URL_PROMPT_WAES)
            doc_intro = "A continuación te envío los documentos del caso. Ten en cuenta lo siguiente:\n\n"
            message_parts = []
            
            if patient_files_tuple:
                logger.info(f"Adjuntando {len(patient_files_tuple)} archivos de evidencia al prompt.")
                for doc_type, path in patient_files_tuple:
                    doc_desc = doc_type.replace('_', ' ')
                    if doc_type == 'TRANSCRIPT_INTERVIEW':
                        doc_desc = 'TRANSCRIPT INTERVIEW'
                        
                    doc_intro += f"- El archivo adjunto '{doc_type}.pdf' es la {doc_desc}.\n"
                    
                    with open(path, "rb") as f:
                        pdf_data = f.read()
                    
                    message_parts.append(Part.from_data(data=pdf_data, mime_type="application/pdf"))

            formatting_rules = (
                "\n\n--- INSTRUCCIONES DE FORMATO OBLIGATORIAS ---\n"
                "1. ANÁLISIS EXHAUSTIVO: El análisis debe ser MUCHO MÁS LARGO y profundo de lo habitual. No resumas ni omitas detalles importantes. Extiéndete todo lo necesario para extraer la información vital de los documentos completos.\n"
                "2. TABLAS REQUERIDAS: Utiliza tablas SIEMPRE que sea posible para estructurar la información (ej. cronologías, comparación de eventos, roles, checklists). Asegúrate de que las tablas usen el formato correcto con pipes (|) y guiones (-).\n"
                "3. MARKDOWN ESTRICTO: Usa doble asterisco para **negritas** y un asterisco o guión para listas (* o -). Evita otros símbolos extraños que rompan el formato.\n"
            )

            final_prompt_text = f"{doc_intro}\n\nInstrucciones de procesamiento:\n{prompt_text}{formatting_rules}"
            message_parts.insert(0, final_prompt_text)

            response = self._send_message_with_timeout(message_parts)
            
            logger.info("Prompt de WAEs completado exitosamente.")
            
            final_markdown = response.text
            usage = response.usage_metadata
            token_counts = {
                "input": usage.prompt_token_count,
                "output": usage.candidates_token_count
            }
            
            model_name = getattr(self.model, '_model_name', 'gemini-2.5-flash')
            if 'models/' in model_name:
                model_name = model_name.split('models/')[-1]

            return final_markdown, token_counts, model_name

        except Exception as e:
            logger.error(f"Error crítico en el flujo de grading de WAEs: {e}")
            raise

    @retry(
        stop=stop_after_attempt(Config.MAX_RETRIES),
        wait=wait_exponential(
            multiplier=1,
            min=Config.RETRY_MIN_WAIT,
            max=Config.RETRY_MAX_WAIT
        ),
        retry=retry_if_exception_type((
            GoogleAPIError,
            InternalServerError,
            ServiceUnavailable,
            TooManyRequests,
            TimeoutError,
            ConnectionError,
            DeadlineExceeded
        )),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def _send_message_with_timeout(self, content):
        """
        Envía mensaje utilizando la API nativa sin streaming.
        Delega el control del socket directamente a la librería de Google.
        """
        try:
            logger.info("📤 Enviando payload a Vertex AI (esperando respuesta completa en bloque)...")
            
            # API nativa: stream=False (Eliminamos request_options que daba error)
            response = self.model.generate_content(
                content,
                stream=False
            )

            if not response or not response.text:
                finish_reason = getattr(response.candidates[0], 'finish_reason', 'UNKNOWN') if response and hasattr(response, 'candidates') and response.candidates else 'NO_CANDIDATES'
                raise ValueError(f"Vertex AI bloqueó la respuesta (ej. filtro de seguridad). finish_reason={finish_reason}")

            logger.debug("Procesamiento completado. Respuesta recibida correctamente.")
            return response

        except Exception as e:
            logger.error(f"❌ Error en la comunicación nativa con Vertex AI: {e}")
            raise

# Instancia global
chat_service = ChatService()