import logging
import time
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
    TooManyRequests
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
            ConnectionError
        )),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def _fetch_doc_text(self, url):
        """Descarga el contenido de texto de un Google Doc dado su URL."""
        try:
            file_id = get_id_from_url(url)
            # Exportar a texto plano
            response = self.drive_service.files().export(
                fileId=file_id,
                mimeType="text/plain"
            ).execute()
            # La respuesta viene como bytes, decodificamos
            return response.decode('utf-8')
        except Exception as e:
            logger.error(f"Error leyendo prompt desde {url}: {e}")
            raise

    def initialize_session(self, cache_obj=None):
        """
        Inicia el modelo para el caso actual.
        Usamos generate_content() (stateless) en lugar de chat sessions
        para evitar acumulación de historial y conexiones más pesadas.
        """
        try:
            # 1. Obtener Instrucciones del Sistema desde el Doc
            system_instr = self._fetch_doc_text(Config.URL_SYSTEM_INSTRUCTIONS)
            logger.info("Instrucciones del sistema cargadas desde Drive.")

            # 2. Instanciar Modelo (sin start_chat - usamos generate_content directo)
            if cache_obj:
                self.model = vertex_client.get_model_from_cache(cache_obj)
                logger.info("Modelo inicializado con Context Caching.")
            else:
                # Fallback sin caché
                from vertexai.generative_models import GenerativeModel
                self.model = GenerativeModel(
                    "gemini-2.0-flash-001",
                    system_instruction=system_instr
                )
                logger.info("Modelo inicializado SIN caché (Modelo estándar).")

            # NOTA: No usamos start_chat() - cada caso es una llamada generate_content() independiente
            self.chat_session = True  # flag de compatibilidad
            
        except Exception as e:
            logger.error(f"Error inicializando modelo: {e}")
            raise

    def execute_grading_flow(self, patient_files_tuple):
        """
        Ejecuta el flujo de Grading para WAEs usando un único prompt.
        patient_files_tuple es una lista de tuplas (doc_type, path).
        """
        if not self.chat_session:
            raise ValueError("La sesión de chat no ha sido inicializada.")

        try:
            logger.info("Ejecutando prompt único de WAEs...")
            
            # 1. Obtener texto del prompt
            prompt_text = self._fetch_doc_text(Config.URL_PROMPT_WAES)
            
            # Construir la parte inicial del mensaje con identificadores de archivos
            doc_intro = "A continuación te envío los documentos del caso. Ten en cuenta lo siguiente:\n\n"
            message_parts = []
            
            # 2. Adjuntar los PDFs del paciente y armar instrucciones extra
            if patient_files_tuple:
                logger.info(f"Adjuntando {len(patient_files_tuple)} archivos de evidencia al prompt.")
                for doc_type, path in patient_files_tuple:
                    # Mapeo de nombres legibles para el prompt
                    doc_desc = doc_type.replace('_', ' ')
                    if doc_type == 'TRANSCRIPT_INTERVIEW':
                        doc_desc = 'TRANSCRIPT INTERVIEW'
                        
                    doc_intro += f"- El archivo adjunto '{doc_type}.pdf' es la {doc_desc}.\n"
                    
                    with open(path, "rb") as f:
                        pdf_data = f.read()
                    
                    # Adjuntar PDF
                    message_parts.append(Part.from_data(data=pdf_data, mime_type="application/pdf"))

            # Instrucciones de formato extendido
            formatting_rules = (
                "\n\n--- INSTRUCCIONES DE FORMATO OBLIGATORIAS ---\n"
                "1. ANÁLISIS EXHAUSTIVO: El análisis debe ser MUCHO MÁS LARGO y profundo de lo habitual. No resumas ni omitas detalles importantes. Extiéndete todo lo necesario para extraer la información vital de los documentos completos.\n"
                "2. TABLAS REQUERIDAS: Utiliza tablas SIEMPRE que sea posible para estructurar la información (ej. cronologías, comparación de eventos, roles, checklists). Asegúrate de que las tablas usen el formato correcto con pipes (|) y guiones (-).\n"
                "3. MARKDOWN ESTRICTO: Usa doble asterisco para **negritas** y un asterisco o guión para listas (* o -). Evita otros símbolos extraños que rompan el formato.\n"
            )

            # Combinar intro + prompt principal + reglas de formato
            final_prompt_text = f"{doc_intro}\n\nInstrucciones de procesamiento:\n{prompt_text}{formatting_rules}"
            message_parts.insert(0, final_prompt_text)

            # 3. Enviar a Vertex con el timeout corregido
            response = self._send_message_with_timeout(message_parts)
            
            logger.info("Prompt de WAEs completado exitosamente.")
            
            final_markdown = response.text
            
            # Obtener métricas
            usage = response.usage_metadata
            token_counts = {
                "input": usage.prompt_token_count,
                "output": usage.candidates_token_count
            }
            
            # Obtener modelo configurado (Ej: gemini-2.5-flash)
            # Como Vertex no retorna el modelo directamente en cada response de igual manera, 
            # podemos extraerlo del objeto de self.model o enviar el nombre manual
            model_name = getattr(self.model, '_model_name', 'gemini-2.5-flash')
            # Limpiar nombre si viene con prefixes como 'publishers/google/models/'
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
            ConnectionError
        )),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def _send_message_with_timeout(self, content):
        """
        Envía mensaje al chat usando STREAMING para mantener la conexión activa.

        Con stream=True, Vertex AI empieza a enviar tokens inmediatamente en lugar
        de esperar a tener la respuesta completa. Esto evita que firewalls
        corporativos corten la conexión TCP por inactividad.
        """
        import concurrent.futures

        try:
            # generate_content(stream=True): llamada STATELESS, sin historial de chat.
            # Mucho más ligera que chat_session.send_message() ya que no carga historial previo.
            logger.info("📤 Enviando payload a Vertex AI via generate_content() stateless...")
            response_stream = self.model.generate_content(content, stream=True)

            # Acumular todos los chunks con logs en tiempo real
            chunks = []
            CHUNK_TIMEOUT_S = 180  # Si no llega nada en 3 min → red muerta
            import threading

            stop_event = threading.Event()
            stream_error = [None]

            def consume_stream():
                try:
                    logger.info("⏳ Esperando primer token de Vertex AI...")
                    for i, chunk in enumerate(response_stream):
                        chunks.append(chunk)
                        # Log del primer chunk (latencia inicial)
                        if i == 0:
                            logger.info("✅ Primer token recibido. Modelo está respondiendo activamente.")
                        # Log cada 10 chunks para confirmar vida
                        elif i % 10 == 0:
                            total_chars = sum(len(c.text) for c in chunks if hasattr(c,'text') and c.text)
                            logger.info(f"   📝 Streaming en progreso... {i} chunks, ~{total_chars} chars generados.")
                        stop_event.set()  # Resetea el watchdog
                    logger.info(f"✅ Streaming finalizado. Total: {len(chunks)} chunks.")
                except Exception as e:
                    stream_error[0] = e
                    stop_event.set()

            # Usamos un ThreadPoolExecutor para controlar el timeout total
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(consume_stream)
                try:
                    future.result(timeout=Config.API_TIMEOUT_SECONDS)
                except concurrent.futures.TimeoutError:
                    logger.error(
                        f"⛔ TIMEOUT ({Config.API_TIMEOUT_SECONDS}s) alcanzado. "
                        f"Se recibieron {len(chunks)} chunks antes del corte. "
                        "Esto indica un problema de red/firewall o una respuesta excesivamente larga."
                    )
                    raise TimeoutError("Timeout alcanzado esperando al modelo.")

            if stream_error[0]:
                raise stream_error[0]

            if not chunks:
                raise ValueError("Vertex AI devolvió una respuesta vacía (0 chunks).")

            # El último chunk tiene usage_metadata completo.
            last_chunk = chunks[-1]

            # Construir texto completo concatenando todos los chunks.
            # Algunos chunks pueden no tener texto (ej: chunk final con solo metadata).
            full_text = "".join(c.text for c in chunks if hasattr(c, 'text') and c.text)

            # Si no hay texto, puede ser un bloqueo por safety filters.
            if not full_text:
                # Intentar extraer información de diagnóstico del último chunk
                finish_reason = getattr(last_chunk.candidates[0], 'finish_reason', 'UNKNOWN') if hasattr(last_chunk, 'candidates') and last_chunk.candidates else 'NO_CANDIDATES'
                safety_ratings = getattr(last_chunk.candidates[0], 'safety_ratings', []) if hasattr(last_chunk, 'candidates') and last_chunk.candidates else []
                logger.error(f"Vertex AI devolvió respuesta sin texto. finish_reason={finish_reason}, safety_ratings={safety_ratings}")
                raise ValueError(f"Vertex AI bloqueó la respuesta. finish_reason={finish_reason}")

            # Sobreescribir el texto del último chunk con el texto completo
            # para que response.text devuelva la respuesta entera.
            # Usamos un wrapper simple para no depender de internals de la librería.
            class StreamResponse:
                def __init__(self, text, usage_metadata):
                    self.text = text
                    self.usage_metadata = usage_metadata

            logger.debug(f"Streaming completado. {len(chunks)} chunks, {len(full_text)} chars.")
            return StreamResponse(
                text=full_text,
                usage_metadata=last_chunk.usage_metadata
            )

        except Exception as e:
            raise e

# Instancia global (ESTA LÍNEA FALTABA)
chat_service = ChatService()