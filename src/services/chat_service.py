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
        self.uploaded_files = [] 
        self.safety_settings = None 

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
            
            # FASE 1: RE-ENCUADRE HUMANITARIO Y CLÍNICO (ESCUDO LEGAL)
            system_instr += (
                "\n\nINSTRUCCIÓN CRÍTICA DE IDENTIDAD Y PROPÓSITO: Eres un Especialista Legal en Derechos Humanos y Protección a Víctimas. "
                "Tu trabajo es vital para que las víctimas de abuso obtengan protección legal, mejoren su estilo de vida y puedan estar más seguros. "
                "Un análisis superficial o resumido pone en riesgo su caso y su integridad. "
                "Por lo tanto, tu reporte final DEBE ser extremadamente exhaustivo (equivalente a 7 a 10 páginas). "
                "METODOLOGÍA OBLIGATORIA: "
                "1. TONO: Utiliza un tono estrictamente clínico, forense y legal. Está estrictamente prohibido usar lenguaje dramático, sensacionalista o narrativo. Limítate a la exposición técnica de los hechos. "
                "2. PROFUNDIDAD: Redacta al menos 3 párrafos de análisis profundo por cada factor legal de VAWA o Visa T. "
                "3. LÍNEA DE TIEMPO: Analiza la línea del tiempo evento por evento sin omitir ninguna fecha mencionada en los documentos. "
                "4. INTEGRIDAD: Asegúrate de TERMINAR COMPLETAMENTE tu análisis hasta la última sección. No dejes ideas a medias."
            )
            logger.info("Instrucciones del sistema cargadas con el Escudo Legal y Humanitario (Fase 1).")

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

            return self._send_with_retry(uploaded_parts)
            
        finally:
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
        doc_intro = "A continuación te envío los documentos clínicos y forenses del caso:\n\n"
        message_parts = []
        
        for doc_type, part_obj in uploaded_parts:
            doc_intro += f"- El archivo adjunto '{doc_type}.pdf'\n"
            message_parts.append(part_obj)

        formatting_rules = (
            "\n\n--- INSTRUCCIONES ESTRICTAS DE FORMATO Y EXTENSIÓN OBLIGATORIAS ---\n"
            "1. **PROHIBIDO RESUMIR:** Tu respuesta debe abarcar de 7 a 10 páginas para garantizar la protección legal del individuo.\n"
            "2. **ESTRUCTURA DE TABLAS:** Las tablas solicitadas NO deben tener viñetas cortas. Cada celda de la tabla DEBE contener un párrafo descriptivo completo y detallado en tono forense.\n"
            "3. **EVIDENCIA EXPLÍCITA:** Por cada afirmación que hagas, debes incluir una cita textual entre comillas extraída de los documentos proporcionados.\n"
            "4. **PROFUNDIDAD DEL ANÁLISIS:** Desarrolla tu análisis de forma técnica hasta agotar la información, desglosa cada incidente del Rapsheet y del Transcript de forma individual y meticulosa.\n"
        )

        final_prompt_text = f"{doc_intro}\n\nInstrucciones de Grading (Análisis Forense y Legal):\n{prompt_text}{formatting_rules}"
        message_parts.insert(0, final_prompt_text)

        return self._raw_send_to_vertex(message_parts)

    def _raw_send_to_vertex(self, content):
        """Envío con STREAMING para evitar cortes de conexión en respuestas largas."""
        timeout_val = Config.API_TIMEOUT_SECONDS
        logger.info(f"📤 Enviando payload a la IA (Modo STREAMING) (Timeout total: {timeout_val}s)...")
        
        gen_config = {
            "temperature": 0.3, 
            "max_output_tokens": 20000, 
        }

        def stream_generator():
            # STREAM=TRUE ES LA CLAVE PARA EVITAR QUE SE CORTE A LA MITAD Y ATRAPAR ERRORES
            response_stream = self.model.generate_content(
                content, 
                stream=True, 
                generation_config=gen_config,
                safety_settings=self.safety_settings
            )
            
            chunks = []
            finish_reason = "UNKNOWN"
            in_tokens, out_tokens = 0, 0
            
            # Ensamblamos los pedazos conforme llegan de la API
            for chunk in response_stream:
                try:
                    if chunk.text:
                        chunks.append(chunk.text)
                except ValueError:
                    logger.warning("⚠️ Un fragmento fue bloqueado por un filtro interno de seguridad.")
                
                # Intentar leer por qué se detuvo y la cantidad de tokens
                try:
                    if chunk.candidates and chunk.candidates[0].finish_reason:
                        finish_reason = str(chunk.candidates[0].finish_reason)
                except:
                    pass
                
                try:
                    if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                        if getattr(chunk.usage_metadata, 'prompt_token_count', 0) > 0:
                            in_tokens = chunk.usage_metadata.prompt_token_count
                        if getattr(chunk.usage_metadata, 'candidates_token_count', 0) > 0:
                            out_tokens = chunk.usage_metadata.candidates_token_count
                except:
                    pass

            return "".join(chunks), finish_reason, in_tokens, out_tokens

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(stream_generator)
            try:
                final_markdown, finish_reason, input_tokens, output_tokens = future.result(timeout=timeout_val)
                
                if not final_markdown:
                    raise ValueError("La IA devolvió una respuesta vacía o totalmente bloqueada.")

                logger.info(f"✅ Respuesta ensamblada correctamente. Motivo de parada oficial: {finish_reason}")
                
                # Fallback por si la API no reportó los tokens durante el stream
                if output_tokens == 0:
                    output_tokens = len(final_markdown) // 4
                
                token_counts = {
                    "input": input_tokens,
                    "output": output_tokens
                }
                
                return final_markdown, token_counts, "gemini-2.5-flash"

            except concurrent.futures.TimeoutError:
                logger.error(f"⏰ ¡TIMEOUT! El stream excedió los {timeout_val}s.")
                raise Exception("TIMEOUT_INTERNAL_ERROR")
            except Exception as e:
                print("\n" + "!"*60)
                print("DETALLE TÉCNICO DEL ERROR (STACK TRACE):")
                traceback.print_exc()
                print("!"*60 + "\n")
                raise e

# INSTANCIA GLOBAL
chat_service = ChatService()