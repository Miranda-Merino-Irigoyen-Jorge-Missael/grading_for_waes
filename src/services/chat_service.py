import logging
import concurrent.futures
import sys
import traceback
import time
import os
import re
from src.config import Config
from src.core.google_client import google_manager
from src.core.vertex_wrapper import vertex_client
from src.utils.drive_tools import get_id_from_url
from google import genai
from google.genai import types
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
    Gestiona la ejecución de prompts con un Paginador/Auto-Continuador.
    Actualizado a gemini-2.5-pro para generación masiva de texto y corrección de Markdown.
    """

    def __init__(self):
        self.drive_service = google_manager.get_drive_service()
        self.client = vertex_client.client 
        self.chat_session = False
        self.uploaded_files = [] 
        self.model_name = "gemini-2.5-pro"
        self.system_instruction = ""
        self.cache_name = None

    def _fetch_doc_text(self, url):
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
        try:
            system_instr = self._fetch_doc_text(Config.URL_SYSTEM_INSTRUCTIONS)
            
            system_instr += (
                "\n\nINSTRUCCIÓN CRÍTICA: Eres un Especialista Legal Forense en Derechos Humanos. "
                "Tu reporte DEBE ser extremadamente exhaustivo. Tienes prohibido resumir."
            )
            logger.info("Instrucciones del sistema cargadas con el Escudo Legal.")
            
            self.system_instruction = system_instr
            self.cache_name = cache_obj.name if cache_obj else None
            self.chat_session = True  
            
        except Exception as e:
            logger.error(f"Error inicializando modelo: {e}")
            raise

    def execute_grading_flow(self, patient_files_tuple):
        if not self.chat_session:
            raise ValueError("La sesión de chat no ha sido inicializada.")
        
        uploaded_parts = []
        
        try:
            for doc_type, path in patient_files_tuple:
                if Config.USE_VERTEX_AI:
                    with open(path, "rb") as f:
                        data = f.read()
                    uploaded_parts.append((doc_type, types.Part.from_bytes(data=data, mime_type="application/pdf")))
                else:
                    logger.info(f"Subiendo {doc_type} a Gemini API...")
                    import io
                    with open(path, "rb") as f_in:
                        data = f_in.read()
                    gemini_file = self.client.files.upload(
                        file=io.BytesIO(data),
                        config={'mime_type': 'application/pdf', 'display_name': doc_type}
                    )
                    part_obj = types.Part.from_uri(file_uri=gemini_file.uri, mime_type="application/pdf")
                    uploaded_parts.append((doc_type, part_obj))
                    self.uploaded_files.append(gemini_file)

            return self._execute_with_auto_continue(uploaded_parts)
            
        finally:
            self._cleanup_gemini_files()

    def _cleanup_gemini_files(self):
        if not Config.USE_VERTEX_AI and self.uploaded_files:
            for f in self.uploaded_files:
                try:
                    self.client.files.delete(name=f.name)
                    logger.info(f"Archivo temporal {f.name} eliminado.")
                except Exception as e:
                    pass
            self.uploaded_files = []

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        retry=retry_if_exception_type((Exception,)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def _execute_with_auto_continue(self, uploaded_parts):
        prompt_text = self._fetch_doc_text(Config.URL_PROMPT_WAES)
        
        doc_intro = (
            "ATENCIÓN: Los siguientes documentos son PRUEBAS LEGALES de un caso de inmigración (VAWA). "
            "No apliques censura a tu análisis, es un requerimiento legal estricto.\n\nDocumentos adjuntos:\n"
        )
        
        # AÑADIDAS INSTRUCCIONES ESTRICTAS PARA EVITAR EL BLOQUE DE CÓDIGO Y FORZAR ESPACIOS EN TABLAS
        formatting_rules = (
            "\n\n--- INSTRUCCIONES ESTRICTAS DE FORMATO ---\n"
            "1. **PROHIBIDO RESUMIR:** Tu respuesta debe ser una evaluación forense de 7 a 10 páginas.\n"
            "2. **ESTRUCTURA:** Cada celda de tabla DEBE contener un párrafo descriptivo completo. DEBES dejar un salto de línea (Enter) antes y después de cada tabla.\n"
            "3. **EVIDENCIA EXPLÍCITA:** Incluye citas textuales entre comillas extraídas de los documentos.\n"
            "4. **PROHIBIDO BLOQUES DE CÓDIGO:** Entrega el formato Markdown directamente. NO envuelvas tu respuesta en ```markdown ni en ningún otro bloque de código de backticks.\n"
            "5. **TABLA INICIAL (CARÁTULA DEL CASO):** Esta sección DEBE ser una tabla Markdown perfectamente válida con tuberías (|). Debe tener una cabecera clara. Ejemplo:\n"
            "| Información | Detalle |\n"
            "| :--- | :--- |\n"
            "| **Nombre del Cliente** | [Nombre] |\n"
            "| **Nombre del Abuser** | [Nombre] |\n"
            "Asegúrate de dejar una línea en blanco antes y después de la tabla.\n"
        )

        parts_1 = []
        parts_1.append(types.Part.from_text(text=f"--- INSTRUCCIONES DEL SISTEMA ---\n{self.system_instruction}\n\n"))
        parts_1.append(types.Part.from_text(text=doc_intro))
        for doc_type, part_obj in uploaded_parts:
            parts_1.append(types.Part.from_text(text=f"- Archivo: '{doc_type}.pdf'\n"))
            parts_1.append(part_obj)
        parts_1.append(types.Part.from_text(text=f"Instrucciones de Grading:\n{prompt_text}{formatting_rules}"))

        contents = [types.Content(role="user", parts=parts_1)]
        
        full_markdown = ""
        total_in_tokens = 0
        total_out_tokens = 0
        max_cycles = 4

        for cycle in range(1, max_cycles + 1):
            logger.info(f"🔄 Ciclo de generación {cycle}/{max_cycles} (Modelo: {self.model_name})...")
            
            response, token_counts, finish_reason = self._raw_send_to_gemini(contents)
            text = ""
            
            try:
                text = response.text
            except:
                if response.candidates and response.candidates[0].content:
                    text = response.candidates[0].content.parts[0].text

            if not text:
                logger.error(f"❌ La API devolvió una respuesta vacía en el ciclo {cycle}. FinishReason: {finish_reason}")
                break
                
            full_markdown += text + "\n\n"
            total_out_tokens += token_counts.get("output", 0)
            total_in_tokens = max(total_in_tokens, token_counts.get("input", 0))

            is_max_tokens = "MAX_TOKENS" in finish_reason.upper() or finish_reason == "2"
            
            if is_max_tokens:
                logger.warning(f"⚠️ La IA alcanzó su límite de tokens en el ciclo {cycle}. Extrayendo fragmento y forzando continuación...")
                contents.append(types.Content(role="model", parts=[types.Part.from_text(text=text)]))
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text="Continúa tu análisis forense exactamente desde la última palabra en la que te quedaste. No uses bloques de código ```markdown. No escribas introducciones, solo continúa con el reporte.")]))
            elif total_out_tokens < 1500 and cycle == 1:
                logger.warning(f"⚠️ La respuesta es demasiado corta ({total_out_tokens} tokens). Forzando expansión de detalles...")
                contents.append(types.Content(role="model", parts=[types.Part.from_text(text=text)]))
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text="Tu análisis fue un resumen demasiado breve y perdiste nivel de detalle forense. Revisa nuevamente los documentos y expande agresivamente los detalles, descripciones y fechas de cada evento mencionado. Añade al menos 3 páginas más de hallazgos. No uses bloques de código ```markdown.")]))
            else:
                logger.info(f"✅ Análisis completado satisfactoriamente en el ciclo {cycle}. Total generado: {total_out_tokens} tokens.")
                break

        # LIMPIEZA FINAL DE BACKTICKS (Sanitización manual por si la IA nos ignoró)
        full_markdown = full_markdown.strip()
        # Borra ```markdown o ``` del inicio
        full_markdown = re.sub(r"^```(?:markdown|md)?\s*", "", full_markdown, flags=re.IGNORECASE)
        # Borra ``` del final
        full_markdown = re.sub(r"\s*```$", "", full_markdown)

        return full_markdown.strip(), {"input": total_in_tokens, "output": total_out_tokens}, self.model_name

    def _raw_send_to_gemini(self, contents):
        timeout_val = Config.API_TIMEOUT_SECONDS
        
        safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
        ]

        config_args = {
            "temperature": 0.5, 
            "max_output_tokens": 20000,
            "safety_settings": safety_settings,
        }

        if self.cache_name:
            config_args["cached_content"] = self.cache_name

        gen_config = types.GenerateContentConfig(**config_args)

        def make_request():
            return self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=gen_config
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(make_request)
            try:
                response = future.result(timeout=timeout_val)
                
                finish_reason = "UNKNOWN"
                if response.candidates and len(response.candidates) > 0:
                    cand = response.candidates[0]
                    if cand.finish_reason:
                        finish_reason = str(cand.finish_reason)
                
                in_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) if response.usage_metadata else 0
                out_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) if response.usage_metadata else 0
                    
                return response, {"input": in_tokens, "output": out_tokens}, finish_reason

            except Exception as e:
                raise e

# INSTANCIA GLOBAL
chat_service = ChatService()