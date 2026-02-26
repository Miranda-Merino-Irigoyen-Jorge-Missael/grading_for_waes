import os
# Soluciona problemas de DNS y red en Windows con gRPC
os.environ.setdefault("GRPC_DNS_RESOLVER", "native")

import logging
import datetime
from src.config import Config
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

logger = logging.getLogger(__name__)

class AIClientWrapper:
    def __init__(self):
        self.use_vertex = Config.USE_VERTEX_AI
        self.model_name = "gemini-2.5-flash" # Respetando tu indicación del modelo
        
        if self.use_vertex:
            self._init_vertex()
        else:
            self._init_genai()

    def _init_vertex(self):
        import vertexai
        try:
            vertexai.init(project=Config.PROJECT_ID, location=Config.LOCATION)
            logger.info(f"Vertex AI inicializado en {Config.PROJECT_ID} ({Config.LOCATION})")
        except Exception as e:
            logger.error(f"Error inicializando Vertex AI: {e}")
            raise

    def _init_genai(self):
        import google.generativeai as genai
        try:
            genai.configure(api_key=Config.GEMINI_API_KEY)
            logger.info("Gemini API Directa inicializada correctamente.")
        except Exception as e:
            logger.error(f"Error inicializando Gemini API: {e}")
            raise

    @retry(
        stop=stop_after_attempt(Config.MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=Config.RETRY_MIN_WAIT, max=Config.RETRY_MAX_WAIT),
        retry=retry_if_exception_type((Exception,)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def create_cache(self, cache_name, file_paths, system_instruction, ttl_hours=12):
        logger.info(f"Creando caché '{cache_name}' con {len(file_paths)} documentos. Modo Vertex: {self.use_vertex}")
        
        if self.use_vertex:
            return self._create_cache_vertex(cache_name, file_paths, system_instruction, ttl_hours)
        else:
            return self._create_cache_genai(cache_name, file_paths, system_instruction, ttl_hours)

    def _create_cache_vertex(self, cache_name, file_paths, system_instruction, ttl_hours):
        from vertexai.preview.caching import CachedContent
        from vertexai.generative_models import Part
        
        parts = []
        for path in file_paths:
            try:
                with open(path, "rb") as f:
                    data = f.read()
                parts.append(Part.from_data(data=data, mime_type="application/pdf"))
            except Exception as e:
                logger.error(f"No se pudo leer el archivo {path}: {e}")
                raise

        cache = CachedContent.create(
            model_name=self.model_name,
            display_name=cache_name,
            system_instruction=system_instruction,
            contents=parts,
            ttl=datetime.timedelta(hours=ttl_hours)
        )
        logger.info(f"Caché Vertex creado exitosamente. Expira: {cache.expire_time}")
        return cache

    def _create_cache_genai(self, cache_name, file_paths, system_instruction, ttl_hours):
        import google.generativeai as genai
        
        uploaded_files = []
        for path in file_paths:
            try:
                logger.info(f"Subiendo archivo a Gemini File API para caché: {path}")
                uploaded_file = genai.upload_file(path=path, mime_type="application/pdf")
                uploaded_files.append(uploaded_file)
            except Exception as e:
                logger.error(f"No se pudo subir {path} a Gemini File API: {e}")
                raise
        
        cache = genai.caching.CachedContent.create(
            model=f"models/{self.model_name}",
            display_name=cache_name,
            system_instruction=system_instruction,
            contents=uploaded_files,
            ttl=datetime.timedelta(hours=ttl_hours)
        )
        logger.info(f"Caché Gemini API creado exitosamente. Expira: {cache.expire_time}")
        return cache

    def get_model_from_cache(self, cache_obj):
        try:
            if self.use_vertex:
                from vertexai.generative_models import GenerativeModel
                return GenerativeModel.from_cached_content(cached_content=cache_obj)
            else:
                import google.generativeai as genai
                return genai.GenerativeModel.from_cached_content(cached_content=cache_obj)
        except Exception as e:
            logger.error(f"Error instanciando modelo desde caché: {e}")
            raise

# Instanciamos la clase con el mismo nombre de variable para no romper las importaciones en otros archivos
vertex_client = AIClientWrapper()