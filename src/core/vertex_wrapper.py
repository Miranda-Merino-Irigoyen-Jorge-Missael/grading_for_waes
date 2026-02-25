import os
# Soluciona problemas de DNS y red en Windows con gRPC
os.environ.setdefault("GRPC_DNS_RESOLVER", "native")

import logging
import datetime
import vertexai
from vertexai.preview.caching import CachedContent
from vertexai.generative_models import GenerativeModel, Part
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
    ServiceUnavailable,
    TooManyRequests
)

logger = logging.getLogger(__name__)

class VertexWrapper:
    def __init__(self):
        self.project_id = Config.PROJECT_ID
        self.location = Config.LOCATION
        self._init_vertex()

    def _init_vertex(self):
        try:
            vertexai.init(project=self.project_id, location=self.location)
            logger.info(f"Vertex AI inicializado en {self.project_id} ({self.location})")
        except Exception as e:
            logger.error(f"Error inicializando Vertex AI: {e}")
            raise

    @retry(
        stop=stop_after_attempt(Config.MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=Config.RETRY_MIN_WAIT, max=Config.RETRY_MAX_WAIT),
        retry=retry_if_exception_type((GoogleAPIError, InternalServerError, ServiceUnavailable, TooManyRequests, TimeoutError, ConnectionError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def create_cache(self, cache_name, file_paths, system_instruction, ttl_hours=12):
        logger.info(f"Creando caché '{cache_name}' con {len(file_paths)} documentos...")
        parts = []
        for path in file_paths:
            try:
                with open(path, "rb") as f:
                    data = f.read()
                parts.append(Part.from_data(data=data, mime_type="application/pdf"))
            except Exception as e:
                logger.error(f"No se pudo leer el archivo {path}: {e}")
                raise

        try:
            cache = CachedContent.create(
                model_name="gemini-2.5-flash",
                display_name=cache_name,
                system_instruction=system_instruction,
                contents=parts,
                ttl=datetime.timedelta(hours=ttl_hours)
            )
            logger.info(f"Caché creado exitosamente. Expira: {cache.expire_time}")
            return cache
        except Exception as e:
            logger.error(f"Error creando CachedContent: {e}")
            raise

    def get_model_from_cache(self, cache_name_or_obj):
        try:
            if isinstance(cache_name_or_obj, CachedContent):
                return GenerativeModel.from_cached_content(cached_content=cache_name_or_obj)
        except Exception as e:
            logger.error(f"Error instanciando modelo desde caché: {e}")
            raise

vertex_client = VertexWrapper()