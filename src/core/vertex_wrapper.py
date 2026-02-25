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
    """
    Wrapper para manejar Vertex AI y el sistema de Caching.
    Implementa la lógica de 'GestorCache' adaptada al proyecto.
    """

    def __init__(self):
        self.project_id = Config.PROJECT_ID
        self.location = Config.LOCATION
        self._init_vertex()

    def _init_vertex(self):
        """Inicializa la conexión con Vertex AI.
        
        Configura gRPC keepalive para evitar que firewalls corporativos
        corten las conexiones TCP durante respuestas largas del modelo.
        """
        import os
        # Keepalive: enviar ping cada 60s si la conexión está inactiva
        os.environ.setdefault("GRPC_KEEPALIVE_TIME_MS", "60000")
        # Esperar hasta 20s la respuesta al ping antes de cerrar
        os.environ.setdefault("GRPC_KEEPALIVE_TIMEOUT_MS", "20000")
        # Permitir keepalive incluso sin streams activos
        os.environ.setdefault("GRPC_KEEPALIVE_PERMIT_WITHOUT_CALLS", "1")
        # Número máximo de pings sin datos (evita que el servidor rechace)
        os.environ.setdefault("GRPC_HTTP2_MAX_PINGS_WITHOUT_DATA", "0")
        
        try:
            vertexai.init(project=self.project_id, location=self.location)
            logger.info(f"Vertex AI inicializado en {self.project_id} ({self.location}) [gRPC keepalive activo]")
        except Exception as e:
            logger.error(f"Error inicializando Vertex AI: {e}")
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
    def create_cache(self, cache_name, file_paths, system_instruction, ttl_hours=12):
        """
        Crea un contexto en caché con los documentos PDF proporcionados.
        """
        logger.info(f"Creando caché '{cache_name}' con {len(file_paths)} documentos...")
        
        parts = []
        for path in file_paths:
            try:
                with open(path, "rb") as f:
                    data = f.read()
                # Asumimos PDF por defecto dado el requerimiento
                mime_type = "application/pdf"
                parts.append(Part.from_data(data=data, mime_type=mime_type))
                logger.debug(f"Documento cargado para caché: {path}")
            except Exception as e:
                logger.error(f"No se pudo leer el archivo {path}: {e}")
                raise

        try:
            # Crear el caché usando la API de preview
            cache = CachedContent.create(
                model_name="gemini-2.5-flash", # Modelo recomendado para producción/contexto largo
                display_name=cache_name,
                system_instruction=system_instruction,
                contents=parts,
                ttl=datetime.timedelta(hours=ttl_hours)
            )
            logger.info(f"Caché creado exitosamente. Expira: {cache.expire_time}")
            return cache
        except Exception as e:
            logger.error(f"Error creando CachedContent en Vertex: {e}")
            raise

    def get_model_from_cache(self, cache_name_or_obj):
        """
        Devuelve una instancia de GenerativeModel conectada al caché.
        """
        try:
            # Si recibimos el objeto caché directo
            if isinstance(cache_name_or_obj, CachedContent):
                return GenerativeModel.from_cached_content(cached_content=cache_name_or_obj)
            
            # TODO: Si en el futuro necesitas recuperar por nombre (string),
            # aquí implementaríamos la lógica de listar y buscar por display_name.
            pass
        except Exception as e:
            logger.error(f"Error instanciando modelo desde caché: {e}")
            raise

# Instancia global para uso fácil
vertex_client = VertexWrapper()