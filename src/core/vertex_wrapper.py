import os
import logging
import datetime
import tempfile
import shutil
from src.config import Config
from google import genai
from google.genai import types
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

# Soluciona problemas de DNS y red en Windows con gRPC
os.environ.setdefault("GRPC_DNS_RESOLVER", "native")

logger = logging.getLogger(__name__)

class AIClientWrapper:
    def __init__(self):
        self.use_vertex = Config.USE_VERTEX_AI
        # CAMBIO VITAL: Unificamos a gemini-2.5-pro para que coincida con chat_service.py
        self.model_name = "gemini-2.5-pro" 
        
        # El nuevo SDK unifica la inicialización
        if self.use_vertex:
            self.client = genai.Client(
                vertexai=True,
                project=Config.PROJECT_ID,
                location=Config.LOCATION
            )
            logger.info(f"Google GenAI Client (Vertex AI) inicializado en {Config.PROJECT_ID}.")
        else:
            self.client = genai.Client(api_key=Config.GEMINI_API_KEY)
            logger.info("Google GenAI Client (Gemini API Directa) inicializado.")

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        retry=retry_if_exception_type((Exception,)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def _upload_file_with_retry(self, path):
        """Sube el archivo a Gemini usando un nombre seguro para evadir el error Unicode de Windows."""
        logger.info(f"Subiendo archivo a Gemini API: {path}")
        
        # Trampa: Crear una copia temporal con nombre puramente ASCII (evita WinError Unicode)
        temp_dir = tempfile.mkdtemp()
        safe_path = os.path.join(temp_dir, "temp_safe_upload.pdf")
        shutil.copy2(path, safe_path)
        
        try:
            # Subimos el archivo con el nombre seguro, pero le decimos a Gemini su nombre original
            uploaded_file = self.client.files.upload(
                file=safe_path,
                config=types.UploadFileConfig(
                    display_name=os.path.basename(path) # Aquí sí acepta acentos sin explotar
                )
            )
            return uploaded_file
        finally:
            # Limpiamos el rastro
            shutil.rmtree(temp_dir, ignore_errors=True)

    def create_cache(self, cache_name, file_paths, system_instruction, ttl_hours=12):
        logger.info(f"Creando caché '{cache_name}' con {len(file_paths)} documentos. Modo Vertex: {self.use_vertex}. Modelo: {self.model_name}")
        
        parts = []
        uploaded_files = []
        
        try:
            for path in file_paths:
                if self.use_vertex:
                    # En Vertex podemos mandar los bytes directamente en la petición
                    with open(path, "rb") as f:
                        data = f.read()
                    parts.append(types.Part.from_bytes(data=data, mime_type="application/pdf"))
                    logger.info(f"✅ Archivo leído localmente para Vertex: {os.path.basename(path)}")
                else:
                    # En la API directa requerimos subir el archivo
                    uploaded_file = self._upload_file_with_retry(path)
                    uploaded_files.append(uploaded_file)
                    parts.append(types.Part.from_uri(file_uri=uploaded_file.uri, mime_type="application/pdf"))
                    logger.info(f"✅ Subida exitosa confirmada: {os.path.basename(path)}")
            
            # Una vez preparados los parts, creamos el caché usando el nuevo SDK unificado
            logger.info("Generando caché en la IA...")
            cache = self.client.caches.create(
                model=self.model_name,
                config=types.CreateCachedContentConfig(
                    contents=[types.Content(role="user", parts=parts)],
                    system_instruction=system_instruction,
                    display_name=cache_name,
                    ttl=f"{ttl_hours * 3600}s"
                )
            )
            logger.info(f"🎉 Caché creado exitosamente. ID: {cache.name}")
            return cache

        except Exception as e:
            logger.error(f"Error generando el caché: {e}")
            # Limpieza en caso de fallo crítico
            if not self.use_vertex:
                for f in uploaded_files:
                    try:
                        self.client.files.delete(name=f.name)
                    except:
                        pass
            raise

# Instanciamos la clase globalmente
vertex_client = AIClientWrapper()