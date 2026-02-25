import logging
import os
from src.config import Config
from src.core.vertex_wrapper import vertex_client

logger = logging.getLogger(__name__)

class CacheService:
    """
    Servicio dedicado a gestionar el Cache de Conocimientos (Fundamentos).
    """

    def ensure_fundamentos_cache(self):
        """
        Busca los documentos en la carpeta 'fundamentos/' y crea/recupera el caché.
        """
        # 1. Identificar archivos en carpeta fundamentos
        fundamentos_dir = Config.FUNDAMENTOS_DIR
        if not fundamentos_dir.exists():
            raise FileNotFoundError(f"La carpeta de fundamentos no existe: {fundamentos_dir}")

        # Listar todos los PDFs
        files = [f for f in os.listdir(fundamentos_dir) if f.lower().endswith('.pdf')]
        
        if not files:
            raise ValueError(f"No se encontraron PDFs en {fundamentos_dir}. ¡Debes colocar los 4 archivos base ahí!")

        file_paths = [os.path.join(fundamentos_dir, f) for f in files]
        logger.info(f"Archivos base encontrados para caché: {files}")

        # 2. Definir instrucciones del sistema para el Caché
        # (Opcional: puedes poner instrucciones genéricas aquí, ya que las específicas
        # se cargan en el chat_service, pero el caché requiere algo de base).
        system_instruction_cache = """
        Eres un experto analista legal especializado en casos VAWA y Visa T.
        Tu conocimiento base proviene estrictamente de los documentos adjuntos en este contexto.
        Utiliza esta información para auditar y clasificar casos.
        """

        # 3. Crear el caché
        # Nota: En un entorno productivo ideal, guardaríamos el ID del caché en un archivo
        # json para no crearlo de nuevo si ya existe y es válido. 
        # Por ahora, crearemos uno nuevo para asegurar frescura en cada ejecución masiva.
        cache_name = "vawa-fundamentos-cache"
        
        try:
            cache = vertex_client.create_cache(
                cache_name=cache_name,
                file_paths=file_paths,
                system_instruction=system_instruction_cache,
                ttl_hours=12
            )
            return cache
        except Exception as e:
            logger.error(f"Error fatal creando el caché de fundamentos: {e}")
            raise

# Instancia global
cache_service = CacheService()