import os
from pathlib import Path
from dotenv import load_dotenv

class Config:
    """
    Configuración centralizada para Grading VAWA.
    """
    APP_VERSION = "v1.1" # Actualizamos versión para control

    # 1. Definición de Rutas Base
    # src/config.py -> src/ -> grading_vawa/
    BASE_DIR = Path(__file__).resolve().parent.parent
    FUNDAMENTOS_DIR = BASE_DIR / "fundamentos"
    OUTPUT_DIR = BASE_DIR / "output"
    LOCAL_OUTPUT_DIR = OUTPUT_DIR / "grading_results"  # Respaldo local permanente
    
    # Archivos de credenciales
    CREDENTIALS_FILE = BASE_DIR / "credentials.json"  # Service Account
    OAUTH_CREDENTIALS_FILE = BASE_DIR / "client_secret_907757756276-qu2lj8eh0cp49c1oeqqumh8j1412295v.apps.googleusercontent.com.json"  # OAuth
    TOKEN_FILE = BASE_DIR / "token.json"

    # 2. Carga de variables de entorno
    load_dotenv(BASE_DIR / ".env")

    # 3. Configuración de Google Cloud (Vertex AI)
    PROJECT_ID = os.getenv("PROJECT_ID")
    LOCATION = os.getenv("LOCATION", "us-west1") 
    
    # 4. Configuración de Drive y Sheets
    SPREADSHEET_ID = os.getenv("SPREADSHEET_ID")
    SHEET_NAME = os.getenv("SHEET_NAME")
    DRIVE_OUTPUT_FOLDER_ID = os.getenv("DRIVE_OUTPUT_FOLDER_ID")
    
    # 5. Scopes (Permisos)
    # Service Account: Necesita cloud-platform para Vertex AI
    SERVICE_ACCOUNT_SCOPES = [
        'https://www.googleapis.com/auth/drive',
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/cloud-platform'
    ]
    
    # OAuth (Usuario): Solo necesita Drive y Sheets para uploads
    OAUTH_SCOPES = [
        'https://www.googleapis.com/auth/drive',
        'https://www.googleapis.com/auth/spreadsheets'
    ]
    
    # Para compatibilidad con código existente (Vertex usa service account)
    SCOPES = SERVICE_ACCOUNT_SCOPES

    # 6. Configuración de Timeouts y Reintentos para IA
    # NOTA: Aumentado a 1200s (20 mins) para WAEs ya que mega-prompts tardan más
    # en procesarse por Vertex AI que los chats secuenciales.
    API_TIMEOUT_SECONDS = 1200 
    
    MAX_RETRIES = 7           # Número máximo de reintentos
    RETRY_MIN_WAIT = 5        # Espera mínima entre reintentos (segundos)
    RETRY_MAX_WAIT = 120      # Segundos máximos entre reintentos

    # 7. URLs de Documentación y Prompts (Definidos por el usuario)
    URL_SYSTEM_INSTRUCTIONS = "https://docs.google.com/document/d/1QsCOdhuV0N-gbujvloZFBKmMKlPRpHc4LNRY8qCMb18/edit?usp=sharing"
    
    # Nuevo prompt único para el proceso de WAEs
    URL_PROMPT_WAES = "https://docs.google.com/document/d/1pK9tyWMvLWJiIwWKGI_L04f7Me-LPvCttaMiF1-lmlQ/edit?usp=sharing"

    @classmethod
    def validate(cls):
        """Asegura que las variables críticas existan antes de arrancar."""
        missing = []
        if not cls.PROJECT_ID: missing.append("PROJECT_ID")
        if not cls.SPREADSHEET_ID: missing.append("SPREADSHEET_ID")
        
        if missing:
            raise ValueError(f"Faltan variables en el .env: {', '.join(missing)}")
        
        if not cls.FUNDAMENTOS_DIR.exists():
            # Intentar crearla si no existe, o avisar
            try:
                os.makedirs(cls.FUNDAMENTOS_DIR, exist_ok=True)
            except Exception:
                pass

# Validar al importar
Config.validate()