import os
from pathlib import Path
from dotenv import load_dotenv

class Config:
    """
    Configuración centralizada para Grading VAWA.
    """
    APP_VERSION = "v1.2" # Actualizamos versión para control

    # 1. Definición de Rutas Base
    BASE_DIR = Path(__file__).resolve().parent.parent
    FUNDAMENTOS_DIR = BASE_DIR / "fundamentos"
    OUTPUT_DIR = BASE_DIR / "output"
    LOCAL_OUTPUT_DIR = OUTPUT_DIR / "grading_results"  
    
    # Archivos de credenciales
    CREDENTIALS_FILE = BASE_DIR / "credentials.json"  
    OAUTH_CREDENTIALS_FILE = BASE_DIR / "client_secret_907757756276-qu2lj8eh0cp49c1oeqqumh8j1412295v.apps.googleusercontent.com.json"  
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
    SERVICE_ACCOUNT_SCOPES = [
        'https://www.googleapis.com/auth/drive',
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/cloud-platform'
    ]
    
    OAUTH_SCOPES = [
        'https://www.googleapis.com/auth/drive',
        'https://www.googleapis.com/auth/spreadsheets'
    ]
    
    SCOPES = SERVICE_ACCOUNT_SCOPES

    # 6. Configuración de Timeouts y Reintentos para IA
    # 🚨 REDUCIDO: 480 segundos (8 minutos) es más que suficiente para un mega-prompt.
    API_TIMEOUT_SECONDS = 480 
    
    # 🚨 REDUCIDO: Bajamos de 7 a 2 reintentos para no quedarnos pasmados horas
    MAX_RETRIES = 2           
    RETRY_MIN_WAIT = 5        
    RETRY_MAX_WAIT = 30       

    # 7. URLs de Documentación y Prompts
    URL_SYSTEM_INSTRUCTIONS = "https://docs.google.com/document/d/1QsCOdhuV0N-gbujvloZFBKmMKlPRpHc4LNRY8qCMb18/edit?usp=sharing"
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
            try:
                os.makedirs(cls.FUNDAMENTOS_DIR, exist_ok=True)
            except Exception:
                pass

# Validar al importar
Config.validate()