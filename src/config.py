import os
from pathlib import Path
from dotenv import load_dotenv

class Config:
    """
    Configuración centralizada para Grading VAWA.
    Soporta conexión vía Vertex AI o Gemini API Key directa.
    """
    APP_VERSION = "v1.4" # Versión actualizada para reflejar el aumento de timeouts y tokens

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

    # 3. Configuración de IA (Vertex AI vs API Key)
    PROJECT_ID = os.getenv("PROJECT_ID")
    LOCATION = os.getenv("LOCATION", "us-west1") 
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    # Flag para decidir el método de conexión
    USE_VERTEX_AI = os.getenv("USE_VERTEX_AI", "false").lower() == "true"
    
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
    API_TIMEOUT_SECONDS = 300
    MAX_RETRIES = 2           
    RETRY_MIN_WAIT = 5        
    RETRY_MAX_WAIT = 60 # Aumentado también el tiempo máximo de espera entre reintentos      

    # 7. URLs de Documentación y Prompts
    URL_SYSTEM_INSTRUCTIONS = "https://docs.google.com/document/d/10A2RkozCS_HGl5L9b0YZO_NlLy_gA4Ou698XmH1Rzlc/edit?usp=sharing"
    URL_PROMPT_WAES = "https://docs.google.com/document/d/1kRdIgBTcZwEesJnEwvgz7GzVhEYe7jRj3Mod8QWAtTw/edit?usp=sharing"

    @classmethod
    def validate(cls):
        """Asegura que las variables críticas existan antes de arrancar según el modo."""
        missing = []
        if not cls.SPREADSHEET_ID: missing.append("SPREADSHEET_ID")
        
        # Validación dinámica basada en el entorno elegido
        if cls.USE_VERTEX_AI:
            if not cls.PROJECT_ID: missing.append("PROJECT_ID (Requerido para Vertex)")
        else:
            if not cls.GEMINI_API_KEY: missing.append("GEMINI_API_KEY (Requerido para API Directa)")
        
        if missing:
            raise ValueError(f"Faltan variables en el .env: {', '.join(missing)}")
        
        if not cls.FUNDAMENTOS_DIR.exists():
            try:
                os.makedirs(cls.FUNDAMENTOS_DIR, exist_ok=True)
            except Exception:
                pass

# Validar al importar
Config.validate()