import os
import logging
import gspread
from google.oauth2.service_account import Credentials
from google.oauth2.credentials import Credentials as OAuthCredentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
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
    ServiceUnavailable
)

logger = logging.getLogger(__name__)

class GoogleClientManager:
    """
    Singleton para manejar las conexiones autenticadas a Google.
    Usa Service Account Credentials definidas en Config.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GoogleClientManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._creds = None
        self._oauth_creds = None
        self._drive_service = None
        self._oauth_drive_service = None
        self._sheets_client = None
        self._initialized = True

    def _get_creds(self):
        """Carga las credenciales de la Service Account."""
        if not self._creds:
            try:
                logger.info(f"Cargando credenciales desde: {Config.CREDENTIALS_FILE}")
                
                # Verificamos que el archivo exista
                if not os.path.exists(Config.CREDENTIALS_FILE):
                    raise FileNotFoundError(f"No se encontró el archivo de credenciales: {Config.CREDENTIALS_FILE}")

                self._creds = Credentials.from_service_account_file(
                    Config.CREDENTIALS_FILE, 
                    scopes=Config.SCOPES
                )
            except Exception as e:
                logger.error(f"Error cargando credenciales: {e}")
                raise
        return self._creds

    def get_drive_service(self):
        """Retorna el servicio de Google Drive API v3."""
        if not self._drive_service:
            creds = self._get_creds()
            self._drive_service = build('drive', 'v3', credentials=creds)
        return self._drive_service

    def get_sheets_client(self):
        """Retorna el cliente de gspread para Sheets."""
        if not self._sheets_client:
            creds = self._get_creds()
            self._sheets_client = gspread.authorize(creds)
        return self._sheets_client
    
    def _get_oauth_creds(self):
        """Carga las credenciales OAuth del usuario (token.json)."""
        if not self._oauth_creds:
            try:
                logger.info("Cargando credenciales OAuth desde token.json...")
                
                # Intentar cargar el token existente
                if os.path.exists(Config.TOKEN_FILE):
                    self._oauth_creds = OAuthCredentials.from_authorized_user_file(
                        Config.TOKEN_FILE, Config.OAUTH_SCOPES
                    )
                    
                    # Si el token ha expirado, intentar refrescarlo
                    if self._oauth_creds and self._oauth_creds.expired and self._oauth_creds.refresh_token:
                        logger.info("Refrescando token OAuth...")
                        self._oauth_creds.refresh(Request())
                        
                        # Guardar el token refrescado
                        with open(Config.TOKEN_FILE, 'w') as token:
                            token.write(self._oauth_creds.to_json())
                        logger.info("Token OAuth refrescado exitosamente.")
                    elif self._oauth_creds and self._oauth_creds.valid:
                        logger.info("Token OAuth válido encontrado.")
                    else:
                        # Token inválido, necesitamos reautenticar
                        logger.warning("Token OAuth inválido, iniciando nueva autenticación...")
                        self._oauth_creds = None
                
                # Si no hay token válido, iniciar el flujo OAuth
                if not self._oauth_creds:
                    logger.info("Iniciando flujo OAuth...")
                    if not os.path.exists(Config.OAUTH_CREDENTIALS_FILE):
                        raise FileNotFoundError(
                            f"No se encontró el archivo de credenciales OAuth: {Config.OAUTH_CREDENTIALS_FILE}"
                        )
                    
                    flow = InstalledAppFlow.from_client_secrets_file(
                        Config.OAUTH_CREDENTIALS_FILE, 
                        Config.OAUTH_SCOPES
                    )
                    self._oauth_creds = flow.run_local_server(port=0)
                    
                    # Guardar las credenciales para la próxima ejecución
                    with open(Config.TOKEN_FILE, 'w') as token:
                        token.write(self._oauth_creds.to_json())
                    logger.info("Token OAuth guardado exitosamente.")
                
            except Exception as e:
                logger.error(f"Error cargando credenciales OAuth: {e}")
                raise
        return self._oauth_creds
    
    def get_oauth_drive_service(self):
        """Retorna el servicio de Google Drive API v3 usando OAuth (usuario)."""
        if not self._oauth_drive_service:
            creds = self._get_oauth_creds()
            self._oauth_drive_service = build('drive', 'v3', credentials=creds)
        return self._oauth_drive_service
    
    @retry(
        stop=stop_after_attempt(Config.MAX_RETRIES),
        wait=wait_exponential(
            multiplier=1,
            min=Config.RETRY_MIN_WAIT,
            max=Config.RETRY_MAX_WAIT
        ),
        retry=retry_if_exception_type((
            HttpError,
            GoogleAPIError,
            TimeoutError,
            ConnectionError
        )),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def upload_file(self, file_path, file_metadata, mime_type='text/markdown'):
        """
        Helper para subir archivos usando OAuth (usuario) para evitar
        problemas de cuota de almacenamiento con Service Accounts.
        """
        from googleapiclient.http import MediaFileUpload
        # Usar OAuth Drive service en lugar de Service Account
        service = self.get_oauth_drive_service()
        
        media_body = MediaFileUpload(file_path, mimetype=mime_type, resumable=True)
        
        file = service.files().create(
            body=file_metadata,
            media_body=media_body,
            fields='id, webViewLink'
        ).execute()
        
        return file

# Instancia global
google_manager = GoogleClientManager()