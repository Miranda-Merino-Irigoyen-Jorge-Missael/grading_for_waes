import logging
import datetime
from src.core.google_client import google_manager
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

class SheetsService:
    """
    Servicio para interactuar con la Google Sheet 'VAWA NEW GRADING'.
    Maneja lectura de pendientes, actualizaciones de estado y escritura de métricas.
    """
    
    # Mapeo de Columnas (1-based index para gspread)
    COL_STATUS = 3          # Columna C
    COL_VISA_TYPE = 4       # Columna D
    COL_TRANSCRIPT = 5      # Columna E 
    COL_DOE_ABUSE = 6       # Columna F
    COL_DOE_GMC = 7         # Columna G
    COL_DAIR = 8            # Columna H
    COL_FAIR = 9            # Columna I
    COL_RAPSHEET = 10       # Columna J
    COL_SUMMARY = 11        # Columna K
    
    COL_GRADING_LINK = 13   # Columna M 
    
    COL_TOKENS_IN = 15      # Columna O
    COL_TOKENS_OUT = 16     # Columna P
    COL_LLM_USED = 17       # Columna Q
    COL_APP_VERSION = 18    # Columna R
    COL_START_TIME = 19     # Columna S
    COL_END_TIME = 20       # Columna T

    def __init__(self):
        self.client = google_manager.get_sheets_client()
        self.spreadsheet_id = Config.SPREADSHEET_ID
        self.sheet_name = Config.SHEET_NAME
        self._sheet = None

    @property
    def sheet(self):
        """Lazy load de la hoja para no conectar hasta que sea necesario."""
        if not self._sheet:
            try:
                # Abrir spreadsheet por ID y seleccionar la hoja por nombre
                sh = self.client.open_by_key(self.spreadsheet_id)
                self._sheet = sh.worksheet(self.sheet_name)
            except Exception as e:
                logger.error(f"Error conectando a Sheet {self.sheet_name}: {e}")
                raise
        return self._sheet

    def get_pending_rows(self):
        """
        Busca todas las filas con status 'PENDING PROCESSING'.
        Retorna una lista de diccionarios con la info necesaria y el número de fila.
        """
        rows_data = []
        try:
            # Obtener todos los valores (es más eficiente que iterar celdas)
            all_values = self.sheet.get_all_values()
            
            # Iterar saltando encabezados (asumimos fila 1 headers)
            for i, row in enumerate(all_values):
                row_idx = i + 1  # 1-based index
                if row_idx == 1: continue 

                # Asegurar que la fila tenga suficientes columnas para leer el status
                if len(row) >= self.COL_STATUS:
                    status = row[self.COL_STATUS - 1].strip() # -1 porque lista es 0-based
                    
                    if status == 'PENDING PROCESSING':
                        
                        # Validar si hay enlaces presentes. Nos aseguraremos de pasar los que existan.
                        row_data = {
                            'row_idx': row_idx,
                            'client_id': row[0],
                            'client_name': row[1],
                            'visa_type': row[self.COL_VISA_TYPE - 1] if len(row) >= self.COL_VISA_TYPE else "",
                            'links': {
                                'transcript': row[self.COL_TRANSCRIPT - 1] if len(row) >= self.COL_TRANSCRIPT else "",
                                'doe_abuse': row[self.COL_DOE_ABUSE - 1] if len(row) >= self.COL_DOE_ABUSE else "",
                                'doe_gmc': row[self.COL_DOE_GMC - 1] if len(row) >= self.COL_DOE_GMC else "",
                                'dair': row[self.COL_DAIR - 1] if len(row) >= self.COL_DAIR else "",
                                'fair': row[self.COL_FAIR - 1] if len(row) >= self.COL_FAIR else "",
                                'rapsheet': row[self.COL_RAPSHEET - 1] if len(row) >= self.COL_RAPSHEET else "",
                                'summary': row[self.COL_SUMMARY - 1] if len(row) >= self.COL_SUMMARY else ""
                            }
                        }
                        rows_data.append(row_data)
            
            return rows_data

        except Exception as e:
            logger.error(f"Error leyendo filas pendientes: {e}")
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
            TimeoutError,
            ConnectionError
        )),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def update_status(self, row_idx, status):
        """Actualiza la columna C (Status)."""
        try:
            self.sheet.update_cell(row_idx, self.COL_STATUS, status)
            logger.info(f"Fila {row_idx} status actualizado a: {status}")
        except Exception as e:
            logger.error(f"Error actualizando status fila {row_idx}: {e}")
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
            TimeoutError,
            ConnectionError
        )),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def mark_processing_start(self, row_idx):
        """Marca inicio: Status PROCESSING y Fecha de Inicio (Columna S)."""
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        updates = [
            {'range': f'C{row_idx}', 'values': [['PROCESSING']]},
            {'range': f'S{row_idx}', 'values': [[now]]} # Columna S - Start Time
        ]
        try:
            self.sheet.batch_update(updates)
        except Exception as e:
            logger.error(f"Error marcando inicio fila {row_idx}: {e}")
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
            TimeoutError,
            ConnectionError
        )),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def write_grading_results(self, row_idx, result_data):
        """
        Escribe los resultados finales en las columnas M (link), O, P, Q, R, S, T.
        """
        try:
            end_time = datetime.datetime.now()
            start_time = result_data.get('start_time', end_time)
            
            # Preparar la fila desde M hasta T
            # M N O P Q R S T
            # Col N is empty/future -> ''
            values = [
                [
                    result_data.get('grading_url', ''), # M - Grading Link
                    '',                                 # N - (Empty/Future)
                    result_data.get('tokens_in', 0),    # O - Tokens Input
                    result_data.get('tokens_out', 0),   # P - Tokens Output
                    result_data.get('llm_used', 'Gemini'), # Q - LLM Used
                    Config.APP_VERSION,                 # R - Version
                    start_time.strftime("%Y-%m-%d %H:%M:%S"), # S - Start Time
                    end_time.strftime("%Y-%m-%d %H:%M:%S")    # T - End Time
                ]
            ]
            
            # Rango de actualización
            range_name = f'M{row_idx}:T{row_idx}'
            self.sheet.update(range_name=range_name, values=values)
            
            # Actualizar status a COMPLETED
            self.update_status(row_idx, 'COMPLETED')
            logger.info(f"Fila {row_idx} completada exitosamente.")

        except Exception as e:
            logger.error(f"Error escribiendo resultados fila {row_idx}: {e}")
            # Intentar marcar error en status (esto también tiene retry)
            try:
                self.update_status(row_idx, "ERROR SAVING RESULTS")
            except:
                pass  # Si ni siquiera podemos marcar el error, al menos loggeamos
            raise

# Instancia global
sheets_service = SheetsService()