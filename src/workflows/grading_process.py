import logging
import os
import datetime
import tempfile
import shutil
from src.services.sheets_service import sheets_service
from src.services.drive_service import drive_service
from src.services.chat_service import chat_service
from src.services.cache_service import cache_service
from src.core.google_client import google_manager
from src.utils.drive_tools import get_id_from_url
from src.config import Config

logger = logging.getLogger(__name__)

class GradingProcess:
    def __init__(self):
        """Inicializar el caché compartido una sola vez."""
        self.cache_obj = None
    
    def run(self):
        logger.info(">>> INICIANDO PROCESO DE GRADING VAWA <<<")
        
        # 1. Preparar Caché (Fundamentos) - UNA SOLA VEZ para todas las filas
        try:
            logger.info("Cargando Cache de Fundamentos...")
            self.cache_obj = cache_service.ensure_fundamentos_cache()
            logger.info("Cache de fundamentos cargado exitosamente.")
        except Exception as e:
            logger.critical(f"Fallo crítico inicializando caché: {e}")
            return # Detener todo si no hay conocimiento base

        # 2. Obtener trabajo pendiente
        pending_rows = sheets_service.get_pending_rows()
        logger.info(f"Se encontraron {len(pending_rows)} casos pendientes de procesar.")

        for row in pending_rows:
            self.process_single_case(row)

        logger.info(">>> PROCESO FINALIZADO <<<")

    def process_single_case(self, row_data):
        row_idx = row_data['row_idx']
        client_name = row_data['client_name']
        logger.info(f"--- Procesando Fila {row_idx}: {client_name} ---")
        
        # ✅ IMPORTANTE: Inicializar NUEVA sesión de chat para ESTE caso específico
        # Esto garantiza que NO hay contaminación cruzada entre casos
        try:
            logger.info(f"Inicializando sesión de chat independiente para {client_name}...")
            chat_service.initialize_session(cache_obj=self.cache_obj)
        except Exception as e:
            logger.error(f"Error inicializando sesión de chat para {client_name}: {e}")
            sheets_service.update_status(row_idx, f"ERROR: No se pudo iniciar chat - {str(e)[:40]}")
            return

        # Crear carpeta temporal para este caso
        temp_dir = tempfile.mkdtemp()
        patient_pdfs = []

        try:
            # A. Marcar inicio en Sheets
            sheets_service.mark_processing_start(row_idx)

            # B. Descargar y Normalizar Documentos del Paciente
            links = row_data['links']
            # Mapeo de nombre legible -> URL
            docs_to_download = [
                ('TRANSCRIPT_INTERVIEW', links['transcript']), # MAIN
                ('DOE_ABUSE', links['doe_abuse']),
                ('DOE_GMC', links['doe_gmc']),
                ('DAIR', links['dair']),
                ('FAIR', links['fair']),
                ('RAPSHEET', links['rapsheet']),
                ('AI_SUMMARY', links['summary'])        # MAIN
            ]

            logger.info("Descargando documentos del cliente...")
            for doc_type, url in docs_to_download:
                if url and len(url) > 5: # Validación simple de URL
                    try:
                        file_id = get_id_from_url(url)
                        output_path = os.path.join(temp_dir, f"{doc_type}.pdf")
                        
                        # DriveService convierte todo a PDF automágicamente
                        drive_service.download_as_pdf(file_id, output_path)
                        patient_pdfs.append((doc_type, output_path))
                    except Exception as e:
                        logger.warning(f"No se pudo descargar {doc_type} ({url}): {e}")
                        # No detenemos el proceso, pero el chat tendrá menos contexto

            if not patient_pdfs:
                raise ValueError("No se pudieron descargar documentos válidos para el cliente.")

            # C. Ejecutar Chat de Grading
            logger.info("Iniciando auditoría con IA para WAEs. Esto puede tardar varios minutos debido a la longitud esperada...")
            final_markdown, tokens, model_name = chat_service.execute_grading_flow(patient_pdfs)

            # --- NUEVO: MONITOR DE TOKENS EN CONSOLA ---
            logger.info("="*50)
            logger.info(f"📊 REPORTE DE TOKENS -> Entrada: {tokens['input']} | SALIDA: {tokens['output']}")
            if tokens['output'] < 1500:
                logger.warning("⚠️ ALERTA: La IA generó menos de 1500 tokens. Sigue siendo una respuesta corta.")
            else:
                logger.info(f"✅ EXCELENTE: La IA generó una respuesta extensa de {tokens['output']} tokens.")
            logger.info("="*50)
            # ------------------------------------------

            # D. GUARDAR RESULTADO LOCALMENTE PRIMERO (RESPALDO PERMANENTE)
            # Crear directorio organizado por fecha
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            local_dir = Config.LOCAL_OUTPUT_DIR / today
            os.makedirs(local_dir, exist_ok=True)
            
            output_filename = f"{row_data['client_id']}_{client_name.replace(' ', '_')}_grading_waes_vawa.md"
            local_output_path = local_dir / output_filename
            
            # Guardar archivo local
            with open(local_output_path, 'w', encoding='utf-8') as f:
                f.write(final_markdown)
            logger.info(f"✅ Resultado guardado localmente en: {local_output_path}")

            # E. Subir a Drive (usando el archivo local como fuente)
            file_metadata = {
                'name': output_filename,
                'parents': [Config.DRIVE_OUTPUT_FOLDER_ID]
            }
            uploaded_file = google_manager.upload_file(
                str(local_output_path),  # Usar archivo local como fuente
                file_metadata,
                mime_type='text/markdown'
            )
            
            grading_link = uploaded_file.get('webViewLink')
            logger.info(f"✅ Resultado subido a Drive: {grading_link}")

            # F. Guardar Resultados en Sheets
            result_data = {
                'grading_url': grading_link,
                'tokens_in': tokens['input'],
                'tokens_out': tokens['output'],
                'llm_used': model_name,
                'start_time': datetime.datetime.now()  # Para cálculo de duración
            }
            sheets_service.write_grading_results(row_idx, result_data)
            logger.info(f"✅ Resultados guardados en Sheets para fila {row_idx}")

        except Exception as e:
            logger.error(f"Error procesando caso {client_name}: {e}")
            try:
                sheets_service.update_status(row_idx, f"ERROR: {str(e)[:50]}")
            except Exception as sheet_err:
                logger.error(f"Red tan inestable que no se pudo actualizar status de error para fila {row_idx}: {sheet_err}")
        finally:
            # Limpieza - Solo borrar PDFs temporales del paciente
            shutil.rmtree(temp_dir, ignore_errors=True)
            # NOTA: El archivo .md local NO se borra, es respaldo permanente

# Instancia global
grading_workflow = GradingProcess()