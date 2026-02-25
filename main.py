import logging
import socket
from src.workflows.grading_process import grading_workflow

# Configuración básica de logs para ver todo en consola
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

def main():
    # --- NUEVO: Blindaje contra conexiones de internet inestables ---
    # Si la red se cae y no hay respuesta en 60 segundos, aborta la petición 
    # actual forzando a que la librería 'tenacity' haga un reintento.
    socket.setdefaulttimeout(300.0)
    # ----------------------------------------------------------------

    try:
        grading_workflow.run()
    except KeyboardInterrupt:
        print("\nProceso detenido por el usuario.")
    except Exception as e:
        print(f"Error fatal en la ejecución: {e}")

if __name__ == "__main__":
    main()