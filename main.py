import logging
import sys
import os
from src.workflows.grading_process import grading_workflow

# Configuración de logs optimizada para depuración inmediata
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout) # Asegura que salga a la terminal
    ]
)

logger = logging.getLogger("main")

def main():
    print("\n" + "="*50)
    print("SISTEMA DE GRADING - MODO DE DEPURACIÓN ACTIVO")
    print("="*50 + "\n")

    try:
        # Ejecutamos el flujo principal
        grading_workflow.run()
        
    except KeyboardInterrupt:
        print("\n[!] Proceso detenido manualmente por el usuario (Ctrl+C).")
    except Exception as e:
        print(f"\n[!!!] ERROR FATAL NO CONTROLADO EN MAIN: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n" + "="*50)
        print("PROCESO TERMINADO")
        print("="*50)

if __name__ == "__main__":
    # Forzar que los prints se vean al momento en la consola de Windows/VSCode
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
    main()