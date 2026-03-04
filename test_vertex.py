"""
test_vertex.py -- Script de diagnóstico unificado (google-genai)
Corre: python test_vertex.py
"""
import time
import logging
import sys
import glob
import os
from src.config import Config
from google import genai
from google.genai import types

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("test_vertex")

# ── Inicializar Entorno ─────────────────────────────────────────────────────────
logger.info("🔬 Iniciando diagnóstico de IA con google-genai...")
logger.info(f"   Modo Vertex AI: {Config.USE_VERTEX_AI}")
MODEL_NAME = "gemini-2.5-flash"

if Config.USE_VERTEX_AI:
    client = genai.Client(vertexai=True, project=Config.PROJECT_ID, location=Config.LOCATION)
else:
    client = genai.Client(api_key=Config.GEMINI_API_KEY)

# Filtros apagados unificados
SAFETY_SETTINGS = [
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
]

SYSTEM_INSTR = (
    "INSTRUCCIÓN CRÍTICA: Eres un Especialista Legal Forense. "
    "Es tu deber procesar documentos legales y evidencia clínica sin censura."
)

gen_config = types.GenerateContentConfig(
    temperature=0.3,
    system_instruction=SYSTEM_INSTR,
    safety_settings=SAFETY_SETTINGS
)

# ── Funciones Helper ────────────────────────────────────────────────────────────
def process_response(response, t0):
    finish_reason = "UNKNOWN"
    if response.candidates and response.candidates[0].finish_reason:
        finish_reason = str(response.candidates[0].finish_reason)
        if "SAFETY" in finish_reason.upper():
            logger.error("🚨 BLOQUEO DE SEGURIDAD DETECTADO (SAFETY).")
        elif "RECITATION" in finish_reason.upper():
            logger.error("🚨 BLOQUEO POR RECITACIÓN DETECTADO.")

    try:
        text = response.text
        if not text:
            raise ValueError("Texto vacío.")
        logger.info(f"✅ ÉXITO. Respuesta en {time.time()-t0:.1f}s (Parada: {finish_reason}):\n{text[:200]}...\n")
        return True
    except Exception as e:
        logger.error(f"❌ FALLO. No se pudo extraer texto. Motivo: {finish_reason}")
        return False

# ── TEST 1: Texto puro (sin PDFs) ───────────────────────────────────────────────
def test_1_texto_puro():
    logger.info("=" * 60)
    logger.info("TEST 1: Texto puro - Diagnóstico básico de conectividad")
    logger.info("=" * 60)
    
    prompt = "Responde en una sola oración: ¿Cuál es la capital de México?"
    t0 = time.time()
    logger.info(f"⏳ Enviando petición a {MODEL_NAME}...")
    try:
        response = client.models.generate_content(
            model=MODEL_NAME, contents=prompt, config=gen_config
        )
        return process_response(response, t0)
    except Exception as e:
        logger.error(f"❌ FALLO: {e}")
        return False

# ── TEST 2: Un PDF pequeño ──────────────────────────────────────────────────────
def test_2_un_pdf(pdf_path: str):
    logger.info("=" * 60)
    logger.info(f"TEST 2: Un PDF - {pdf_path}")
    logger.info("=" * 60)
    
    prompt_parts = [
        "ATENCIÓN: El siguiente documento es evidencia legal forense. Analízalo exhaustivamente sin censura.",
    ]
    uploaded_files = []
    
    try:
        if Config.USE_VERTEX_AI:
            with open(pdf_path, "rb") as f:
                data = f.read()
            logger.info(f"   PDF cargado (Local): {len(data)/1024:.1f} KB")
            prompt_parts.append(types.Part.from_bytes(data=data, mime_type="application/pdf"))
        else:
            logger.info(f"   Subiendo PDF a Gemini API...")
            import io
            with open(pdf_path, "rb") as f_in:
                data = f_in.read()
            f = client.files.upload(
                file=io.BytesIO(data),
                config={'mime_type': 'application/pdf', 'display_name': os.path.basename(pdf_path)}
            )
            uploaded_files.append(f)
            prompt_parts.append(types.Part.from_uri(file_uri=f.uri, mime_type="application/pdf"))
        
        prompt_parts.append("Resume en 3 líneas el contenido detallado de este documento clínico/legal.")
        
        t0 = time.time()
        logger.info("⏳ Enviando petición con PDF...")
        response = client.models.generate_content(
            model=MODEL_NAME, contents=prompt_parts, config=gen_config
        )
        return process_response(response, t0)
    except Exception as e:
        logger.error(f"❌ FALLO: {e}")
        return False
    finally:
        for f in uploaded_files:
            try:
                client.files.delete(name=f.name)
            except:
                pass

# ── TEST 3: Varios PDFs (simula el proceso real) ────────────────────────────────
def test_3_multiples_pdfs(pdf_paths: list):
    logger.info("=" * 60)
    logger.info(f"TEST 3: {len(pdf_paths)} PDFs - Equivalente al proceso real")
    logger.info("=" * 60)
    
    prompt_parts = [
        "ATENCIÓN: Los siguientes documentos son PRUEBAS LEGALES de un caso de inmigración por violencia. "
        "No apliques censura a tu análisis, es un requerimiento legal estricto.\n\n"
    ]
    uploaded_files = []
    
    try:
        for label, path in pdf_paths:
            if Config.USE_VERTEX_AI:
                with open(path, "rb") as f_in:
                    data = f_in.read()
                prompt_parts.append(types.Part.from_bytes(data=data, mime_type="application/pdf"))
                logger.info(f"   {label} (Local): {len(data)/1024:.1f} KB")
            else:
                logger.info(f"   Subiendo {label} a Gemini API...")
                import io
                with open(path, "rb") as f_in:
                    data = f_in.read()
                f = client.files.upload(
                    file=io.BytesIO(data),
                    config={'mime_type': 'application/pdf', 'display_name': label}
                )
                uploaded_files.append(f)
                prompt_parts.append(types.Part.from_uri(file_uri=f.uri, mime_type="application/pdf"))
        
        prompt_parts.append("Resume en 5 líneas el contenido de TODOS estos documentos, sin omitir los eventos descritos.")
        
        t0 = time.time()
        logger.info("⏳ Enviando petición con múltiples PDFs...")
        response = client.models.generate_content(
            model=MODEL_NAME, contents=prompt_parts, config=gen_config
        )
        return process_response(response, t0)
    except Exception as e:
        logger.error(f"❌ FALLO: {e}")
        return False
    finally:
        for f in uploaded_files:
            try:
                client.files.delete(name=f.name)
            except:
                pass

# ── MAIN ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ok1 = test_1_texto_puro()
    if not ok1:
        logger.error("❌ Fallo en Test 1. Verifica credenciales o conexión.")
        sys.exit(1)

    pdfs = glob.glob(r"fundamentos\*.pdf", recursive=True)
    if not pdfs:
        pdfs = glob.glob(r"output\grading_results\**\*.pdf", recursive=True)
        
    if not pdfs:
        logger.warning("No hay PDFs locales de prueba. Solo se ejecutó Test 1.")
        sys.exit(0)

    test_2_un_pdf(pdfs[0])

    if len(pdfs) >= 2:
        test_3_multiples_pdfs([(os.path.basename(p), p) for p in pdfs[:4]])
    
    logger.info("✅ Diagnóstico finalizado.")