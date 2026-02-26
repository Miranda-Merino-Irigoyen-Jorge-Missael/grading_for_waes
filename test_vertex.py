"""
test_vertex.py -- Script de diagnóstico para Vertex AI o Gemini API Directa
Corre: python test_vertex.py
"""
import time
import logging
import sys
import glob
import os
from src.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("test_vertex")

# ── Inicializar Entorno ─────────────────────────────────────────────────────────
logger.info("🔬 Iniciando diagnóstico de IA...")
logger.info(f"   Modo Vertex AI: {Config.USE_VERTEX_AI}")
MODEL_NAME = "gemini-2.5-flash"

if Config.USE_VERTEX_AI:
    import vertexai
    from vertexai.generative_models import GenerativeModel, Part
    vertexai.init(project=Config.PROJECT_ID, location=Config.LOCATION)
else:
    import google.generativeai as genai
    genai.configure(api_key=Config.GEMINI_API_KEY)

# ── Funciones Helper ────────────────────────────────────────────────────────────
def get_model():
    if Config.USE_VERTEX_AI:
        return GenerativeModel(MODEL_NAME)
    else:
        return genai.GenerativeModel(MODEL_NAME)

def process_stream(response_stream, t0):
    chunks = []
    for i, chunk in enumerate(response_stream):
        if i == 0:
            logger.info(f"✅ Primer token en {time.time()-t0:.1f}s")
        chunks.append(chunk.text if hasattr(chunk, 'text') and chunk.text else "")
    full_text = "".join(chunks)
    logger.info(f"✅ ÉXITO. Respuesta en {time.time()-t0:.1f}s: {full_text[:100]}...")
    return full_text

# ── TEST 1: Texto puro (sin PDFs) ───────────────────────────────────────────────
def test_1_texto_puro():
    logger.info("=" * 60)
    logger.info("TEST 1: Texto puro - Diagnóstico básico de conectividad")
    logger.info("=" * 60)
    
    model = get_model()
    prompt = "Responde en una sola oración: ¿Cuál es la capital de México?"
    
    t0 = time.time()
    logger.info(f"⏳ Enviando petición a {MODEL_NAME}...")
    try:
        response_stream = model.generate_content(prompt, stream=True)
        process_stream(response_stream, t0)
        return True
    except Exception as e:
        logger.error(f"❌ FALLO: {e}")
        return False

# ── TEST 2: Un PDF pequeño ──────────────────────────────────────────────────────
def test_2_un_pdf(pdf_path: str):
    logger.info("=" * 60)
    logger.info(f"TEST 2: Un PDF - {pdf_path}")
    logger.info("=" * 60)
    
    model = get_model()
    prompt_text = "Resume en 3 líneas el contenido de este documento."
    uploaded_files = []
    
    try:
        if Config.USE_VERTEX_AI:
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            logger.info(f"   PDF cargado (Local): {len(pdf_bytes)/1024:.1f} KB")
            prompt_parts = [Part.from_data(data=pdf_bytes, mime_type="application/pdf"), prompt_text]
        else:
            logger.info(f"   Subiendo PDF a Gemini File API...")
            f = genai.upload_file(pdf_path, mime_type="application/pdf")
            uploaded_files.append(f)
            prompt_parts = [f, prompt_text]
        
        t0 = time.time()
        logger.info("⏳ Enviando petición con PDF...")
        response_stream = model.generate_content(prompt_parts, stream=True)
        process_stream(response_stream, t0)
        return True
    except Exception as e:
        logger.error(f"❌ FALLO: {e}")
        return False
    finally:
        # Limpieza para API de Gemini
        for f in uploaded_files:
            try:
                genai.delete_file(f.name)
                logger.info(f"   Limpieza: Archivo {f.name} eliminado de la nube.")
            except:
                pass

# ── TEST 3: Varios PDFs (simula el proceso real) ────────────────────────────────
def test_3_multiples_pdfs(pdf_paths: list):
    logger.info("=" * 60)
    logger.info(f"TEST 3: {len(pdf_paths)} PDFs - Equivalente al proceso real")
    logger.info("=" * 60)
    
    model = get_model()
    prompt_text = "Resume en 5 líneas el contenido de TODOS estos documentos."
    parts = []
    uploaded_files = []
    
    try:
        total_kb = 0
        for label, path in pdf_paths:
            if Config.USE_VERTEX_AI:
                with open(path, "rb") as f_in:
                    data = f_in.read()
                total_kb += len(data)/1024
                parts.append(Part.from_data(data=data, mime_type="application/pdf"))
                logger.info(f"   {label} (Local): {len(data)/1024:.1f} KB")
            else:
                logger.info(f"   Subiendo {label} a Gemini File API...")
                f = genai.upload_file(path, mime_type="application/pdf")
                uploaded_files.append(f)
                parts.append(f)
        
        parts.insert(0, prompt_text)
        
        t0 = time.time()
        logger.info("⏳ Enviando petición con múltiples PDFs...")
        response_stream = model.generate_content(parts, stream=True)
        process_stream(response_stream, t0)
        return True
    except Exception as e:
        logger.error(f"❌ FALLO: {type(e).__name__}: {e}")
        return False
    finally:
        # Limpieza para API de Gemini
        for f in uploaded_files:
            try:
                genai.delete_file(f.name)
                logger.info(f"   Limpieza: Archivo {f.name} eliminado de la nube.")
            except:
                pass

# ── MAIN ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    
    # TEST 1: Sin PDFs
    ok1 = test_1_texto_puro()

    if not ok1:
        logger.error("❌ Fallo en Test 1 (texto puro). Problema crítico de conectividad con la IA.")
        logger.error("   -> Verifica credenciales (API Key o Vertex) y acceso a internet.")
        sys.exit(1)

    # Buscar PDFs en fundamentos o en output para tests 2 y 3
    pdfs = glob.glob(r"fundamentos\*.pdf", recursive=True)
    if not pdfs:
        pdfs = glob.glob(r"output\grading_results\**\*.pdf", recursive=True)
        
    if not pdfs:
        logger.warning("No hay PDFs locales de prueba (en 'fundamentos/' ni en 'output/'). Solo se ejecutó Test 1.")
        sys.exit(0)

    # TEST 2: Un PDF
    test_2_un_pdf(pdfs[0])

    # TEST 3: Múltiples PDFs si hay suficientes
    if len(pdfs) >= 2:
        test_3_multiples_pdfs([(os.path.basename(p), p) for p in pdfs[:4]])
    
    logger.info("✅ Diagnóstico finalizado.")