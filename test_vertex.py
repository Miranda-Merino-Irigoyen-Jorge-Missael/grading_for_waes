"""
test_vertex.py -- Script de diagnóstico para Vertex AI
Corre: python test_vertex.py
"""
import time
import logging
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from src.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("test_vertex")

# ── Inicializar Vertex ──────────────────────────────────────────────────────────
vertexai.init(project=Config.PROJECT_ID, location=Config.LOCATION)

# ── TEST 1: Texto puro (sin PDFs) ───────────────────────────────────────────────
def test_1_texto_puro():
    logger.info("=" * 60)
    logger.info("TEST 1: Texto puro - Diagnóstico básico de conectividad")
    logger.info("=" * 60)
    model = GenerativeModel("gemini-2.0-flash-001")
    prompt = "Responde en una sola oración: ¿Cuál es la capital de México?"
    
    t0 = time.time()
    logger.info("⏳ Enviando petición...")
    chunks = []
    try:
        response_stream = model.generate_content(prompt, stream=True)
        for i, chunk in enumerate(response_stream):
            if i == 0:
                logger.info(f"✅ Primer token en {time.time()-t0:.1f}s")
            chunks.append(chunk.text if hasattr(chunk, 'text') and chunk.text else "")
        full_text = "".join(chunks)
        logger.info(f"✅ ÉXITO. Respuesta en {time.time()-t0:.1f}s: {full_text[:100]}")
        return True
    except Exception as e:
        logger.error(f"❌ FALLO: {e}")
        return False


# ── TEST 2: Un PDF pequeño ──────────────────────────────────────────────────────
def test_2_un_pdf(pdf_path: str):
    logger.info("=" * 60)
    logger.info(f"TEST 2: Un PDF - {pdf_path}")
    logger.info("=" * 60)
    model = GenerativeModel("gemini-2.0-flash-001")
    
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    logger.info(f"   PDF cargado: {len(pdf_bytes)/1024:.1f} KB")

    prompt_parts = [
        Part.from_data(data=pdf_bytes, mime_type="application/pdf"),
        "Resume en 3 líneas el contenido de este documento."
    ]
    
    t0 = time.time()
    logger.info("⏳ Enviando petición con PDF...")
    chunks = []
    try:
        response_stream = model.generate_content(prompt_parts, stream=True)
        for i, chunk in enumerate(response_stream):
            if i == 0:
                logger.info(f"✅ Primer token en {time.time()-t0:.1f}s")
            chunks.append(chunk.text if hasattr(chunk, 'text') and chunk.text else "")
        full_text = "".join(chunks)
        logger.info(f"✅ ÉXITO. Completado en {time.time()-t0:.1f}s, {len(full_text)} chars.")
        return True
    except Exception as e:
        logger.error(f"❌ FALLO: {e}")
        return False


# ── TEST 3: Varios PDFs (simula el proceso real) ────────────────────────────────
def test_3_multiples_pdfs(pdf_paths: list):
    logger.info("=" * 60)
    logger.info(f"TEST 3: {len(pdf_paths)} PDFs - Equivalente al proceso real")
    logger.info("=" * 60)
    model = GenerativeModel("gemini-2.0-flash-001")
    
    parts = []
    total_kb = 0
    for label, path in pdf_paths:
        with open(path, "rb") as f:
            data = f.read()
        total_kb += len(data)/1024
        parts.append(Part.from_data(data=data, mime_type="application/pdf"))
        logger.info(f"   {label}: {len(data)/1024:.1f} KB")
    
    logger.info(f"   Total payload: {total_kb:.1f} KB")
    parts.append("Resume en 5 líneas el contenido de TODOS estos documentos.")
    
    t0 = time.time()
    logger.info("⏳ Enviando petición con múltiples PDFs...")
    chunks = []
    try:
        response_stream = model.generate_content(parts, stream=True)
        for i, chunk in enumerate(response_stream):
            if i == 0:
                logger.info(f"✅ Primer token en {time.time()-t0:.1f}s")
            elif i % 5 == 0:
                logger.info(f"   📝 {i} chunks recibidos en {time.time()-t0:.1f}s...")
            chunks.append(chunk.text if hasattr(chunk, 'text') and chunk.text else "")
        full_text = "".join(chunks)
        logger.info(f"✅ ÉXITO. Completado en {time.time()-t0:.1f}s, {len(full_text)} chars.")
        return True
    except Exception as e:
        logger.error(f"❌ FALLO: {type(e).__name__}: {e}")
        return False


# ── MAIN ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import glob, os, sys
    
    logger.info("🔬 Iniciando diagnóstico de Vertex AI...")
    logger.info(f"   Proyecto: {Config.PROJECT_ID}")
    logger.info(f"   Región: {Config.LOCATION}")

    # TEST 1: Sin PDFs
    ok1 = test_1_texto_puro()

    if not ok1:
        logger.error("❌ Fallo en Test 1 (texto puro). Problema crítico de conectividad con Vertex.")
        logger.error("   -> Verifica credenciales y acceso a internet.")
        sys.exit(1)

    # Buscar PDFs en output para tests 2 y 3
    pdfs = glob.glob(r"output\grading_results\**\*.pdf", recursive=True)
    if not pdfs:
        # Intenta en temp si hay algo
        logger.warning("No hay PDFs locales de prueba. Solo se ejecutó Test 1.")
        sys.exit(0)

    # TEST 2: Un PDF
    test_2_un_pdf(pdfs[0])

    # TEST 3: Múltiples PDFs si hay suficientes
    if len(pdfs) >= 2:
        test_3_multiples_pdfs([(os.path.basename(p), p) for p in pdfs[:4]])
    
    logger.info("✅ Diagnóstico finalizado.")
