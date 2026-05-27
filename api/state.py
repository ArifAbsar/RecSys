import os
import threading
import yaml
from Mapping_Engine.embedding_matcher.embedding import Embedding
from sentence_transformers import CrossEncoder
from Mapping_Engine.Domain_loader.domain_loader import build_keyword_intent_index, build_target_registry

# --- Global Directories ---
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DOMAIN_DIR = os.path.join(BASE_DIR, "ecommerce")
KEYWORDS_PATH = os.path.join(DOMAIN_DIR, "keywords.yaml")
QUESTIONS_PATH = os.path.join(DOMAIN_DIR, "questions.yaml")
RULES_MANIFEST = os.path.join(DOMAIN_DIR, "classifier_rules.yaml")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

try:
    with open(QUESTIONS_PATH, 'r', encoding='utf-8') as f:
        questions_config = yaml.safe_load(f) or {}
    when_to_ask = questions_config.get("when_to_ask", {})
    ENTITY_THRESHOLD = float(when_to_ask.get("entity_confidence_below", 0.80))
    FIELD_THRESHOLD = float(when_to_ask.get("field_confidence_below", 0.80))
    print(f"[API State] Dynamically loaded thresholds from questions.yaml: Entity={ENTITY_THRESHOLD}, Field={FIELD_THRESHOLD}", flush=True)
except Exception as e:
    ENTITY_THRESHOLD = 0.80
    FIELD_THRESHOLD = 0.80
    print(f"[API State] Warning: Failed to parse thresholds, falling back to 0.80. Error: {e}", flush=True)

MODEL_READY = False
LOADING_STEP = "Initializing..."
TARGET_REGISTRY = []
MATCHER = None
CROSS_ENCODER = None
KEYWORD_INTENT = {}

def _load_models_background():
    """Background worker thread to warm up HuggingFace models."""
    global MODEL_READY, LOADING_STEP, TARGET_REGISTRY, MATCHER, CROSS_ENCODER, KEYWORD_INTENT
    try:
        LOADING_STEP = "Step 1/4: Building target registry from schema manifests..."
        print(f"[API State] {LOADING_STEP}", flush=True)
        schema_path = os.path.join(DOMAIN_DIR, "schema.yaml")
        metrics_path = os.path.join(DOMAIN_DIR, "metrics.yaml")
        rules_path = os.path.join(DOMAIN_DIR, "rules.yaml")
        validations_path = os.path.join(DOMAIN_DIR, "validations.yaml")
        TARGET_REGISTRY = build_target_registry(
            schema_path=schema_path,
            keywords_path=KEYWORDS_PATH,
            metrics_path=metrics_path,
            rules_path=rules_path,
            validations_path=validations_path,
            questions_path=QUESTIONS_PATH,
        )
        print(f"[API State] Registry built: {len(TARGET_REGISTRY)} target fields.", flush=True)

        LOADING_STEP = "Step 2/4: Loading bi-encoder sentence embedding model..."
        print(f"[API State] {LOADING_STEP}", flush=True)
        MATCHER = Embedding("all-MiniLM-L6-v2")

        LOADING_STEP = f"Step 3/4: Encoding {len(TARGET_REGISTRY)} target fields (CPU-heavy, 1-3 min)..."
        print(f"[API State] {LOADING_STEP}", flush=True)
        MATCHER.fit_targets(TARGET_REGISTRY)
        print("[API State] Target field embeddings ready.", flush=True)

        LOADING_STEP = "Step 4/4: Loading cross-encoder reranker model..."
        print(f"[API State] {LOADING_STEP}", flush=True)
        CROSS_ENCODER = CrossEncoder("cross-encoder/stsb-distilroberta-base")

        KEYWORD_INTENT = build_keyword_intent_index(KEYWORDS_PATH)

        MODEL_READY = True
        LOADING_STEP = "All models ready."
        print("[API State] OK - All models loaded. API is fully ready.", flush=True)
    except Exception as e:
        LOADING_STEP = f"FATAL: {e}"
        print(f"[API State] FATAL: Model loading failed: {e}", flush=True)

def warm_models_background():
    """Triggers non-blocking warmup thread."""
    t = threading.Thread(target=_load_models_background, daemon=True)
    t.start()
