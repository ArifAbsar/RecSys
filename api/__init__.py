from flask import Flask
from api.state import warm_models_background
from api.routes import health_bp, jobs_bp, orchestrator_bp


def create_app() -> Flask:
    """Application factory — creates and wires the Flask app."""
    app = Flask(__name__)

    # Register all blueprints
    app.register_blueprint(health_bp)
    app.register_blueprint(jobs_bp)
    app.register_blueprint(orchestrator_bp)

    # Start warming up HuggingFace models in the background immediately
    print("[API] Flask starting immediately. Models loading in background...", flush=True)
    warm_models_background()

    return app
