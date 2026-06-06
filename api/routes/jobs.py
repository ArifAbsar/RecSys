import os

APP_ENV = os.environ.get("APP_ENV", "local").lower()

if APP_ENV == "production":
    from api.routes.jobs_aws import jobs_bp
else:
    from api.routes.jobs_local import jobs_bp

__all__ = ["jobs_bp"]
