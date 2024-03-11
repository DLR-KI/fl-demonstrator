"""
AI Service
"""
from fl_server_ai.celery_tasks import app as celery_app


__all__ = ["celery_app"]
