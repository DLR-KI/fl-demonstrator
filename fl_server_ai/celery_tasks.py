from celery import Celery
from celery.signals import setup_logging
from django.conf import settings
from logging.config import dictConfig


app = Celery("fl_server")

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
# - namespace="CELERY" means all celery-related configuration keys
#   should have a `CELERY_` prefix.
app.config_from_object("django.conf:settings", namespace="CELERY")
app.conf.update(
    broker_url=settings.CACHES["default"]["LOCATION"],
    result_backend=settings.CACHES["default"]["LOCATION"],
    broker_transport_options={
        "visibility_timeout": 31540000,
        "fanout_prefix": True,
        "fanout_patterns": True,
    }
)

# Load task modules from all registered Django apps.
app.autodiscover_tasks()


# Configure logging
@setup_logging.connect
def init_loggers(*args, **kwargs):
    dictConfig(settings.LOGGING)
