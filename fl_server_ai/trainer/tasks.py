from celery.utils.log import get_task_logger
from django.db import transaction, DatabaseError
from logging import getLogger
from traceback import format_exception
from typing import Type
from uuid import UUID

from fl_server_core.models import Training

from ..celery_tasks import app

from .events import ModelTrainerEvent
from .model_trainer import ModelTrainer


@app.task(bind=False, ignore_result=False)
def process_trainer_task(training_uuid: UUID, event_cls: Type[ModelTrainerEvent]):
    logger = get_task_logger("fl.celery")
    try:
        training = Training.objects.get(id=training_uuid)
        ModelTrainer(training).handle_cls(event_cls)
    except Exception as e:
        error_msg = f"Exception occurred for training {training_uuid}: {e}"
        logger.error(error_msg)
        logger.debug(error_msg + "\n" + "".join(format_exception(e)))
        raise e
    finally:
        logger.info(f"Unlocking training {training_uuid}")
        if training:
            training = Training.objects.get(id=training_uuid)
            training.locked = False
            training.save()


def dispatch_trainer_task(training: Training, event_cls: Type[ModelTrainerEvent], lock_training: bool):
    logger = getLogger("fl.server")
    if lock_training:
        try:
            with transaction.atomic():
                training.refresh_from_db()
                assert not training.locked

                # set lock and do the aggregation
                training.locked = True
                training.save()

            logger.debug(f"Locking training {training.id}")
        except (DatabaseError, AssertionError):
            logger.debug(f"Training {training.id} is locked!")
            return

    # start task async
    process_trainer_task.s(training_uuid=training.id, event_cls=event_cls).apply_async(retry=False)
