from ..notification_type import NotificationType
from .round_start import TrainingRoundStartNotification


class TrainingSWAGRoundStartNotification(TrainingRoundStartNotification):
    type: NotificationType = NotificationType.SWAG_ROUND_START
