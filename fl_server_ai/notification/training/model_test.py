from ..notification_type import NotificationType
from .round_start import TrainingRoundStartNotification


class TrainingModelTestNotification(TrainingRoundStartNotification):
    type: NotificationType = NotificationType.MODEL_TEST_ROUND
