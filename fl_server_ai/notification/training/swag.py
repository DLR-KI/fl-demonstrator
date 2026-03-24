# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
# SPDX-License-Identifier: Apache-2.0

from ..notification_type import NotificationType
from .round_start import TrainingRoundStartNotification


class TrainingSWAGRoundStartNotification(TrainingRoundStartNotification):
    """
    Notification for the start of a SWAG training round.
    """

    type: NotificationType = NotificationType.SWAG_ROUND_START
    """The type of the notification."""
