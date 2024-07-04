# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from ..notification_type import NotificationType
from .round_start import TrainingRoundStartNotification


class TrainingSWAGRoundStartNotification(TrainingRoundStartNotification):
    """
    Notification for the start of a SWAG training round.
    """

    type: NotificationType = NotificationType.SWAG_ROUND_START
    """The type of the notification."""
