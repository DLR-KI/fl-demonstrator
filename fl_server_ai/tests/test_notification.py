from django.test import TestCase
from requests import Response
import responses
from uuid import UUID, uuid4

from fl_server_core.models.training import Training, TrainingState
from fl_server_core.models.user import NotificationReceiver
from fl_server_core.tests.dummy import Dummy

from ..exceptions import ClientNotificationRejectionException, NotificationException
from ..notification.notification import send_notification
from ..notification.training import TrainingRoundStartNotification, TrainingFinishedNotification


class NotificationTest(TestCase):

    DUMMY_MESSAGE_ENDPOINT = "http://example.com/"

    def _create_dummy_notification_receiver(
        self,
        id: UUID = uuid4(),
        message_endpoint: str = DUMMY_MESSAGE_ENDPOINT
    ) -> NotificationReceiver:
        receiver = NotificationReceiver()
        receiver.id = id
        receiver.message_endpoint = message_endpoint
        return receiver

    def test_send_notification(self):
        notification = TrainingFinishedNotification(
            receivers=[self._create_dummy_notification_receiver() for _ in range(10)],
            body=TrainingFinishedNotification.Body(uuid4()),
            training_uuid=uuid4()
        )
        with responses.RequestsMock() as mock:
            mock.add(responses.POST, NotificationTest.DUMMY_MESSAGE_ENDPOINT)
            send_notification(NotificationTest.DUMMY_MESSAGE_ENDPOINT, notification.serialize())
            self.assertEqual(len(mock.calls), 1)

    def test_send_notification_bad(self):
        notification = TrainingFinishedNotification(
            receivers=[self._create_dummy_notification_receiver(message_endpoint="ooo://example.com/")],
            body=TrainingFinishedNotification.Body(uuid4()),
            training_uuid=uuid4()
        )
        with self.assertLogs(level="WARN"), self.assertRaises(NotificationException) as cm:
            send_notification("ooo://example.com/", notification.serialize(), max_retries=0)
        self.assertTrue(isinstance(cm.exception.inner_exception, ValueError))

    def test_make_training_notification_callback_success(self):
        user = Dummy.create_client()
        training = Dummy.create_training(participants=[user])
        notification = TrainingRoundStartNotification(
            receivers=[user],
            body=TrainingRoundStartNotification.Body(1, uuid4()),
            training_uuid=training.id
        )
        with self.assertLogs(logger="fl.celery", level="DEBUG"):
            notification.callback_success(user)

    def test_make_training_notification_callback_failure_client_error(self):
        user_kaputt = Dummy.create_client()
        user_good = Dummy.create_client()
        training = Dummy.create_training(participants=[user_kaputt, user_good])
        notification = TrainingRoundStartNotification(
            receivers=[user_good, user_kaputt],
            body=TrainingRoundStartNotification.Body(1, uuid4()),
            training_uuid=training.id
        )
        response = Response()
        response.status_code = 404
        error = ClientNotificationRejectionException(
            response,
            NotificationTest.DUMMY_MESSAGE_ENDPOINT,
            notification.serialize(), 5, user_kaputt
        )
        with self.assertLogs(level="WARN"):
            notification.callback_error(error)

    def test_training_round_start_notification_callback_failure_server_error(self):
        user = Dummy.create_client()
        training = Dummy.create_training()
        notification = TrainingRoundStartNotification(
            receivers=[user],
            body=TrainingRoundStartNotification.Body(1, uuid4()),
            training_uuid=training.id
        )
        error = NotificationException(
            NotificationTest.DUMMY_MESSAGE_ENDPOINT,
            notification.serialize(), 5, user,
            ValueError("Oops, your LAN-Cable decided to die!")
        )
        with self.assertLogs(level="ERROR"):
            notification.callback_error(error)
        training = Training.objects.get(id=training.id)
        self.assertEqual(TrainingState.ERROR, training.state)
