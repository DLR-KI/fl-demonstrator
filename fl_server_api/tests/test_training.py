import json
from unittest.mock import MagicMock, patch
from uuid import uuid4
from django.test import TestCase

from fl_server_core.tests import BASE_URL, Dummy
from fl_server_core.models.training import TrainingState


class TrainingTests(TestCase):

    def setUp(self):
        self.user = Dummy.create_user_and_authenticate(self.client)

    def test_create_training(self):
        model = Dummy.create_model(owner=self.user)
        request_body = dict(
            model_id=str(model.id),
            target_num_updates=100,
            metric_names=["accuracy", "f1_score"],
            uncertainty_method="NONE",
            aggregation_method="FedAvg"
        )
        response = self.client.post(
            f"{BASE_URL}/trainings/",
            data=json.dumps(request_body),
            content_type="application/json"
        )
        self.assertEqual(201, response.status_code)
        response_json = response.json()
        self.assertIsNotNone(response_json)
        self.assertEqual("Training created successfully!", response_json["detail"])

    def test_create_training_with_clients(self):
        model = Dummy.create_model(owner=self.user)
        clients = [Dummy.create_client(username=f"client-{n}") for n in range(3)]
        request_body = dict(
            model_id=str(model.id),
            target_num_updates=100,
            metric_names=["accuracy", "f1_score"],
            uncertainty_method="NONE",
            aggregation_method="FedAvg",
            clients=list(map(lambda c: str(c.id), clients))
        )
        response = self.client.post(
            f"{BASE_URL}/trainings/",
            data=json.dumps(request_body),
            content_type="application/json"
        )
        self.assertEqual(201, response.status_code)
        response_json = response.json()
        self.assertIsNotNone(response_json)
        self.assertEqual("Training created successfully!", response_json["detail"])

    def test_create_training_invalid_aggregation_method(self):
        model = Dummy.create_model(owner=self.user)
        request_body = dict(
            model_id=str(model.id),
            target_num_updates=100,
            metric_names=["accuracy", "f1_score"],
            uncertainty_method="NONE",
            aggregation_method="INVALID"
        )
        with self.assertLogs("root", level="WARNING"):
            response = self.client.post(
                f"{BASE_URL}/trainings/",
                data=json.dumps(request_body),
                content_type="application/json"
            )
        self.assertEqual(400, response.status_code)

    def test_create_training_not_model_owner(self):
        model = Dummy.create_model()
        request_body = dict(
            model_id=str(model.id),
            target_num_updates=100,
            metric_names=["accuracy", "f1_score"],
            uncertainty_method="NONE",
            aggregation_method="FedAvg"
        )
        with self.assertLogs("django.request", level="WARNING") as cm:
            response = self.client.post(
                f"{BASE_URL}/trainings/",
                data=json.dumps(request_body),
                content_type="application/json"
            )
        self.assertEqual(cm.output, [
            "WARNING:django.request:Forbidden: /api/trainings/",
        ])
        self.assertEqual(403, response.status_code)
        response_json = response.json()
        self.assertIsNotNone(response_json)
        self.assertEqual("You do not have permission to perform this action.", response_json["detail"])

    def test_get_trainings(self):
        # make user actor
        self.user.actor = True
        self.user.save()
        # create trainings - some related to user some not
        [Dummy.create_training() for _ in range(3)]
        trainings = [Dummy.create_training(actor=self.user) for _ in range(3)]
        # get user related trainings
        response = self.client.get(f"{BASE_URL}/trainings/")
        self.assertEqual(200, response.status_code)
        self.assertEqual("application/json", response["content-type"])
        response_json = response.json()
        self.assertEqual(len(trainings), len(response_json))
        self.assertEqual(
            sorted([str(training.id) for training in trainings]),
            sorted([training["id"] for training in response_json])
        )

    def test_get_training_good(self):
        training = Dummy.create_training(actor=self.user)
        response = self.client.get(f"{BASE_URL}/trainings/{training.id}/")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(str(training.actor.id), body["actor"])
        self.assertEqual(TrainingState.INITIAL, body["state"])
        self.assertEqual(0, body["target_num_updates"])

    def test_get_training_bad(self):
        training = Dummy.create_training()
        with self.assertLogs("root", level="WARNING"):
            response = self.client.get(f"{BASE_URL}/trainings/{training.id}/")
        self.assertEqual(response.status_code, 403)

    def test_register_clients_good(self):
        training = Dummy.create_training(actor=self.user)
        users = [str(Dummy.create_user(username=f"client{i}").id) for i in range(1, 5)]
        request_body = dict(clients=users)

        response = self.client.put(
            f"{BASE_URL}/trainings/{training.id}/clients/",
            json.dumps(request_body)
        )
        self.assertEqual(response.status_code, 202)
        body = response.json()
        self.assertEqual("Users registered as participants!", body["detail"])

    def test_register_clients_bad(self):
        training = Dummy.create_training(actor=self.user)
        users = [str(Dummy.create_user(username=f"client{i}").id) for i in range(1, 5)] + [str(uuid4())]
        request_body = dict(clients=users)
        with self.assertLogs("root", level="WARNING"):
            response = self.client.put(
                f"{BASE_URL}/trainings/{training.id}/clients/",
                json.dumps(request_body)
            )
        self.assertEqual(response.status_code, 400)
        self.assertIsNotNone(response.content)
        response_body = response.json()
        self.assertEqual("Not all provided users were found!", response_body["detail"])

    def test_remove_clients_good(self):
        training = Dummy.create_training(actor=self.user)
        users = [str(t.id) for t in training.participants.all()]
        assert users
        request_body = dict(clients=users)

        response = self.client.delete(
            f"{BASE_URL}/trainings/{training.id}/clients/",
            json.dumps(request_body)
        )
        self.assertEqual(response.status_code, 200)
        response = self.client.get(f"{BASE_URL}/trainings/{training.id}/")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(0, len(body["participants"]))

    @patch("fl_server_ai.notification.notification.send_notifications.apply_async")
    def test_start_training(self, apply_async: MagicMock):
        user = Dummy.create_user(message_endpoint="http://example.com")
        training = Dummy.create_training(actor=self.user)
        training.participants.set([user])
        training.save()
        response = self.client.post(f"{BASE_URL}/trainings/{training.id}/start/")
        self.assertEqual(response.status_code, 202)
        self.assertEqual(2, apply_async.call_count)  # TrainingStartNotification, TrainingRoundStartNotification

    def test_start_training_no_participants(self):
        training = Dummy.create_training(actor=self.user)
        training.participants.set([])
        training.save()
        with self.assertLogs("root", level="WARNING"):
            response = self.client.post(f"{BASE_URL}/trainings/{training.id}/start/")
        self.assertEqual(response.status_code, 400)
        self.assertIsNotNone(response.content)
        response_body = response.json()
        self.assertEqual("At least one participant must be registered!", response_body["detail"])

    def test_start_training_not_initial_state(self):
        user = Dummy.create_user(message_endpoint="http://example.com")
        training = Dummy.create_training(actor=self.user, state=TrainingState.ONGOING)
        training.participants.set([user])
        training.save()
        with self.assertLogs("django.request", level="WARNING") as cm:
            response = self.client.post(f"{BASE_URL}/trainings/{training.id}/start/")
        self.assertEqual(cm.output, [
            f"WARNING:django.request:Bad Request: /api/trainings/{training.id}/start/",
        ])
        self.assertEqual(response.status_code, 400)
        self.assertIsNotNone(response.content)
        response_body = response.json()
        self.assertEqual(f"Training {training.id} is not in state INITIAL!", response_body["detail"])

    def test_create_training_with_trained_model(self):
        training = Dummy.create_training(actor=self.user)
        model = training.model
        request_body = dict(
            model_id=str(model.id),
            target_num_updates=100,
            metric_names=["accuracy", "f1_score"],
            uncertainty_method="NONE",
            aggregation_method="FedAvg"
        )
        response = self.client.post(
            f"{BASE_URL}/trainings/",
            data=json.dumps(request_body),
            content_type="application/json"
        )
        self.assertEqual(201, response.status_code)
        response_json = response.json()
        self.assertIsNotNone(response_json)
        self.assertEqual("Training created successfully!", response_json["detail"])

        response = self.client.get(f"{BASE_URL}/trainings/{response_json['training_id']}/")
        self.assertEqual(200, response.status_code)
        response_json = response.json()
        self.assertIsNotNone(response_json)
        self.assertNotEqual(model.id, response_json['id'])
