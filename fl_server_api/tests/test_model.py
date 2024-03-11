from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TransactionTestCase
import pickle
import torch
from unittest.mock import MagicMock, patch

from fl_server_core.models import GlobalModel, MeanModel, Model, SWAGModel
from fl_server_core.models.training import TrainingState
from fl_server_core.tests import BASE_URL, Dummy
from fl_server_ai.trainer.events import SWAGRoundFinished, TrainingRoundFinished


class ModelTests(TransactionTestCase):

    def setUp(self):
        self.user = Dummy.create_user_and_authenticate(self.client)

    def test_unauthorized(self):
        del self.client.defaults["HTTP_AUTHORIZATION"]
        with self.assertLogs("root", level="WARNING"):
            response = self.client.post(
                f"{BASE_URL}/models/",
                {"model_file": b"Hello World!"}
            )
        self.assertEqual(401, response.status_code)
        response_json = response.json()
        self.assertIsNotNone(response_json)
        self.assertEqual("Authentication credentials were not provided.", response_json["detail"])

    def test_get_all_models(self):
        # make user actor and client
        self.user.actor = True
        self.user.client = True
        self.user.save()
        # create models and trainings - some related to user some not
        [Dummy.create_model() for _ in range(2)]
        models = [Dummy.create_model(owner=self.user) for _ in range(2)]
        [Dummy.create_training() for _ in range(2)]
        trainings = [Dummy.create_training(actor=self.user) for _ in range(2)]
        trainings += [Dummy.create_training(participants=[self.user]) for _ in range(2)]
        models += [t.model for t in trainings]
        # get user related models
        response = self.client.get(f"{BASE_URL}/models/")
        self.assertEqual(200, response.status_code)
        self.assertEqual("application/json", response["content-type"])
        response_json = response.json()
        self.assertEqual(len(models), len(response_json))
        self.assertEqual(
            sorted([str(model.id) for model in models]),
            sorted([model["id"] for model in response_json])
        )

    def test_get_model_metadata(self):
        model = Dummy.create_model(input_shape="torch.FloatTensor(None, 1, 1)")
        response = self.client.get(f"{BASE_URL}/models/{model.id}/metadata/")
        self.assertEqual(200, response.status_code)
        self.assertEqual("application/json", response["content-type"])
        response_json = response.json()
        self.assertEqual(str(model.id), response_json["id"])
        self.assertEqual(str(model.name), response_json["name"])
        self.assertEqual(str(model.description), response_json["description"])
        self.assertEqual(str(model.input_shape), response_json["input_shape"])

    def test_get_model(self):
        model = Dummy.create_model(weights=b"Hello World!")
        response = self.client.get(f"{BASE_URL}/models/{model.id}/")
        self.assertEqual(200, response.status_code)
        self.assertEqual("application/octet-stream", response["content-type"])
        self.assertEqual(b"Hello World!", response.getvalue())

    def test_get_model_and_unpickle(self):
        model = Dummy.create_model()
        response = self.client.get(f"{BASE_URL}/models/{model.id}/")
        self.assertEqual(200, response.status_code)
        self.assertEqual("application/octet-stream", response["content-type"])
        torch_model = pickle.loads(response.content)
        self.assertIsNotNone(torch_model)
        self.assertTrue(isinstance(torch_model, torch.nn.Module))

    def test_upload(self):
        torch_model = torch.nn.Sequential(
            torch.nn.Linear(3, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, 1),
        )
        model_file = SimpleUploadedFile(
            "model.pkl",
            pickle.dumps(torch_model),
            content_type="application/octet-stream"
        )
        response = self.client.post(f"{BASE_URL}/models/", {
            "model_file": model_file,
            "name": "Test Model",
            "description": "Test Model Description - Test Model Description Test",
            "input_shape": "torch.FloatTensor(None, 3)"
        })
        self.assertEqual(201, response.status_code)
        response_json = response.json()
        self.assertIsNotNone(response_json)
        self.assertEqual("Model Upload Accepted", response_json["detail"])
        uuid = response_json["model_id"]
        self.assertIsNotNone(uuid)
        self.assertIsNot("", uuid)
        self.assertEqual(GlobalModel, type(Model.objects.get(id=uuid)))
        self.assertEqual("torch.FloatTensor(None, 3)", Model.objects.get(id=uuid).input_shape)

    def test_upload_swag_model(self):
        torch_model = torch.nn.Sequential(
            torch.nn.Linear(3, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, 1),
        )
        model_file = SimpleUploadedFile(
            "model.pkl",
            pickle.dumps(torch_model),
            content_type="application/octet-stream"
        )
        response = self.client.post(f"{BASE_URL}/models/", {
            "type": "SWAG",
            "model_file": model_file,
            "name": "Test SWAG Model",
            "description": "Test SWAG Model Description - Test SWAG Model Description Test",
        })
        self.assertEqual(201, response.status_code)
        response_json = response.json()
        self.assertIsNotNone(response_json)
        self.assertEqual("Model Upload Accepted", response_json["detail"])
        uuid = response_json["model_id"]
        self.assertIsNotNone(uuid)
        self.assertIsNot("", uuid)
        self.assertEqual(SWAGModel, type(Model.objects.get(id=uuid)))

    def test_upload_mean_model(self):
        models = [Dummy.create_model(owner=self.user) for _ in range(10)]
        model_uuids = [str(m.id) for m in models]
        response = self.client.post(f"{BASE_URL}/models/", {
            "type": "MEAN",
            "name": "Test MEAN Model",
            "description": "Test MEAN Model Description - Test MEAN Model Description Test",
            "models": model_uuids,
        }, "application/json")
        self.assertEqual(201, response.status_code)
        response_json = response.json()
        self.assertIsNotNone(response_json)
        self.assertEqual("Model Upload Accepted", response_json["detail"])
        uuid = response_json["model_id"]
        self.assertIsNotNone(uuid)
        self.assertIsNot("", uuid)
        self.assertEqual(MeanModel, type(Model.objects.get(id=uuid)))

    @patch("fl_server_ai.trainer.tasks.process_trainer_task.apply_async")
    def test_upload_update(self, apply_async: MagicMock):
        model = Dummy.create_model(owner=self.user, round=0)
        Dummy.create_training(model=model, actor=self.user, state=TrainingState.ONGOING,
                              participants=[self.user, Dummy.create_user()])
        model_update_file = SimpleUploadedFile(
            "model.pkl",
            pickle.dumps(torch.nn.Sequential(
                torch.nn.Linear(3, 1),
                torch.nn.Sigmoid()
            )),
            content_type="application/octet-stream"
        )
        response = self.client.post(
            f"{BASE_URL}/models/{model.id}/",
            {"model_file": model_update_file, "round": 0,
             "sample_size": 100}
        )
        self.assertEqual(201, response.status_code)
        response_json = response.json()
        self.assertIsNotNone(response_json)
        self.assertEqual("Model Update Accepted", response_json["detail"])
        self.assertFalse(apply_async.called)

    def test_upload_update_bad_keys(self):
        model = Dummy.create_model(owner=self.user, round=0)
        model_update_file = SimpleUploadedFile(
            "model.pkl",
            pickle.dumps(torch.nn.Sequential(
                torch.nn.Linear(3, 1),
                torch.nn.Sigmoid()
            )),
            content_type="application/octet-stream"
        )
        with self.assertLogs("django.request", level="WARNING"):
            response = self.client.post(
                f"{BASE_URL}/models/{model.id}/",
                {"xXx_model_file_xXx": model_update_file, "round": 0, "sample_size": 100}
            )
        self.assertEqual(400, response.status_code)
        response_json = response.json()
        self.assertIsNotNone(response_json)
        self.assertEqual("No uploaded file 'model_file' found.", response_json["detail"])

    def test_upload_update_no_training(self):
        model = Dummy.create_model(owner=self.user, round=0)
        model_update_file = SimpleUploadedFile(
            "model.pkl",
            pickle.dumps(torch.nn.Sequential(
                torch.nn.Linear(3, 1),
                torch.nn.Sigmoid()
            )),
            content_type="application/octet-stream"
        )
        with self.assertLogs("django.request", level="WARNING"):
            response = self.client.post(
                f"{BASE_URL}/models/{model.id}/",
                {"model_file": model_update_file, "round": 0, "sample_size": 100}
            )
        self.assertEqual(404, response.status_code)
        response_json = response.json()
        self.assertIsNotNone(response_json)
        self.assertEqual(f"Model with ID {model.id} does not have a training process running", response_json["detail"])

    def test_upload_update_no_participant(self):
        self.client.defaults["HTTP_ACCEPT"] = "application/json"
        actor = Dummy.create_actor()
        model = Dummy.create_model(owner=actor, round=0)
        training = Dummy.create_training(
            model=model, actor=actor, state=TrainingState.ONGOING,
            participants=[actor, Dummy.create_client()]
        )
        model_update_file = SimpleUploadedFile(
            "model.pkl",
            pickle.dumps(torch.nn.Sequential(
                torch.nn.Linear(3, 1),
                torch.nn.Sigmoid()
            )),
            content_type="application/octet-stream"
        )
        with self.assertLogs("root", level="WARNING"):
            response = self.client.post(
                f"{BASE_URL}/models/{model.id}/",
                {"model_file": model_update_file, "round": 0,
                 "sample_size": 500}
            )
        self.assertEqual(403, response.status_code)
        response_json = response.json()
        self.assertIsNotNone(response_json)
        self.assertEqual(f"You are not a participant of training {training.id}!", response_json["detail"])

    @patch("fl_server_ai.trainer.tasks.process_trainer_task.apply_async")
    def test_upload_update_and_aggregate(self, apply_async: MagicMock):
        model = Dummy.create_model(owner=self.user, round=0)
        train = Dummy.create_training(model=model, actor=self.user, state=TrainingState.ONGOING,
                                      participants=[self.user])
        model_update_file = SimpleUploadedFile(
            "model.pkl",
            pickle.dumps(torch.nn.Sequential(
                torch.nn.Linear(3, 1),
                torch.nn.Sigmoid()
            )),
            content_type="application/octet-stream"
        )
        response = self.client.post(
            f"{BASE_URL}/models/{model.id}/",
            {"model_file": model_update_file, "round": 0,
             "sample_size": 100}
        )
        self.assertEqual(201, response.status_code)
        response_json = response.json()
        self.assertIsNotNone(response_json)
        self.assertEqual("Model Update Accepted", response_json["detail"])
        self.assertTrue(apply_async.called)
        apply_async.assert_called_once_with(
            (),
            {"training_uuid": train.id, "event_cls": TrainingRoundFinished},
            retry=False
        )

    @patch("fl_server_ai.trainer.tasks.process_trainer_task.apply_async")
    def test_upload_update_and_not_aggregate_since_training_is_locked(self, apply_async: MagicMock):
        model = Dummy.create_model(owner=self.user, round=0)
        training = Dummy.create_training(
            model=model, actor=self.user, state=TrainingState.ONGOING, participants=[self.user]
        )
        training.locked = True
        training.save()
        model_update_file = SimpleUploadedFile(
            "model.pkl",
            pickle.dumps(torch.nn.Sequential(
                torch.nn.Linear(3, 1),
                torch.nn.Sigmoid()
            )),
            content_type="application/octet-stream"
        )
        response = self.client.post(
            f"{BASE_URL}/models/{model.id}/",
            {"model_file": model_update_file, "round": 0,
             "sample_size": 100}
        )
        self.assertEqual(201, response.status_code)
        response_json = response.json()
        self.assertIsNotNone(response_json)
        self.assertEqual("Model Update Accepted", response_json["detail"])
        self.assertFalse(apply_async.called)

    def test_upload_update_with_metrics(self):
        model = Dummy.create_model(owner=self.user, round=0)
        Dummy.create_training(model=model, actor=self.user, state=TrainingState.ONGOING,
                              participants=[self.user, Dummy.create_user()])
        model_update_file = SimpleUploadedFile(
            "model.pkl",
            pickle.dumps(torch.nn.Sequential(
                torch.nn.Linear(3, 1),
                torch.nn.Sigmoid()
            )),
            content_type="application/octet-stream"
        )

        response = self.client.post(
            f"{BASE_URL}/models/{model.id}/",
            {
                "model_file": model_update_file,
                "round": 0,
                "metric_names": ["loss", "accuracy", "dummy_binary"],
                "metric_values": [1999.0, 0.12, b"Hello World!"],
                "sample_size": 50
            },
        )
        self.assertEqual(201, response.status_code)
        response_json = response.json()
        self.assertIsNotNone(response_json)
        self.assertEqual("Model Update Accepted", response_json["detail"])

    def test_upload_update_with_metrics_bad(self):
        model = Dummy.create_model(owner=self.user)
        Dummy.create_training(model=model, actor=self.user, state=TrainingState.ONGOING,
                              participants=[self.user, Dummy.create_user()])
        model_update_file = SimpleUploadedFile(
            "model.pkl",
            pickle.dumps(torch.nn.Sequential(
                torch.nn.Linear(3, 1),
                torch.nn.Sigmoid()
            )),
            content_type="application/octet-stream"
        )
        with self.assertLogs("root", level="WARNING"):
            response = self.client.post(
                f"{BASE_URL}/models/{model.id}/",
                {"model_file": model_update_file, "round": 0, "metric_names": 5,
                 "sample_size": 500}
            )
        self.assertEqual(400, response.status_code)

    def test_upload_global_model_metrics(self):
        model = Dummy.create_model(owner=self.user, round=0)
        metrics = dict(
            metric_names=["loss", "accuracy", "dummy_binary"],
            metric_values=[1999.0, 0.12, b"Hello World!"],
        )
        with self.assertLogs("fl.server", level="WARNING") as cm:
            response = self.client.post(
                f"{BASE_URL}/models/{model.id}/metrics/",
                metrics,
            )
        self.assertEqual(cm.output, [
            f"WARNING:fl.server:Global model {model.id} is not connected to any training.",
        ])
        self.assertEqual(201, response.status_code)
        response_json = response.json()
        self.assertIsNotNone(response_json)
        self.assertEqual("Model Metrics Upload Accepted", response_json["detail"])
        self.assertEqual(str(model.id), response_json["model_id"])

    def test_upload_local_model_metrics(self):
        model = Dummy.create_model_update(owner=self.user)
        metrics = dict(
            metric_names=["loss", "accuracy", "dummy_binary"],
            metric_values=[1999.0, 0.12, b"Hello World!"],
        )
        response = self.client.post(
            f"{BASE_URL}/models/{model.id}/metrics/",
            metrics,
        )
        self.assertEqual(201, response.status_code)
        response_json = response.json()
        self.assertIsNotNone(response_json)
        self.assertEqual("Model Metrics Upload Accepted", response_json["detail"])
        self.assertEqual(str(model.id), response_json["model_id"])

    def test_upload_bad_metrics(self):
        model = Dummy.create_model(owner=self.user, round=0)
        metrics = dict(
            metric_names=["loss", "accuracy", "dummy_binary"],
            metric_values=[1999.0, b"Hello World!"],
        )
        with self.assertLogs("django.request", level="WARNING"):
            response = self.client.post(
                f"{BASE_URL}/models/{model.id}/metrics/",
                metrics,
            )
        self.assertEqual(400, response.status_code)
        response_json = response.json()
        self.assertIsNotNone(response_json)
        self.assertEqual("Metric names and values must have the same length", response_json["detail"])

    @patch("fl_server_ai.trainer.tasks.process_trainer_task.apply_async")
    def test_upload_swag_stats(self, apply_async: MagicMock):
        model = Dummy.create_model(owner=self.user, round=0)
        train = Dummy.create_training(
            model=model,
            actor=self.user,
            state=TrainingState.SWAG_ROUND,
            participants=[self.user]
        )

        first_moment_file = SimpleUploadedFile(
            "first_moment.pkl",
            pickle.dumps(torch.nn.Sequential(
                torch.nn.Linear(3, 1),
                torch.nn.Sigmoid()
            ).state_dict()),
            content_type="application/octet-stream"
        )
        second_moment_file = SimpleUploadedFile(
            "second_moment.pkl",
            pickle.dumps(torch.nn.Sequential(
                torch.nn.Linear(3, 1),
                torch.nn.Sigmoid()
            ).state_dict()),
            content_type="application/octet-stream"
        )
        response = self.client.post(f"{BASE_URL}/models/{model.id}/swag/", {
            "first_moment_file": first_moment_file,
            "second_moment_file": second_moment_file,
            "sample_size": 100,
            "round": 0
        })
        self.assertEqual(201, response.status_code)
        response_json = response.json()
        self.assertIsNotNone(response_json)
        self.assertEqual("SWAG Statistic Accepted", response_json["detail"])
        self.assertTrue(apply_async.called)
        apply_async.assert_called_once_with(
            (),
            {"training_uuid": train.id, "event_cls": SWAGRoundFinished},
            retry=False
        )

    def test_get_global_model_metrics(self):
        model = Dummy.create_model(owner=self.user)
        metric = Dummy.create_metric(model=model)
        response = self.client.get(f"{BASE_URL}/models/{model.id}/metrics/")
        self.assertEqual(200, response.status_code)
        body = response.json()
        self.assertEqual(1, len(body))
        self.assertEqual(metric.value_float, body[0]["value_float"])
        self.assertEqual(metric.key, body[0]["key"])

    def test_get_local_model_metrics(self):
        model = Dummy.create_model_update(owner=self.user)
        metric = Dummy.create_metric(model=model)
        response = self.client.get(f"{BASE_URL}/models/{model.id}/metrics/")
        self.assertEqual(200, response.status_code)
        body = response.json()
        self.assertEqual(1, len(body))
        self.assertEqual(metric.value_float, body[0]["value_float"])
        self.assertEqual(metric.key, body[0]["key"])
