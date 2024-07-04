# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from django.core.exceptions import ObjectDoesNotExist
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TransactionTestCase
import io
import pickle
import torch
import torchvision
from torchvision.transforms import v2 as transforms
from unittest.mock import MagicMock, patch
from uuid import uuid4

from fl_server_ai.trainer.events import SWAGRoundFinished, TrainingRoundFinished
from fl_server_api.utils import get_entity
from fl_server_core.models import GlobalModel, MeanModel, Model, SWAGModel
from fl_server_core.models.training import Training, TrainingState
from fl_server_core.tests import BASE_URL, Dummy
from fl_server_core.utils.torch_serialization import from_torch_module


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

    def test_get_all_models_for_a_training(self):
        # make user actor and client
        self.user.actor = True
        self.user.client = True
        self.user.save()
        # create participants
        participants = [Dummy.create_user() for _ in range(4)]
        participant_rounds = [3, 4, 4, 3]
        # create models and trainings - some related to user some not
        [Dummy.create_training() for _ in range(2)]
        [Dummy.create_training(actor=self.user) for _ in range(2)]
        [Dummy.create_training(participants=[self.user]) for _ in range(2)]
        [Dummy.create_model_update() for _ in range(2)]
        [Dummy.create_model_update(owner=self.user) for _ in range(2)]
        training = Dummy.create_training(actor=self.user, participants=participants)
        # create model update for 4 users
        base_model = training.model
        models = [base_model]
        for participant, rounds in zip(participants, participant_rounds):
            for round_idx in range(rounds):
                model = Dummy.create_model_update(base_model=base_model, owner=participant, round=round_idx+1)
                models.append(model)
        # get user related models for a special training
        response = self.client.get(f"{BASE_URL}/trainings/{training.pk}/models/")
        self.assertEqual(200, response.status_code)
        self.assertEqual("application/json", response["content-type"])
        response_json = response.json()
        self.assertEqual(len(models), len(response_json))
        self.assertEqual(
            sorted([str(model.id) for model in models]),
            sorted([model["id"] for model in response_json])
        )

    def test_get_all_models_for_a_training_latest_only(self):
        # make user actor and client
        self.user.actor = True
        self.user.client = True
        self.user.save()
        # create participants
        participants = [Dummy.create_user() for _ in range(4)]
        participant_rounds = [3, 4, 4, 3]
        # create models and trainings - some related to user some not
        [Dummy.create_training() for _ in range(2)]
        [Dummy.create_training(actor=self.user) for _ in range(2)]
        [Dummy.create_training(participants=[self.user]) for _ in range(2)]
        [Dummy.create_model_update() for _ in range(2)]
        [Dummy.create_model_update(owner=self.user) for _ in range(2)]
        training = Dummy.create_training(actor=self.user, participants=participants)
        # create model update for 4 users
        base_model = training.model
        models_latest = [base_model]
        for participant, rounds in zip(participants, participant_rounds):
            for round_idx in range(rounds):
                model = Dummy.create_model_update(base_model=base_model, owner=participant, round=round_idx+1)
            models_latest.append(model)
        # get user related "latest" models for a special training
        response = self.client.get(f"{BASE_URL}/trainings/{training.pk}/models/latest/")
        self.assertEqual(200, response.status_code)
        self.assertEqual("application/json", response["content-type"])
        response_json = response.json()
        self.assertEqual(len(models_latest), len(response_json))
        models_latest = sorted(models_latest, key=lambda m: str(m.pk))
        response_models = sorted(response_json, key=lambda m: m["id"])
        self.assertEqual(
            [str(model.id) for model in models_latest],
            [model["id"] for model in response_models]
        )
        self.assertEqual(
            [model.round for model in models_latest],
            [model["round"] for model in response_models]
        )

    def test_get_model_metadata(self):
        model_bytes = from_torch_module(torch.nn.Sequential(
            torch.nn.Linear(3, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, 1),
        ))
        model = Dummy.create_model(weights=model_bytes, input_shape=[None, 3])
        response = self.client.get(f"{BASE_URL}/models/{model.id}/metadata/")
        self.assertEqual(200, response.status_code)
        self.assertEqual("application/json", response["content-type"])
        response_json = response.json()
        self.assertEqual(str(model.id), response_json["id"])
        self.assertEqual(str(model.name), response_json["name"])
        self.assertEqual(str(model.description), response_json["description"])
        self.assertEqual(model.input_shape, response_json["input_shape"])
        # check stats
        stats = response_json["stats"]
        self.assertIsNotNone(stats)
        self.assertEqual([[1, 3]], stats["input_size"])
        self.assertIsNotNone(stats["total_input"])
        self.assertIsNotNone(stats["total_mult_adds"])
        self.assertIsNotNone(stats["total_output_bytes"])
        self.assertIsNotNone(stats["total_param_bytes"])
        self.assertIsNotNone(stats["total_params"])
        self.assertIsNotNone(stats["trainable_params"])
        # layer 1 stats
        layer1 = stats["summary_list"][0]
        self.assertEqual("Sequential", layer1["class_name"])
        self.assertEqual(0, layer1["depth"])
        self.assertEqual(1, layer1["depth_index"])
        self.assertEqual(True, layer1["executed"])
        self.assertEqual("Sequential", layer1["var_name"])
        self.assertEqual(False, layer1["is_leaf_layer"])
        self.assertEqual(False, layer1["contains_lazy_param"])
        self.assertEqual(False, layer1["is_recursive"])
        self.assertEqual([1, 3], layer1["input_size"])
        self.assertEqual([1, 1], layer1["output_size"])
        self.assertEqual(None, layer1["kernel_size"])
        self.assertIsNotNone(layer1["trainable_params"])
        self.assertIsNotNone(layer1["num_params"])
        self.assertIsNotNone(layer1["param_bytes"])
        self.assertIsNotNone(layer1["output_bytes"])
        self.assertIsNotNone(layer1["macs"])
        # layer 2 stats
        layer2 = stats["summary_list"][1]
        self.assertEqual("Linear", layer2["class_name"])
        self.assertEqual(1, layer2["depth"])
        self.assertEqual(1, layer2["depth_index"])
        self.assertEqual(True, layer2["executed"])
        self.assertEqual("0", layer2["var_name"])
        self.assertEqual(True, layer2["is_leaf_layer"])
        self.assertEqual(False, layer2["contains_lazy_param"])
        self.assertEqual(False, layer2["is_recursive"])
        self.assertEqual([1, 3], layer2["input_size"])
        self.assertEqual([1, 64], layer2["output_size"])
        self.assertEqual(None, layer2["kernel_size"])
        self.assertIsNotNone(layer2["trainable_params"])
        self.assertIsNotNone(layer2["num_params"])
        self.assertIsNotNone(layer2["param_bytes"])
        self.assertIsNotNone(layer2["output_bytes"])
        self.assertIsNotNone(layer2["macs"])
        # layer 3 stats
        layer3 = stats["summary_list"][2]
        self.assertEqual("ELU", layer3["class_name"])
        self.assertEqual(1, layer3["depth"])
        self.assertEqual(2, layer3["depth_index"])
        self.assertEqual(True, layer3["executed"])
        self.assertEqual("1", layer3["var_name"])
        self.assertEqual(True, layer3["is_leaf_layer"])
        self.assertEqual(False, layer3["contains_lazy_param"])
        self.assertEqual(False, layer3["is_recursive"])
        self.assertEqual([1, 64], layer3["input_size"])
        self.assertEqual([1, 64], layer3["output_size"])
        self.assertEqual(None, layer3["kernel_size"])
        self.assertIsNotNone(layer3["trainable_params"])
        self.assertIsNotNone(layer3["num_params"])
        self.assertIsNotNone(layer3["param_bytes"])
        self.assertIsNotNone(layer3["output_bytes"])
        self.assertIsNotNone(layer3["macs"])
        # layer 4 stats
        layer4 = stats["summary_list"][3]
        self.assertEqual("Linear", layer4["class_name"])
        self.assertEqual(1, layer4["depth"])
        self.assertEqual(3, layer4["depth_index"])
        self.assertEqual(True, layer4["executed"])
        self.assertEqual("2", layer4["var_name"])
        self.assertEqual(True, layer4["is_leaf_layer"])
        self.assertEqual(False, layer4["contains_lazy_param"])
        self.assertEqual(False, layer4["is_recursive"])
        self.assertEqual([1, 64], layer4["input_size"])
        self.assertEqual([1, 1], layer4["output_size"])
        self.assertEqual(None, layer4["kernel_size"])
        self.assertIsNotNone(layer4["trainable_params"])
        self.assertIsNotNone(layer4["num_params"])
        self.assertIsNotNone(layer4["param_bytes"])
        self.assertIsNotNone(layer4["output_bytes"])
        self.assertIsNotNone(layer4["macs"])

    def test_get_model_metadata_torchscript_model(self):
        torchscript_model_bytes = from_torch_module(torch.jit.script(torch.nn.Sequential(
            torch.nn.Linear(3, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, 1),
        )))
        model = Dummy.create_model(weights=torchscript_model_bytes, input_shape=[None, 3])
        response = self.client.get(f"{BASE_URL}/models/{model.id}/metadata/")
        self.assertEqual(200, response.status_code)
        self.assertEqual("application/json", response["content-type"])
        response_json = response.json()
        self.assertEqual(str(model.id), response_json["id"])
        self.assertEqual(str(model.name), response_json["name"])
        self.assertEqual(str(model.description), response_json["description"])
        self.assertEqual(model.input_shape, response_json["input_shape"])
        # check stats
        stats = response_json["stats"]
        self.assertIsNotNone(stats)
        self.assertEqual([[1, 3]], stats["input_size"])
        self.assertIsNotNone(stats["total_input"])
        self.assertIsNotNone(stats["total_mult_adds"])
        self.assertIsNotNone(stats["total_output_bytes"])
        self.assertIsNotNone(stats["total_param_bytes"])
        self.assertIsNotNone(stats["total_params"])
        self.assertIsNotNone(stats["trainable_params"])
        self.assertEqual(4, len(stats["summary_list"]))

    def test_get_model(self):
        model = Dummy.create_model(weights=b"Hello World!")
        response = self.client.get(f"{BASE_URL}/models/{model.id}/")
        self.assertEqual(200, response.status_code)
        self.assertEqual("application/octet-stream", response["content-type"])
        self.assertEqual(b"Hello World!", response.getvalue())

    def test_delete_model_without_training_as_model_owner(self):
        model = Dummy.create_model(owner=self.user)
        response = self.client.delete(f"{BASE_URL}/models/{model.id}/")
        self.assertEqual(200, response.status_code)
        body = response.json()
        self.assertEqual("Model removed!", body["detail"])
        self.assertRaises(ObjectDoesNotExist, Model.objects.get, pk=model.id)

    def test_delete_global_model_with_training_as_model_owner(self):
        model = Dummy.create_model(owner=self.user)
        training = Dummy.create_training(model=model)
        response = self.client.delete(f"{BASE_URL}/models/{model.id}/")
        self.assertEqual(200, response.status_code)
        body = response.json()
        self.assertEqual("Model removed!", body["detail"])
        self.assertRaises(ObjectDoesNotExist, Model.objects.get, pk=model.id)
        # due to cascade delete (in the case of GlobalModel), training should also be deleted
        self.assertRaises(ObjectDoesNotExist, Training.objects.get, pk=training.id)

    def test_delete_local_model_with_training_as_model_owner(self):
        global_model = Dummy.create_model()
        local_model = Dummy.create_model_update(base_model=global_model, owner=self.user)
        training = Dummy.create_training(model=global_model)
        response = self.client.delete(f"{BASE_URL}/models/{local_model.id}/")
        self.assertEqual(200, response.status_code)
        body = response.json()
        self.assertEqual("Model removed!", body["detail"])
        self.assertRaises(ObjectDoesNotExist, Model.objects.get, pk=local_model.id)
        self.assertIsNotNone(Model.objects.get(pk=global_model.id))
        self.assertIsNotNone(Training.objects.get(pk=training.id))

    def test_delete_model_as_training_owner(self):
        model = Dummy.create_model()
        training = Dummy.create_training(model=model, actor=self.user)
        response = self.client.delete(f"{BASE_URL}/models/{model.id}/")
        self.assertEqual(200, response.status_code)
        body = response.json()
        self.assertEqual("Model removed!", body["detail"])
        self.assertRaises(ObjectDoesNotExist, Model.objects.get, pk=model.id)
        # due to cascade delete (in the case of GlobalModel), training should also be deleted
        self.assertRaises(ObjectDoesNotExist, Training.objects.get, pk=training.id)

    def test_delete_model_as_training_participant(self):
        model = Dummy.create_model()
        Dummy.create_training(model=model, participants=[Dummy.create_client(), self.user, Dummy.create_client()])
        with self.assertLogs("django.request", level="WARNING"):
            response = self.client.delete(f"{BASE_URL}/models/{model.id}/")
        self.assertEqual(403, response.status_code)
        body = response.json()
        self.assertEqual(
            "You are neither the owner of the model nor the actor of the corresponding training.",
            body["detail"]
        )
        self.assertIsNotNone(Model.objects.get(pk=model.id))

    def test_delete_model_with_training_as_unrelated_user(self):
        model = Dummy.create_model()
        Dummy.create_training(model=model)
        with self.assertLogs("django.request", level="WARNING"):
            response = self.client.delete(f"{BASE_URL}/models/{model.id}/")
        self.assertEqual(403, response.status_code)
        body = response.json()
        self.assertEqual(
            "You are neither the owner of the model nor the actor of the corresponding training.",
            body["detail"]
        )
        self.assertIsNotNone(Model.objects.get(pk=model.id))

    def test_delete_model_without_training_as_unrelated_user(self):
        model = Dummy.create_model()
        with self.assertLogs("django.request", level="WARNING"):
            response = self.client.delete(f"{BASE_URL}/models/{model.id}/")
        self.assertEqual(403, response.status_code)
        body = response.json()
        self.assertEqual(
            "You are neither the owner of the model nor the actor of the corresponding training.",
            body["detail"]
        )
        self.assertIsNotNone(Model.objects.get(pk=model.id))

    def test_delete_non_existing_model(self):
        model_id = str(uuid4())
        with self.assertLogs("django.request", level="WARNING"):
            response = self.client.delete(f"{BASE_URL}/models/{model_id}/")
        self.assertEqual(400, response.status_code)
        body = response.json()
        self.assertEqual(f"Model {model_id} not found.", body["detail"])

    def test_get_model_and_unpickle(self):
        model = Dummy.create_model()
        response = self.client.get(f"{BASE_URL}/models/{model.id}/")
        self.assertEqual(200, response.status_code)
        self.assertEqual("application/octet-stream", response["content-type"])
        torch_model = torch.jit.load(io.BytesIO(response.content))
        self.assertIsNotNone(torch_model)
        self.assertTrue(isinstance(torch_model, torch.nn.Module))

    def test_upload(self):
        torch_model = torch.jit.script(torch.nn.Sequential(
            torch.nn.Linear(3, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, 1),
        ))
        model_file = SimpleUploadedFile(
            "model.pt",
            from_torch_module(torch_model),  # torchscript model
            content_type="application/octet-stream"
        )
        response = self.client.post(f"{BASE_URL}/models/", {
            "model_file": model_file,
            "name": "Test Model",
            "description": "Test Model Description - Test Model Description Test",
            "input_shape": [None, 3]
        })
        self.assertEqual(201, response.status_code)
        response_json = response.json()
        self.assertIsNotNone(response_json)
        self.assertEqual("Model Upload Accepted", response_json["detail"])
        uuid = response_json["model_id"]
        self.assertIsNotNone(uuid)
        self.assertIsNot("", uuid)
        self.assertEqual(GlobalModel, type(Model.objects.get(id=uuid)))
        self.assertEqual([None, 3], Model.objects.get(id=uuid).input_shape)

    def test_upload_swag_model(self):
        torch_model = torch.jit.script(torch.nn.Sequential(
            torch.nn.Linear(3, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, 1),
        ))
        model_file = SimpleUploadedFile(
            "model.pt",
            from_torch_module(torch_model),  # torchscript model
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

    def test_upload_with_preprocessing(self):
        torch_model = torch.jit.script(torch.nn.Sequential(
            torch.nn.Linear(3, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, 1),
        ))
        model_file = SimpleUploadedFile(
            "model.pt",
            from_torch_module(torch_model),  # torchscript model
            content_type="application/octet-stream"
        )
        torch_model_preprocessing = torch.jit.script(torch.nn.Sequential(
            transforms.Normalize(mean=(0.,), std=(1.,)),
        ))
        model_preprocessing_file = SimpleUploadedFile(
            "preprocessing.pt",
            from_torch_module(torch_model_preprocessing),  # torchscript model
            content_type="application/octet-stream"
        )
        response = self.client.post(f"{BASE_URL}/models/", {
            "model_file": model_file,
            "model_preprocessing_file": model_preprocessing_file,
            "name": "Test Model",
            "description": "Test Model Description - Test Model Description Test",
            "input_shape": [None, 3]
        })
        self.assertEqual(201, response.status_code)
        response_json = response.json()
        self.assertIsNotNone(response_json)
        self.assertEqual("Model Upload Accepted", response_json["detail"])
        uuid = response_json["model_id"]
        self.assertIsNotNone(uuid)
        self.assertIsNot("", uuid)
        self.assertEqual(GlobalModel, type(Model.objects.get(id=uuid)))
        self.assertEqual([None, 3], Model.objects.get(id=uuid).input_shape)
        model = get_entity(GlobalModel, pk=uuid)
        self.assertIsNotNone(model)
        self.assertIsNotNone(model.weights)
        self.assertIsNotNone(model.preprocessing)
        self.assertTrue(isinstance(model.get_torch_model(), torch.nn.Module))
        self.assertTrue(isinstance(model.get_preprocessing_torch_model(), torch.nn.Module))

    def test_upload_model_preprocessing(self):
        model = Dummy.create_model(owner=self.user, preprocessing=None)
        torch_model_preprocessing = torch.jit.script(torch.nn.Sequential(
            transforms.Normalize(mean=(0.,), std=(1.,)),
        ))
        model_preprocessing_file = SimpleUploadedFile(
            "preprocessing.pt",
            from_torch_module(torch_model_preprocessing),  # torchscript model
            content_type="application/octet-stream"
        )
        response = self.client.post(f"{BASE_URL}/models/{model.id}/preprocessing/", {
            "model_preprocessing_file": model_preprocessing_file,
        })
        self.assertEqual(202, response.status_code)
        response_json = response.json()
        self.assertIsNotNone(response_json)
        self.assertEqual("Proprocessing Model Upload Accepted", response_json["detail"])
        model.refresh_from_db()
        self.assertIsNotNone(model)
        self.assertIsNotNone(model.preprocessing)
        self.assertTrue(isinstance(model.get_preprocessing_torch_model(), torch.nn.Module))

    def test_upload_model_preprocessing_v1_Compose_bad(self):
        model = Dummy.create_model(owner=self.user, preprocessing=None)
        # torchvision.transforms.Compose (v1 not v2) does not inherit from torch.nn.Module!!
        torch_model_preprocessing = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.,), std=(1.,)),
        ])
        model_preprocessing_file = SimpleUploadedFile(
            "preprocessing.pt",
            from_torch_module(torch_model_preprocessing),  # (normal) transforms.Compose
            content_type="application/octet-stream"
        )
        with self.assertLogs("fl.server", level="ERROR"):  # Loaded torch object is not of expected type.
            with self.assertLogs("django.request", level="WARNING"):  # Bad Request
                response = self.client.post(f"{BASE_URL}/models/{model.id}/preprocessing/", {
                    "model_preprocessing_file": model_preprocessing_file,
                })
        self.assertEqual(400, response.status_code)
        response_json = response.json()
        self.assertIsNotNone(response_json)
        self.assertEqual(
            "Invalid preprocessing file: Loaded torch object is not of expected type.",
            response_json[0],
        )

    def test_upload_model_preprocessing_v2_Compose_good(self):
        # Maybe good now
        model = Dummy.create_model(owner=self.user, preprocessing=None)
        torch_model_preprocessing = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=(0.,), std=(1.,)),
        ])
        model_preprocessing_file = SimpleUploadedFile(
            "preprocessing.pt",
            from_torch_module(torch_model_preprocessing),  # (normal) transforms.Compose
            content_type="application/octet-stream"
        )
        response = self.client.post(f"{BASE_URL}/models/{model.id}/preprocessing/", {
            "model_preprocessing_file": model_preprocessing_file,
        })
        self.assertEqual(202, response.status_code)
        response_json = response.json()
        self.assertIsNotNone(response_json)
        self.assertEqual("Proprocessing Model Upload Accepted", response_json["detail"])
        model.refresh_from_db()
        self.assertIsNotNone(model)
        self.assertIsNotNone(model.preprocessing)
        self.assertTrue(isinstance(model.get_preprocessing_torch_model(), torch.nn.Module))

    @patch("fl_server_ai.trainer.tasks.process_trainer_task.apply_async")
    def test_upload_update(self, apply_async: MagicMock):
        model = Dummy.create_model(owner=self.user, round=0)
        Dummy.create_training(model=model, actor=self.user, state=TrainingState.ONGOING,
                              participants=[self.user, Dummy.create_user()])
        model_update_file = SimpleUploadedFile(
            "model.pt",
            from_torch_module(torch.jit.script(torch.nn.Sequential(
                torch.nn.Linear(3, 1),
                torch.nn.Sigmoid()
            ))),
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
            "model.pt",
            from_torch_module(torch.jit.script(torch.nn.Sequential(
                torch.nn.Linear(3, 1),
                torch.nn.Sigmoid()
            ))),
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
            "model.pt",
            from_torch_module(torch.jit.script(torch.nn.Sequential(
                torch.nn.Linear(3, 1),
                torch.nn.Sigmoid()
            ))),
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
            "model.pt",
            from_torch_module(torch.jit.script(torch.nn.Sequential(
                torch.nn.Linear(3, 1),
                torch.nn.Sigmoid()
            ))),
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
            "model.pt",
            from_torch_module(torch.jit.script(torch.nn.Sequential(
                torch.nn.Linear(3, 1),
                torch.nn.Sigmoid()
            ))),
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
            "model.pt",
            from_torch_module(torch.jit.script(torch.nn.Sequential(
                torch.nn.Linear(3, 1),
                torch.nn.Sigmoid()
            ))),
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
            "model.pt",
            from_torch_module(torch.jit.script(torch.nn.Sequential(
                torch.nn.Linear(3, 1),
                torch.nn.Sigmoid()
            ))),
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
            "model.pt",
            from_torch_module(torch.jit.script(torch.nn.Sequential(
                torch.nn.Linear(3, 1),
                torch.nn.Sigmoid()
            ))),
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
