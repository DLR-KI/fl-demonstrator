from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
import json
import pickle
import torch
from uuid import uuid4

from fl_server_core.tests import BASE_URL, Dummy


class mxb(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 2*x + 5


class InferenceTests(TestCase):

    def setUp(self):
        self.user = Dummy.create_user_and_authenticate(self.client)

    def test_inference_success(self):
        inp = pickle.dumps(torch.zeros(3, 3))
        training = Dummy.create_training(actor=self.user)
        input_file = SimpleUploadedFile(
            "input.pkl",
            inp,
            content_type="application/octet-stream"
        )
        response = self.client.post(
            f"{BASE_URL}/inference/",
            {"model_id": str(training.model.id), "model_input": input_file}
        )
        self.assertEqual(response.status_code, 200)

        results = pickle.loads(response.content)
        self.assertEqual({}, results["uncertainty"])
        inference = results["inference"]
        self.assertIsNotNone(inference)
        results = torch.as_tensor(inference)
        self.assertTrue(torch.all(results <= 1))
        self.assertTrue(torch.all(results >= 0))

    def test_inference_json(self):
        inp = torch.zeros(3, 3).tolist()
        training = Dummy.create_training(actor=self.user)
        response = self.client.post(
            f"{BASE_URL}/inference/",
            json.dumps({"model_id": str(training.model.id), "model_input": inp}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertEqual({}, response_json["uncertainty"])
        inference = response_json["inference"]
        self.assertIsNotNone(inference)
        results = torch.as_tensor(inference)
        self.assertTrue(torch.all(results <= 1))
        self.assertTrue(torch.all(results >= 0))

    def test_inference_with_unknown_content_type(self):
        with self.assertLogs("root", level="INFO") as cm:
            response = self.client.post(
                f"{BASE_URL}/inference/",
                {"model_id": "not important", "model_input": "not important"},
                "application/octet-stream"
            )
        self.assertEqual(cm.output, [
            "ERROR:fl.server:Unknown Content-Type 'application/octet-stream'",
            "WARNING:django.request:Unsupported Media Type: /api/inference/",
        ])
        self.assertEqual(response.status_code, 415)

    def test_model_not_exist(self):
        inp = pickle.dumps(torch.zeros(3, 3))
        Dummy.create_model()
        unused_id = uuid4()
        input_file = SimpleUploadedFile(
            "input.pkl",
            inp,
            content_type="application/octet-stream"
        )
        with self.assertLogs("root", level="WARNING") as cm:
            response = self.client.post(
                f"{BASE_URL}/inference/",
                {"model_id": unused_id, "model_input": input_file},
                # 'multipart/form-data; boundary=...' is set automatically (default)
            )
        self.assertEqual(cm.output, [
            "WARNING:django.request:Bad Request: /api/inference/",
        ])
        self.assertEqual(response.status_code, 400)
        response_json = response.json()
        self.assertIsNotNone(response_json)
        self.assertEqual(f"Model {unused_id} not found.", response_json["detail"])

    def test_model_weights_corrupted(self):
        inp = pickle.dumps(torch.zeros(3, 3))
        model = Dummy.create_broken_model()
        Dummy.create_training(model=model, actor=self.user)
        input_file = SimpleUploadedFile(
            "input.pkl",
            inp,
            content_type="application/octet-stream"
        )
        with self.assertLogs("root", level="ERROR"):
            response = self.client.post(
                f"{BASE_URL}/inference/",
                {"model_id": model.id, "model_input": input_file},
            )
        self.assertEqual(response.status_code, 500)
        response_json = response.json()
        self.assertIsNotNone(response_json)
        self.assertEqual("Unpickled object is not a torch object.", response_json["detail"])

    def test_inference_result(self):
        torch_model = mxb()
        model = Dummy.create_model(owner=self.user, weights=pickle.dumps(torch_model))
        training = Dummy.create_training(actor=self.user, model=model)
        inputs = torch.as_tensor([
            [0.9102, 1.0899, 2.0304, -0.8448],
            [2.2616, -0.2974, 0.3805, -0.9301],
            [0.4804, 0.2510,  0.2702, -0.1529],
        ])
        input_file = SimpleUploadedFile(
            "input.pkl",
            pickle.dumps(inputs),
            content_type="application/octet-stream"
        )
        response = self.client.post(
            f"{BASE_URL}/inference/",
            {"model_id": str(training.model.id), "model_input": input_file}
        )
        self.assertEqual(response.status_code, 200)

        results = pickle.loads(response.content)
        self.assertEqual({}, results["uncertainty"])
        inference = results["inference"]
        self.assertIsNotNone(inference)
        inference_tensor = torch.as_tensor(inference)
        self.assertTrue(torch.all(torch.tensor([2, 0, 0]) == inference_tensor))
