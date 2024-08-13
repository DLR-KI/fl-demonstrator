# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

import base64
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
import json
import io
import pickle
import torch
import torch.nn
from torchvision.transforms.functional import to_pil_image
from uuid import uuid4

from fl_server_core.tests import BASE_URL, Dummy
from fl_server_core.utils.torch_serialization import from_torch_module, from_torch_tensor


class mxb(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 2*x + 5


class InferenceTests(TestCase):

    def setUp(self):
        self.user = Dummy.create_user_and_authenticate(self.client)

    def test_inference_success(self):
        inp = from_torch_tensor(torch.zeros(3, 3))
        training = Dummy.create_training(actor=self.user)
        input_file = SimpleUploadedFile(
            "input.pt",
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

    def test_inference_json_binary_output(self):
        inp = torch.zeros(3, 3).tolist()
        training = Dummy.create_training(actor=self.user)
        response = self.client.post(
            f"{BASE_URL}/inference/",
            json.dumps({"model_id": str(training.model.id), "model_input": inp, "return_format": "binary"}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        results = pickle.loads(response.content)
        self.assertEqual({}, results["uncertainty"])
        inference = results["inference"]
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
        inp = from_torch_tensor(torch.zeros(3, 3))
        Dummy.create_model()
        unused_id = uuid4()
        input_file = SimpleUploadedFile(
            "input.pt",
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
        inp = from_torch_tensor(torch.zeros(3, 3))
        model = Dummy.create_broken_model()
        Dummy.create_training(model=model, actor=self.user)
        input_file = SimpleUploadedFile(
            "input.pt",
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
        self.assertEqual("Error loading torch object", response_json["detail"])

    def test_inference_result_torchscript_model(self):
        torch_model = torch.jit.script(mxb())  # torchscript model
        self._inference_result(torch_model)

    def test_inference_result_normal_model(self):
        torch_model = mxb()  # normal model
        self._inference_result(torch_model)

    def _inference_result(self, torch_model: torch.nn.Module):
        model = Dummy.create_model(owner=self.user, weights=from_torch_module(torch_model))
        training = Dummy.create_training(actor=self.user, model=model)
        inputs = torch.as_tensor([
            [0.9102, 1.0899, 2.0304, -0.8448],
            [2.2616, -0.2974, 0.3805, -0.9301],
            [0.4804, 0.2510,  0.2702, -0.1529],
        ])
        input_file = SimpleUploadedFile(
            "input.pt",
            from_torch_tensor(inputs),
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

    def test_inference_input_shape_positive(self):
        inp = from_torch_tensor(torch.zeros(3, 3))
        model = Dummy.create_model(input_shape=[None, 3])
        training = Dummy.create_training(actor=self.user, model=model)
        input_file = SimpleUploadedFile(
            "input.pt",
            inp,
            content_type="application/octet-stream"
        )
        response = self.client.post(
            f"{BASE_URL}/inference/",
            {"model_id": str(training.model.id), "model_input": input_file}
        )
        self.assertEqual(response.status_code, 200)

    def test_inference_input_shape_negative(self):
        inp = from_torch_tensor(torch.zeros(3, 3))
        model = Dummy.create_model(input_shape=[None, 5])
        training = Dummy.create_training(actor=self.user, model=model)
        input_file = SimpleUploadedFile(
            "input.pt",
            inp,
            content_type="application/octet-stream"
        )
        with self.assertLogs("root", level="WARNING") as cm:
            response = self.client.post(
                f"{BASE_URL}/inference/",
                {"model_id": str(training.model.id), "model_input": input_file}
            )
        self.assertEqual(cm.output, [
            "WARNING:django.request:Bad Request: /api/inference/",
        ])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()[0], "Input shape does not match model input shape.")

    def test_inference_input_pil_image(self):
        img = to_pil_image(torch.zeros(1, 5, 5))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="jpeg")
        img_byte_arr = img_byte_arr.getvalue()

        torch.manual_seed(42)
        torch_model = torch.jit.script(torch.nn.Sequential(
            torch.nn.Conv2d(1, 2, 3),
            torch.nn.Flatten(),
            torch.nn.Linear(3*3, 2)
        ))
        model = Dummy.create_model(input_shape=[None, 5, 5], weights=from_torch_module(torch_model))
        training = Dummy.create_training(actor=self.user, model=model)
        input_file = SimpleUploadedFile(
            "input.pt",
            img_byte_arr,
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
        self.assertTrue(torch.all(torch.tensor([0, 0]) == inference_tensor))

    def test_inference_input_pil_image_base64(self):
        img = to_pil_image(torch.zeros(1, 5, 5))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="jpeg")
        img_byte_arr = img_byte_arr.getvalue()
        inp = base64.b64encode(img_byte_arr)

        torch.manual_seed(42)
        torch_model = torch.jit.script(torch.nn.Sequential(
            torch.nn.Conv2d(1, 2, 3),
            torch.nn.Flatten(),
            torch.nn.Linear(3*3, 2)
        ))
        model = Dummy.create_model(input_shape=[None, 5, 5], weights=from_torch_module(torch_model))
        training = Dummy.create_training(actor=self.user, model=model)
        input_file = SimpleUploadedFile(
            "input.pt",
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
        inference_tensor = torch.as_tensor(inference)
        self.assertTrue(torch.all(torch.tensor([0, 0]) == inference_tensor))
