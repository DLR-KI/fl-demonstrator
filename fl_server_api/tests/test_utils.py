# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from django.core.files.uploadedfile import SimpleUploadedFile
from django.http import HttpRequest
from django.test import TestCase
from rest_framework.exceptions import ParseError
from uuid import uuid4
import torch
from typing import Optional

from fl_server_core.tests import Dummy
from fl_server_core.models import User
from fl_server_core.utils.strings import str2bool
from fl_server_core.utils.torch_serialization import from_torch_tensor

from ..utils import get_entity, get_file, is_json


class UtilsTest(TestCase):

    def test_views_get_entity(self):
        user = Dummy.create_actor()
        user2 = get_entity(User, pk=user.id)
        self.assertIsNotNone(user2)
        self.assertEqual(user.id, user2.id)

    def test_views_get_entity_not_exists(self):
        uid = uuid4()
        with self.assertRaises(ParseError) as context:
            get_entity(User, pk=uid)
        self.assertTrue(str(context.exception).__contains__(f"User {uid} not found."))

    def test_views_get_entity_custom_error_identifier(self):
        uid = uuid4()
        error_identifier = "'hello world!'"
        with self.assertRaises(ParseError) as context:
            get_entity(User, error_identifier=error_identifier, pk=uid)
        self.assertTrue(str(context.exception).__contains__(f"User {error_identifier} not found."))

    def test_views_get_entity_custom_error_message(self):
        uid = uuid4()
        error_message = "hello world!"
        with self.assertRaises(ParseError) as context:
            get_entity(User, error_message=error_message, pk=uid)
        self.assertTrue(str(context.exception).__contains__(error_message))

    def test_views_get_file(self):
        key = "model-key"
        inp = from_torch_tensor(torch.zeros(3, 3))
        input_file = SimpleUploadedFile("input.pt", inp, content_type="application/octet-stream")
        request = HttpRequest()
        request.FILES.appendlist(key, input_file)
        file_content = get_file(request, key)
        self.assertIsNotNone(file_content)
        self.assertEqual(len(inp), len(file_content))

    def test_views_get_file_file_not_exist(self):
        key = "model-key"
        request = HttpRequest()
        request.FILES.appendlist(key, None)
        with self.assertRaises(ParseError) as context:
            get_file(request, key)
        self.assertTrue(str(context.exception).__contains__(f"No uploaded file '{key}' found."))

    def test_views_get_file_custom_validator(self):
        key = "model-key"
        inp = from_torch_tensor(torch.zeros(3, 3))
        input_file = SimpleUploadedFile("input.pt", inp, content_type="application/octet-stream")
        request = HttpRequest()
        request.FILES.appendlist(key, input_file)

        def validator(req, name, uploaded_file, file_content, **kwargs) -> Optional[str]:
            self.assertIsNotNone(req)
            self.assertEqual(key, name)
            self.assertIsNotNone(uploaded_file)
            self.assertEqual(len(inp), len(file_content))
            self.assertEqual(5, kwargs["x"])
            return None

        file_content = get_file(request, key, validator=validator, x=5)
        self.assertIsNotNone(file_content)
        self.assertEqual(len(inp), len(file_content))

    def test_views_get_file_custom_validator_error(self):
        key = "model-key"
        inp = from_torch_tensor(torch.zeros(3, 3))
        input_file = SimpleUploadedFile("input.pt", inp, content_type="application/octet-stream")
        request = HttpRequest()
        request.FILES.appendlist(key, input_file)
        error_message = "ERROR: Hello World!"
        with self.assertRaises(ParseError) as context:
            get_file(request, key, validator=lambda req, name, uploaded_file, file_content, **kwargs: error_message)
        self.assertTrue(str(context.exception).__contains__(error_message))

    def test_str2bool_true(self):
        TRUES = [True, "yes", "true", "t", "y", "1"]
        for true in TRUES:
            self.assertTrue(str2bool(true))

    def test_str2bool_false(self):
        FALSES = [False, "no", "false", "f", "n", "0"]  # cspell:ignore FALSES
        for false in FALSES:
            self.assertFalse(str2bool(false))

    def test_str2bool_error(self):
        value = "hello world!"
        with self.assertRaises(ValueError) as context:
            str2bool(value)
        self.assertTrue(str(context.exception).__contains__(f"Can not convert '{value}' to boolean."))

    def test_str2bool_fallback(self):
        self.assertTrue(str2bool("hello world!", fallback=True))
        self.assertFalse(str2bool("hello world!", fallback=False))
        self.assertTrue(str2bool(None, fallback=True))
        self.assertFalse(str2bool(None, fallback=False))

    def test_is_json_true(self):
        self.assertEqual({"hello": "world"}, is_json('{"hello": "world"}'))

    def test_is_json_false(self):
        self.assertFalse(is_json('hello world'))
