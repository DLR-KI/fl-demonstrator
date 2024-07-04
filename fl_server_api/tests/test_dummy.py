# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from django.test import TestCase
import json

from ..views.dummy import DummyView


class DummyTests(TestCase):

    def test_create_dummy_metrics_and_models(self):
        response = DummyView().create_dummy_metrics_and_models(_request=None)
        self.assertEqual(200, response.status_code)
        response_json = json.loads(response.content)
        self.assertEqual("Created Dummy Data in Metrics Database!", response_json["message"])

    def test_create_dummy_metrics_and_models_twice(self):
        response = DummyView().create_dummy_metrics_and_models(_request=None)
        self.assertEqual(200, response.status_code)
        response_json = json.loads(response.content)
        self.assertEqual("Created Dummy Data in Metrics Database!", response_json["message"])
        # second time
        with self.assertLogs("fl.server", level="WARNING") as cm:
            response = DummyView().create_dummy_metrics_and_models(_request=None)
        self.assertEqual(cm.output, [
            "WARNING:fl.server:Dummy User already exists",
        ])
        self.assertEqual(200, response.status_code)
        response_json = json.loads(response.content)
        self.assertEqual("Created Dummy Data in Metrics Database!", response_json["message"])
