# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from django.test import TestCase
import json
import yaml

from fl_server_core.tests import BASE_URL


class OpenApiTests(TestCase):

    def test_get_openapi(self):
        response = self.client.get(f"{BASE_URL}/schema/swagger-ui/")
        self.assertEqual(200, response.status_code)

    def test_get_redoc(self):
        response = self.client.get(f"{BASE_URL}/schema/redoc/")
        self.assertEqual(200, response.status_code)

    def test_get_yaml(self):
        response = self.client.get(f"{BASE_URL}/schema/", HTTP_ACCEPT="application/yaml")
        self.assertEqual(200, response.status_code)
        self.assertEqual("application/yaml", response["content-type"].split(";")[0].strip())
        response_yaml = yaml.load(response.content, Loader=yaml.FullLoader)
        self.assertIsNotNone(response_yaml)

    def test_get_openapi_yaml(self):
        response = self.client.get(f"{BASE_URL}/schema/", HTTP_ACCEPT="application/vnd.oai.openapi")
        self.assertEqual(200, response.status_code)
        self.assertEqual("application/vnd.oai.openapi", response["content-type"].split(";")[0].strip())
        response_yaml = yaml.load(response.content, Loader=yaml.FullLoader)
        self.assertIsNotNone(response_yaml)

    def test_get_json(self):
        response = self.client.get(f"{BASE_URL}/schema/", HTTP_ACCEPT="application/json")
        self.assertEqual(200, response.status_code)
        self.assertEqual("application/json", response["content-type"].split(";")[0].strip())
        response_json = response.json()
        self.assertIsNotNone(response_json)

    def test_get_openapi_json(self):
        response = self.client.get(f"{BASE_URL}/schema/", HTTP_ACCEPT="application/vnd.oai.openapi+json")
        self.assertEqual(200, response.status_code)
        self.assertEqual("application/vnd.oai.openapi+json", response["content-type"].split(";")[0].strip())
        response_json = json.loads(response.content)  # Django `response.json()` requires Content-Type json
        self.assertIsNotNone(response_json)
