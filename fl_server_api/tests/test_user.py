import base64
from django.test import TestCase
from rest_framework.authtoken.models import Token
from typing import Any, Dict, Optional, Union
from uuid import uuid4

from fl_server_core.tests import BASE_URL, Dummy
from fl_server_core.models import User

from .utils import parse


class UserTests(TestCase):

    def assertUserEqual(
        self,
        expected: Union[User, Dict],
        actual: Union[User, Dict],
        *,
        include_id_check: Optional[bool] = True
    ) -> None:
        expected_obj: Any = parse(expected) if isinstance(expected, Dict) else expected
        actual_obj: Any = parse(actual) if isinstance(actual, Dict) else actual
        # user model
        if include_id_check:
            self.assertEqual(str(expected_obj.id), str(actual_obj.id))
        self.assertEqual(bool(expected_obj.actor), bool(actual_obj.actor))
        self.assertEqual(bool(expected_obj.client), bool(actual_obj.client))
        self.assertEqual(str(expected_obj.message_endpoint), str(actual_obj.message_endpoint))
        # auth user model
        self.assertEqual(str(expected_obj.username), str(actual_obj.username))
        self.assertEqual(str(expected_obj.first_name), str(actual_obj.first_name))
        self.assertEqual(str(expected_obj.last_name), str(actual_obj.last_name))
        self.assertEqual(str(expected_obj.email), str(actual_obj.email))

    def test_get_user_via_auth(self):
        user = Dummy.create_user_and_authenticate(self.client)
        response = self.client.get(f"{BASE_URL}/users/")
        self.assertEqual(200, response.status_code)
        self.assertEqual("application/json", response["content-type"])
        response_json = response.json()
        self.assertEqual(1, len(response_json))
        self.assertUserEqual(user, response_json[0])

    def test_get_user_via_url_token(self):
        user = Dummy.create_user()
        token = Token.objects.get(user=user).key
        credentials = base64.b64encode(token.encode("utf-8")).decode("utf-8")
        self.client.defaults["HTTP_AUTHORIZATION"] = "Basic " + credentials
        # Note: Django self.client does not support: http://username:password@localhost:8000/api/...
        response = self.client.get(f"{BASE_URL}/users/")
        self.assertEqual(200, response.status_code)
        self.assertEqual("application/json", response["content-type"])
        response_json = response.json()
        self.assertEqual(1, len(response_json))
        self.assertUserEqual(user, response_json[0])

    def test_get_user_via_auth_token(self):
        user = Dummy.create_user()
        token = Token.objects.get(user=user).key
        self.client.defaults["HTTP_AUTHORIZATION"] = "Token " + token
        response = self.client.get(f"{BASE_URL}/users/")
        self.assertEqual(200, response.status_code)
        self.assertEqual("application/json", response["content-type"])
        response_json = response.json()
        self.assertEqual(1, len(response_json))
        self.assertUserEqual(user, response_json[0])

    def test_get_user_via_auth_unauthenticate(self):
        with self.assertLogs("django.request", level="WARNING") as cm:
            response = self.client.get(f"{BASE_URL}/users/")
        self.assertEqual(cm.output, [
            "WARNING:django.request:Unauthorized: /api/users/",
        ])
        self.assertEqual(401, response.status_code)
        self.assertEqual("application/json", response["content-type"])
        response_json = response.json()
        self.assertEqual("Authentication credentials were not provided.", response_json["detail"])

    def test_get_user_via_uuid(self):
        user = Dummy.create_user_and_authenticate(self.client)
        response = self.client.get(f"{BASE_URL}/users/{user.id}/")
        self.assertEqual(200, response.status_code)
        self.assertEqual("application/json", response["content-type"])
        response_json = response.json()
        self.assertUserEqual(user, response_json)

    def test_get_user_via_uuid_unauthenticate(self):
        user = Dummy.create_user()
        with self.assertLogs("django.request", level="WARNING") as cm:
            response = self.client.get(f"{BASE_URL}/users/{user.id}/")
        self.assertEqual(cm.output, [
            f"WARNING:django.request:Unauthorized: /api/users/{user.id}/",
        ])
        self.assertEqual(401, response.status_code)
        self.assertEqual("application/json", response["content-type"])
        response_json = response.json()
        self.assertEqual("Authentication credentials were not provided.", response_json["detail"])

    def test_create_user(self):
        user = dict(
            # user model
            actor=False,
            client=True,
            message_endpoint="https://example.com",
            # auth user model
            username=str(uuid4()).split("-")[0],
            first_name="Jane",
            last_name="Doe",
            email="jane.doe@example.com",
            password="secret",
        )
        response = self.client.post(f"{BASE_URL}/users/", user, content_type="application/json")
        self.assertEqual(201, response.status_code)
        self.assertEqual("application/json", response["content-type"])
        response_json = response.json()
        self.assertUserEqual(user, response_json, include_id_check=False)

    def test_get_user_groups(self):
        user = Dummy.create_user_and_authenticate(self.client)
        [Dummy.create_group() for _ in range(3)]
        groups = [Dummy.create_group() for _ in range(3)]
        [user.groups.add(group) for group in groups]
        response = self.client.get(f"{BASE_URL}/users/groups/")
        self.assertEqual(200, response.status_code)
        self.assertEqual("application/json", response["content-type"])
        response_json = response.json()
        self.assertEqual(3, len(response_json))
        self.assertEqual(
            sorted([group.id for group in groups]),
            sorted([group["id"] for group in response_json])
        )

    def test_get_user_trainings(self):
        user = Dummy.create_user_and_authenticate(self.client)
        [Dummy.create_training() for _ in range(3)]
        trainings = [Dummy.create_training(actor=user) for _ in range(3)]
        response = self.client.get(f"{BASE_URL}/users/trainings/")
        self.assertEqual(200, response.status_code)
        self.assertEqual("application/json", response["content-type"])
        response_json = response.json()
        self.assertEqual(3, len(response_json))
        self.assertEqual(
            sorted([str(training.id) for training in trainings]),
            sorted([training["id"] for training in response_json])
        )
