# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from django.test import TestCase

from fl_server_core.tests import BASE_URL, Dummy


class GroupTests(TestCase):

    def setUp(self):
        self.user = Dummy.create_user_and_authenticate(self.client)

    def test_get_all_groups(self):
        [Dummy.create_group() for _ in range(10)]
        with self.assertLogs("django.request", level="WARNING") as cm:
            response = self.client.get(f"{BASE_URL}/groups/")
        self.assertEqual(cm.output, [
            "WARNING:django.request:Forbidden: /api/groups/",
        ])
        self.assertEqual(403, response.status_code)
        self.assertEqual("application/json", response["content-type"])
        response_json = response.json()
        self.assertEqual("You are not allowed to access all groups.", response_json["detail"])

    def test_get_all_groups_as_superuser(self):
        self.user.is_superuser = True
        self.user.save()
        [Dummy.create_group() for _ in range(10)]
        response = self.client.get(f"{BASE_URL}/groups/")
        self.assertEqual(200, response.status_code)
        self.assertEqual("application/json", response["content-type"])
        response_json = response.json()
        self.assertEqual(10, len(response_json))

    def test_get_own_group(self):
        Dummy.create_group()
        group = Dummy.create_group()
        Dummy.create_group()
        self.user.groups.add(group)
        self.user.save()
        response = self.client.get(f"{BASE_URL}/groups/{group.id}/")
        self.assertEqual(200, response.status_code)
        self.assertEqual("application/json", response["content-type"])
        response_json = response.json()
        self.assertEqual(group.id, response_json["id"])

    def test_get_other_group(self):
        group = Dummy.create_group()
        with self.assertLogs("django.request", level="WARNING") as cm:
            response = self.client.get(f"{BASE_URL}/groups/{group.id}/")
        self.assertEqual(cm.output, [
            f"WARNING:django.request:Forbidden: /api/groups/{group.id}/",
        ])
        self.assertEqual(403, response.status_code)
        self.assertEqual("application/json", response["content-type"])
        response_json = response.json()
        self.assertEqual("You are not allowed to access this group.", response_json["detail"])

    def test_create_group(self):
        group = dict(name="test-group")
        response = self.client.post(f"{BASE_URL}/groups/", group)
        self.assertEqual(201, response.status_code)
        self.assertEqual("application/json", response["content-type"])
        response_json = response.json()
        self.assertEqual("test-group", response_json["name"])

    def test_create_invalid_group(self):
        group = dict(hello="test-group")
        with self.assertLogs("django.request", level="WARNING") as cm:
            response = self.client.post(f"{BASE_URL}/groups/", group)
        self.assertEqual(cm.output, [
            "WARNING:django.request:Bad Request: /api/groups/",
        ])
        self.assertEqual(400, response.status_code)
        self.assertEqual("application/json", response["content-type"])
        response_json = response.json()
        self.assertEqual("This field is required.", response_json["name"][0])

    def test_update_group(self):
        group = Dummy.create_group()
        self.user.groups.add(group)
        self.user.save()
        group_update = dict(name="group-name")
        response = self.client.put(f"{BASE_URL}/groups/{group.id}/", group_update, content_type="application/json")
        self.assertEqual(200, response.status_code)
        self.assertEqual("application/json", response["content-type"])
        response_json = response.json()
        self.assertEqual(group_update["name"], response_json["name"])

    def test_update_group_partial(self):
        group = Dummy.create_group()
        self.user.groups.add(group)
        self.user.save()
        group_update = dict(name="group-name")
        response = self.client.patch(f"{BASE_URL}/groups/{group.id}/", group_update, content_type="application/json")
        self.assertEqual(200, response.status_code)
        self.assertEqual("application/json", response["content-type"])
        response_json = response.json()
        self.assertEqual(group_update["name"], response_json["name"])

    def test_delete_group(self):
        group = Dummy.create_group()
        self.user.groups.add(group)
        self.user.save()
        response = self.client.delete(f"{BASE_URL}/groups/{group.id}/")
        self.assertEqual(204, response.status_code)
        self.assertEqual(0, len(response.content))
