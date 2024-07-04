# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from django.test import TestCase

from ..models.model import clone_model
from .dummy import Dummy


class ModelTest(TestCase):

    def test_is_global_model(self):
        model = Dummy.create_model()
        self.assertTrue(model.is_global_model())
        self.assertFalse(model.is_local_model())

    def test_is_local_model(self):
        model = Dummy.create_model_update()
        self.assertFalse(model.is_global_model())
        self.assertTrue(model.is_local_model())

    def test_clone_model(self):
        model1 = Dummy.create_model()
        model2 = clone_model(model1)
        model2.description = "This Model was duplicated"
        model1.description = "This is the original"

        self.assertEqual(model1.name, model2.name)
        self.assertEqual(model2.description, "This Model was duplicated")
        self.assertEqual(model1.description, "This is the original")
        self.assertEqual(model1.owner, model2.owner)
        self.assertEqual(model1.round, model2.round)
        self.assertEqual(model1.weights, model2.weights)
        self.assertEqual(model1.preprocessing, model2.preprocessing)
