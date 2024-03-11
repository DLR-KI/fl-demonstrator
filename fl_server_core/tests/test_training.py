from django.test import TestCase

from .dummy import Dummy


class TrainingTest(TestCase):

    def test_target_num_updates_without_daisy_chain_period(self):
        training = Dummy.create_training(target_num_updates=42)
        self.assertEqual(42, training.target_num_updates)

    def test_target_num_updates_with_daisy_chain_period(self):
        training = Dummy.create_training(target_num_updates=42, options=dict(daisy_chain_period=5))
        self.assertEqual(42 * 5, training.target_num_updates)
