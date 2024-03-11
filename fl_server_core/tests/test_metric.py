from django.test import TestCase

from .dummy import Dummy


class MetricTest(TestCase):

    def test_value_property_for_float(self):
        value = 0.42
        metric = Dummy.create_metric(value_float=value)
        self.assertTrue(metric.is_float())
        self.assertFalse(metric.is_binary())
        self.assertEqual(value, metric.value_float)
        self.assertEqual(value, metric.value)
        self.assertEqual(None, metric.value_binary)
        value = 0.84
        metric.value_float = value
        self.assertEqual(value, metric.value_float)
        self.assertEqual(value, metric.value)
        self.assertEqual(None, metric.value_binary)
        value = 0.98
        metric.value = value
        self.assertEqual(value, metric.value_float)
        self.assertEqual(value, metric.value)
        self.assertEqual(None, metric.value_binary)
        del metric.value
        self.assertEqual(None, metric.value_float)
        self.assertEqual(None, metric.value)
        self.assertEqual(None, metric.value_binary)

    def test_value_property_for_integer(self):
        value = 42
        metric = Dummy.create_metric(value_float=value)
        self.assertTrue(metric.is_float())
        self.assertFalse(metric.is_binary())
        self.assertEqual(value, metric.value_float)
        self.assertEqual(value, metric.value)
        self.assertEqual(None, metric.value_binary)
        value = 98
        metric.value_float = value
        self.assertEqual(value, metric.value_float)
        self.assertEqual(value, metric.value)
        self.assertEqual(None, metric.value_binary)
        value = 4711
        metric.value = value
        self.assertEqual(value, metric.value_float)
        self.assertEqual(value, metric.value)
        self.assertEqual(None, metric.value_binary)
        del metric.value
        self.assertEqual(None, metric.value_float)
        self.assertEqual(None, metric.value)
        self.assertEqual(None, metric.value_binary)

    def test_value_property_for_binary(self):
        value = b"Hello World!"
        metric = Dummy.create_metric(value_binary=value)
        self.assertFalse(metric.is_float())
        self.assertTrue(metric.is_binary())
        self.assertEqual(None, metric.value_float)
        self.assertEqual(value, metric.value)
        self.assertEqual(value, metric.value_binary)
        value = b"You are looking especially amazing today!"
        metric.value_binary = value
        self.assertEqual(None, metric.value_float)
        self.assertEqual(value, metric.value)
        self.assertEqual(value, metric.value_binary)
        value = b"Chuck Norris can take a screenshot of his blue screen."
        metric.value = value
        self.assertEqual(None, metric.value_float)
        self.assertEqual(value, metric.value)
        self.assertEqual(value, metric.value_binary)
        del metric.value
        self.assertEqual(None, metric.value_float)
        self.assertEqual(None, metric.value)
        self.assertEqual(None, metric.value_binary)
