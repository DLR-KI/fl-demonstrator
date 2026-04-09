# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
# SPDX-License-Identifier: Apache-2.0

from django.apps import AppConfig


class FlServerAiConfig(AppConfig):
    """
    Django AppConfig for the Federated Learning Demonstrator AI module.
    """

    name = "fl_server_ai"
    """The name of the application. This is used in Django's internals."""
    verbose_name = "Federated Learning Demonstrator AI"
    """A human-readable name for the application. This is used in Django's admin interface."""
