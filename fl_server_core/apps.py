# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
# SPDX-License-Identifier: Apache-2.0

from django.apps import AppConfig


class FlServerCoreConfig(AppConfig):
    """
    Django AppConfig for the Federated Learning Demonstrator Core module.
    """

    name = "fl_server_core"
    """The name of the application. This is used in Django's internals."""
    verbose_name = "Federated Learning Demonstrator"
    """A human-readable name for the application. This is used in Django's admin interface."""
