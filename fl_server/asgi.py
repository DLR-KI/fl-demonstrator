# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

"""
ASGI config for fl_server project.

It exposes the ASGI callable as a module-level variable named `application`.

For more information on this file, see
<https://docs.djangoproject.com/en/4.2/howto/deployment/asgi>
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "fl_server.settings.development")

application = get_asgi_application()
