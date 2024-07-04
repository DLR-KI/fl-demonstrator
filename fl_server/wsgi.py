# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

"""
WSGI config for fl_server project.

It exposes the WSGI callable as a module-level variable named `application`.

For more information on this file, see
<https://docs.djangoproject.com/en/4.2/howto/deployment/wsgi>
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "fl_server.settings.development")

application = get_wsgi_application()
