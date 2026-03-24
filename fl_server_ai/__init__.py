# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
# SPDX-License-Identifier: Apache-2.0

"""
AI Service
"""
from fl_server_ai.celery_tasks import app as celery_app


__all__ = ["celery_app"]
