#!/bin/bash
#
# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
# SPDX-License-Identifier: Apache-2.0

python -m celery \
  --workdir . \
  --app fl_server_ai \
  worker \
  --loglevel "${CELERY_LOG_LEVEL:-INFO}"
