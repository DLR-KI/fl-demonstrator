#!/bin/bash

# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

python -m celery \
  --workdir . \
  --app fl_server_ai \
  worker \
  --loglevel "${CELERY_LOG_LEVEL:-INFO}"
