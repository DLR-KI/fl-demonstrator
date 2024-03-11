#!/bin/bash

python -m celery \
  --workdir . \
  --app fl_server_ai \
  worker \
  --loglevel "${CELERY_LOG_LEVEL:-INFO}"
