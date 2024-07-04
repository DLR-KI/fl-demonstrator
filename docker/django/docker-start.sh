#!/bin/bash

# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

# migration
python manage.py migrate

# TODO: collect static files
#python manage.py collectstatic --no-input --no-post-process

# run DLR FL demonstrator server
# NOTE: set insecure otherwise static file will not be served
# TODO: remove insecure flag
python manage.py runserver 0.0.0.0:8000 --insecure
