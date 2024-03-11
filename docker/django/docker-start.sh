#!/bin/bash

# migration
python manage.py migrate

# TODO: collect static files
#python manage.py collectstatic --no-input --no-post-process

# run DLR FL demonstrator server
# NOTE: set insecure otherwise static file will not be served
# TODO: remove insecure flag
python manage.py runserver 0.0.0.0:8000 --insecure
