# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from os import environ

# import and define setting defaults
from .base import *  # noqa: F403  # NOSONAR


# SECRET_KEY and SECRET_KEY_FALLBACK
# https://docs.djangoproject.com/en/4.1/howto/deployment/checklist/#secret-key

SECRET_KEY = "django-insecure-*z=iw5n%b--qdim+x5b+j9=^_gq7hq)kk8@7$^(tyn@oc#_8by"  # cspell:ignore qdim


# DEBUG
# https://docs.djangoproject.com/en/4.1/howto/deployment/checklist/#debug

DEBUG = True  # if `True` celery will result in memory leaks


# ALLOWED_HOSTS
# https://docs.djangoproject.com/en/4.1/howto/deployment/checklist/#allowed-hosts

ALLOWED_HOSTS = ["*"]


# Cross-Origin Resource Sharing (CORS) configuration
# https://github.com/adamchainz/django-cors-headers#configuration

CORS_ALLOW_ALL_ORIGINS = True


# DATABASES
# https://docs.djangoproject.com/en/4.1/howto/deployment/checklist/#databases

# overwrite postgresql host default value from "db" to "localhost"
DATABASES["postgresql"]["HOST"] = environ.get("FL_POSTGRES_HOST", "localhost")  # noqa: F405

# add sqlite3 support
DATABASES["sqlite3"] = {  # noqa: F405
    "ENGINE": "django.db.backends.sqlite3",
    "NAME": "db.sqlite3",
}
DATABASES["default"] = DATABASES[environ.get("FL_DJANGO_DATABASE", "postgresql")]  # noqa: F405


# Cache
# https://docs.djangoproject.com/en/4.2/topics/cache

# overwrite redis host default value from "redis" to "localhost"
CACHES["redis"]["LOCATION"] = get_redis_location(fallback_host="localhost")  # noqa: F405


# HTTPS
# https://docs.djangoproject.com/en/4.1/howto/deployment/checklist/#https

CSRF_COOKIE_SECURE = False
SESSION_COOKIE_SECURE = True
