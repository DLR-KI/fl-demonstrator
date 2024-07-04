# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

"""
General settings for DLR Federated Learning Demonstrator Server project.

For more information on this file, see
<https://docs.djangoproject.com/en/4.1/topics/settings>

For the full list of settings and their values, see
<https://docs.djangoproject.com/en/4.1/ref/settings> or
<https://github.com/django/django/blob/main/django/conf/global_settings.py>
"""

from dlr.ki.logging import get_default_logging_dict
from importlib import metadata
from pathlib import Path
from os import environ
from sys import argv

from fl_server_core.utils.strings import str2bool


# Build paths inside the project like this: BASE_DIR / "subdir".
BASE_DIR = Path(__file__).resolve().parent.parent.parent


# Application definition

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "polymorphic",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "corsheaders",
    "rest_framework",
    "rest_framework.authtoken",
    "fl_server_core",
    "fl_server_ai",
    "fl_server_api",
    "drf_spectacular",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "fl_server.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "fl_server.wsgi.application"

REST_FRAMEWORK = {
    "DEFAULT_SCHEMA_CLASS": "fl_server_api.openapi.CustomAutoSchema",
}

SPECTACULAR_SETTINGS = {
    "TITLE": "Federated Learning Demonstrator Server API",
    "DESCRIPTION": """
This OpenAPI Specification describes the server component of the Federated Learning Demonstrator,
a sophisticated suite of tools designed for machine learning applications within a federated context.
This server component plays a crucial role by orchestrating and notifying all training participants to ensure seamless
and efficient operation.

The complete Federated Learning (FL) demonstation platform acts as a proof of concept within the
[Catena-X](https://catena-x.net/en) project.
The primary goal is to demonstrate the feasibility and effectiveness of federated learning in real-world scenarios.

For a comprehensive understanding and further details about the Federated Learning (FL) platform,
please refer to the official [FL platform documentation](https://dlr-ki.github.io/fl-documentation).

Use this OpenAPI Specification to explore and interact with the various endpoints provided by the server component
of the Federated Learning Demonstrator.
This specification will help developers, researchers, and data scientists to integrate, extend, and utilize
federated learning capabilities effectively.
""",
    "VERSION": metadata.version("fl-demonstrator"),
    "SERVE_AUTHENTICATION": [],
    "PREPROCESSING_HOOKS": ["fl_server_api.openapi.custom_preprocessing_hook"],
}


# Database
# https://docs.djangoproject.com/en/4.1/ref/settings/#databases


DATABASES = {
    "postgresql": {
        "ENGINE": "django.db.backends.postgresql",
        "HOST": environ.get("FL_POSTGRES_HOST", "db"),
        "PORT": environ.get("FL_POSTGRES_PORT", "5432"),
        "NAME": environ.get("FL_POSTGRES_DBNAME", "demo"),
        "USER": environ.get("FL_POSTGRES_USER", "demo"),
        "PASSWORD": environ.get("FL_POSTGRES_PASSWD", "example")
    }
}
DATABASES["default"] = DATABASES["postgresql"]


# Cache
# https://docs.djangoproject.com/en/4.2/topics/cache

def get_redis_location(*, fallback_host: str = "redis") -> str:
    if (location := environ.get("FL_REDIS_LOCATION")) is not None:
        return location
    host = environ.get("FL_REDIS_HOST", fallback_host)
    post = environ.get("FL_REDIS_PORT", "6379")
    return f"redis://{host}:{post}/1"

CACHES = {  # noqa: E305
    "redis": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": get_redis_location(),
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
        }
    }
}
CACHES["default"] = CACHES["redis"]


# Password validation
# https://docs.djangoproject.com/en/4.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]


# Internationalization
# https://docs.djangoproject.com/en/4.1/topics/i18n/

LANGUAGE_CODE = "en-us"

TIME_ZONE = "Europe/Berlin"

USE_I18N = True

USE_TZ = True


# Cross-Origin Resource Sharing (CORS) configuration
# https://github.com/adamchainz/django-cors-headers#configuration

CORS_ORIGIN_WHITELIST = environ.get("CORS_ORIGIN_WHITELIST", "").strip().splitlines()

CORS_ALLOWED_ORIGIN_REGEXES = environ.get("CORS_ALLOWED_ORIGIN_REGEXES", "").strip().splitlines()

CORS_ALLOW_ALL_ORIGINS = str2bool(environ.get("CORS_ALLOW_ALL_ORIGINS", "False"))


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.1/howto/static-files/

STATIC_URL = "static/"


# Default primary key field type
# https://docs.djangoproject.com/en/4.1/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


# Substituting a custom User model
# https://docs.djangoproject.com/en/4.2/topics/auth/customizing/#substituting-a-custom-user-model

AUTH_USER_MODEL = "fl_server_core.User"


# Celery Configuration Options
# https://docs.celeryq.dev/en/stable/userguide/configuration.html

CELERY_TASK_ALWAYS_EAGER = False  # `True` tells celery to run the task in sync (for debugging)
CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_TIME_LIMIT = 30 * 60
CELERY_TIMEZONE = TIME_ZONE
CELERY_TASK_SERIALIZER = "pickle"
CELERY_RESULT_SERIALIZER = "pickle"
CELERY_ACCEPT_CONTENT = ["pickle"]
CELERY_BROKER_CONNECTION_RETRY_ON_STARTUP = True


# initializing and configuring logging
# https://docs.djangoproject.com/en/4.1/topics/logging/

LOGGING = get_default_logging_dict(str(BASE_DIR / "logs" / "server.log"))
# LOGGING["loggers"]["celery"] = LOGGING["loggers"][""]
# LOKI_SETTINGS = {
#     "level": "DEBUG",
#     "class": "logging_loki.LokiHandler",
#     "url": "http://localhost:3100/loki/api/v1/push",
#     "tags": {
#         "app": "fl.server"
#     },
#     "auth": ["admin", "admin"],
#     "version": "1",
# }
# LOGGING["handlers"]["loki"] = LOKI_SETTINGS
# LOGGING["loggers"][""]["handlers"].append("loki")
# LOGGING["loggers"]["urllib3"] = {"level": "WARNING"}


# Federated Learning Properties

MAX_RUNNING_CHILD_PROCESSES = int(environ.get("FL_MAX_CHILD_PROCESSES", 8))
SSL_FORCE = str2bool(environ.get("FL_SSL_FORCE_REQUESTS"), "test" not in argv)
