# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

# https://docs.djangoproject.com/en/4.1/howto/deployment/checklist/

from os import environ

# import and define setting defaults
from .base import *  # noqa: F403  # NOSONAR


def get_secret(env_key: str, *, ensure: bool = False) -> str | None:
    """
    Get secret value from an environment variable or a file.

    This function first checks if there is an environment variable named "{env_key}_FILE".
    If such a variable exists, the function reads the secret value from the file specified by this variable.
    If the variable does not exist, the function retrieves the secret value from the environment variable
    passed by the parameter `env_key`.

    Args:
        env_key (str): The name of the environment variable to retrieve the secret value from.
        ensure (bool, optional): If `True`, the function will raise a `KeyError` if neither "{env_key}_FILE"
                                 nor "{env_key}" is set.
                                 If `False`, the function will return `None` in this case. Defaults to `False`.

    Returns:
        str | None: The secret value, or None if neither "{env_key}_FILE" nor "{env_key}" is set and ensure is `False`.

    Raises:
        KeyError: If neither "{env_key}_FILE" nor "{env_key}" is set and ensure is True.
    """
    if (secret_file := environ.get(f"{env_key}_FILE")) is not None:
        with open(secret_file) as f:
            return f.read().strip()

    if ensure:
        # let this throw an exception if neither SECRET_KEY_FILE nor SECRET_KEY is set
        return environ[env_key]

    return environ.get(env_key)


# SECRET_KEY and SECRET_KEY_FALLBACK
# https://docs.djangoproject.com/en/4.1/howto/deployment/checklist/#secret-key

SECRET_KEY: str = get_secret("FL_DJANGO_SECRET_KEY", ensure=True)  # type:ignore[assignment]
if (_secret_key_fallbacks := get_secret("FL_DJANGO_SECRET_KEY_FALLBACK")) is not None:
    SECRET_KEY_FALLBACKS = _secret_key_fallbacks.strip().splitlines()


# DEBUG
# https://docs.djangoproject.com/en/4.1/howto/deployment/checklist/#debug

DEBUG = False


# ALLOWED_HOSTS
# https://docs.djangoproject.com/en/4.1/howto/deployment/checklist/#allowed-hosts

ALLOWED_HOSTS = environ.get("FL_DJANGO_ALLOWED_HOSTS", "").strip().splitlines()


# DATABASES
# https://docs.djangoproject.com/en/4.1/howto/deployment/checklist/#databases

DATABASES["postgresql"]["PASSWORD"] = str(get_secret("FL_POSTGRES_PASSWD", ensure=True))  # noqa: F405


# EMAIL_BACKEND and related settings
# https://docs.djangoproject.com/en/4.1/howto/deployment/checklist/#email-backend-and-related-settings


# STATIC_ROOT and STATIC_URL
# https://docs.djangoproject.com/en/4.1/howto/deployment/checklist/#static-root-and-static-url

STATIC_ROOT = BASE_DIR / "static"  # noqa: F405
STATIC_URL = "static/"
ADMIN_MEDIA_PREFIX = STATIC_URL + "admin/"


# MEDIA_ROOT and MEDIA_URL
# https://docs.djangoproject.com/en/4.1/howto/deployment/checklist/#media-root-and-media-url

# MEDIA_ROOT = ""
# MEDIA_URL = "media/"


# HTTPS
# https://docs.djangoproject.com/en/4.1/howto/deployment/checklist/#https

CSRF_COOKIE_SECURE = True
SESSION_COOKIE_SECURE = True


# Performance optimizations
# https://docs.djangoproject.com/en/4.1/howto/deployment/checklist/#performance-optimizations


# Error reporting
# https://docs.djangoproject.com/en/4.1/howto/deployment/checklist/#error-reporting
