FROM python:3.10-slim

# system dependencies
RUN apt-get update && \
    apt-get install -y \
        # psycopg2 dependencies
        libpq-dev \
        # translations
        gettext \
        # git dependencies
        git \
        && \
    rm -rf /var/lib/apt/lists/*

# add user: celery
RUN useradd -ms /bin/bash celery
USER celery
WORKDIR /home/celery/app

# install dependencies
COPY --chown=celery:celery pyproject.toml README.md /home/celery/app/
RUN mkdir fl_server fl_server_ai fl_server_api fl_server_core && \
    pip install --no-warn-script-location --extra-index-url https://download.pytorch.org/whl/cpu . && \
    rm -rf fl_server fl_server_ai fl_server_api fl_server_core pyproject.toml README.md

# copy scripts
COPY --chown=celery:celery ./docker/celery/docker-start.sh /home/celery/app/docker-start.sh

# copy fl_server
COPY --chown=celery:celery ./manage.py /home/celery/app/manage.py
COPY --chown=celery:celery ./fl_server /home/celery/app/fl_server
COPY --chown=celery:celery ./fl_server_core /home/celery/app/fl_server_core
COPY --chown=celery:celery ./fl_server_api /home/celery/app/fl_server_api
COPY --chown=celery:celery ./fl_server_ai /home/celery/app/fl_server_ai

ENV DJANGO_SETTINGS_MODULE=fl_server.settings.production
VOLUME [ "./logs" ]
CMD ["./docker-start.sh"]
STOPSIGNAL SIGINT
