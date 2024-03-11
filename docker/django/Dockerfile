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

# add user: django
RUN useradd -ms /bin/bash django
USER django
WORKDIR /home/django/app

# install dependencies
COPY --chown=django:django pyproject.toml README.md /home/django/app/
RUN mkdir fl_server fl_server_ai fl_server_api fl_server_core && \
    pip install --no-warn-script-location --extra-index-url https://download.pytorch.org/whl/cpu . && \
    rm -rf fl_server fl_server_ai fl_server_api fl_server_core pyproject.toml README.md

# copy scripts
COPY --chown=django:django ./docker/django/docker-start.sh /home/django/app/docker-start.sh

# copy fl_server
COPY --chown=django:django ./manage.py /home/django/app/manage.py
COPY --chown=django:django ./fl_server /home/django/app/fl_server
COPY --chown=django:django ./fl_server_core /home/django/app/fl_server_core
COPY --chown=django:django ./fl_server_api /home/django/app/fl_server_api
COPY --chown=django:django ./fl_server_ai /home/django/app/fl_server_ai

ENV DJANGO_SETTINGS_MODULE=fl_server.settings.production
EXPOSE 8000
VOLUME [ "./logs" ]
CMD ["./docker-start.sh"]
STOPSIGNAL SIGINT
