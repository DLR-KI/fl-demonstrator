---
title: Home
hide: navigation
---

<!-- markdownlint-disable-next-line MD025 -->
# Federated Learning Demonstrator

This repository contains the Federated Learning Demonstrator, a comprehensive suite of tools designed for machine learning applications in a Federated context.
This server component that handles the orchestration and notification of all training participants and ensuring smooth and efficient operation.

The demonstrator further provides capabilities for diverse model aggregation methods, merging the model inputs from each participant.
It also offers model inference, enabling the generation of predictions using the trained models.
The repository also includes utilities for quantifying uncertainty, which provide a measure of the reliability of the model's predictions.

This project is the server component of the Federated Learning (FL) platform, serving as a proof of concept for the [Catena-X](https://catena-x.net/en) project.
The FL platform aims to demonstrate the potential of federated learning in a practical, real-world context.

For a comprehensive understanding of the FL platform, please refer to the official [FL platform documentation](https://dlr-ki.github.io/fl-documentation).

A complete list of all repositories relevant to the FL platform can be found [here](https://dlr-ki.github.io/fl-documentation#repositories).

## Quick Start

Along with the general [FL platform documentation](https://dlr-ki.github.io/fl-documentation), there also exists a [tutorial](https://dlr-ki.github.io/fl-documentation/tutorial) in which all components are explained using a simple example.

The simplest way to get the demonstrator up running is to use Docker in combination with Docker Compose.

```yaml title="docker-compose.yml"
version: '3.7'

services:
  web:
    image: ghcr.io/dlr-ki/fl-demonstrator-django:main
    container_name: web
    restart: always
    ports:
      - 8000:8000
    depends_on:
      - db
      - redis
    environment:
      FL_DJANGO_SECRET_KEY: django-insecure-*z=iw5n%b--qdim+x5b+j9=^_gq7hq)kk8@7$^(tyn@oc#_8by
      FL_DJANGO_ALLOWED_HOSTS: '*'
      FL_POSTGRES_HOST: db
      FL_POSTGRES_PORT: 5432
      FL_POSTGRES_DBNAME: demo
      FL_POSTGRES_USER: demo
      FL_POSTGRES_PASSWD: example
      FL_REDIS_HOST: redis

  celery:
    image: ghcr.io/dlr-ki/fl-demonstrator-celery:main
    container_name: celery
    restart: always
    depends_on:
      - redis
    environment:
      FL_REDIS_HOST: redis

  redis:
    image: redis
    container_name: redis
    restart: always

  db:
    image: postgres
    container_name: db
    restart: always
    environment:
      POSTGRES_USER: demo
      POSTGRES_PASSWORD: example
```

After successfully starting all necessary server components with `docker compose up -d` you are ready to go.
A fitting start would be the API documentation: <http://localhost:8000/api/schema/redoc>.
