# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from django.conf import settings
from django.urls import path
from drf_spectacular.views import SpectacularAPIView, SpectacularRedocView, SpectacularSwaggerView

from .views import Group, Inference, Model, Training, User
from .views.dummy import DummyView


urlpatterns = [
    # OpenAPI Specification and UIs
    path("schema/", SpectacularAPIView.as_view(), name="openapi"),
    path("schema/swagger-ui/", SpectacularSwaggerView.as_view(url_name="openapi"), name="swagger-ui"),
    path("schema/redoc/", SpectacularRedocView.as_view(url_name="openapi"), name="redoc"),

    # groups
    path("groups/", view=Group.as_view({"get": "list", "post": "create"}), name="groups"),
    path("groups/<int:id>/", view=Group.as_view({
        "get": "retrieve", "put": "update", "patch": "partial_update", "delete": "destroy"
    }), name="group"),
    # users
    path("users/", view=User.as_view({"get": "get_users", "post": "create_user"}), name="users"),
    path("users/myself/", view=User.as_view({"get": "get_myself"}), name="users"),
    path("users/groups/", view=User.as_view({"get": "get_user_groups"}), name="user-groups"),
    path("users/trainings/", view=User.as_view({"get": "get_user_trainings"}), name="user-trainings"),
    path("users/<str:id>/", view=User.as_view({"get": "get_user"}), name="user"),
    # models
    path("models/", view=Model.as_view({"get": "get_models", "post": "create_model"}), name="models"),
    path("models/<str:id>/", view=Model.as_view(
        {"get": "get_model", "post": "create_local_model", "delete": "remove_model"}
    ), name="model"),
    path("models/<str:id>/metadata/", view=Model.as_view({"get": "get_metadata"}), name="model-metadata"),
    path("models/<str:id>/metrics/", view=Model.as_view(
        {"get": "get_model_metrics", "post": "create_model_metrics"}
    ), name="model-metrics"),
    path("models/<str:id>/preprocessing/", view=Model.as_view(
        {"get": "get_model_proprecessing", "post": "upload_model_preprocessing"}
    ), name="model-preprocessing"),
    path("models/<str:id>/swag/", view=Model.as_view({"post": "create_swag_stats"}), name="model-swag"),
    # trainings
    path("trainings/", view=Training.as_view({"get": "get_trainings", "post": "create_training"}), name="trainings"),
    path("trainings/<str:id>/", view=Training.as_view(
        {"get": "get_training", "delete": "remove_training"}
    ), name="training"),
    path("trainings/<str:training_id>/models/", view=Model.as_view(
        {"get": "get_training_models"}
    ), name="training-models"),
    path("trainings/<str:training_id>/models/latest/", view=Model.as_view(
        {"get": "get_training_models_latest"}
    ), name="training-models-latest"),
    path("trainings/<str:id>/clients/", view=Training.as_view(
        {"put": "register_clients", "delete": "remove_clients"}
    ), name="training-clients"),
    path("trainings/<str:id>/start/", view=Training.as_view({"post": "start_training"}), name="training-start"),
    # inference
    path("inference/", view=Inference.as_view({"post": "inference"}), name="inference"),
]

if settings.DEBUG:
    urlpatterns += [
        # Dummies for testing
        path("dummy/", view=DummyView.as_view({"get": "create_dummy_metrics_and_models"}, name="dummy")),
    ]
