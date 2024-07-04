# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from django.db import IntegrityError, transaction
from django.http import HttpRequest, JsonResponse
from logging import getLogger
import random
from rest_framework.authtoken.models import Token
from rest_framework.viewsets import ViewSet as DjangoViewSet
import torch
import torch.nn
import torchvision.transforms.v2
from uuid import uuid4

from fl_server_core.models import Training, User
from fl_server_core.models.training import TrainingState
from fl_server_core.tests.dummy import Dummy
from fl_server_core.utils.torch_serialization import from_torch_module


class DummyView(DjangoViewSet):
    """
    A ViewSet that creates dummy data for testing.
    """

    _logger = getLogger("fl.server")
    authentication_classes: list = []
    permission_classes: list = []

    def create_dummy_metrics_and_models(self, _request: HttpRequest):
        """
        Create dummy metrics, models, etc.

        This method creates

        - a dummy user,
        - a torch model,
        - a preprocessing model,
        - a model,
        - participants,
        - a training, and
        - metrics of the training.

        Args:
            _request (HttpRequest): The request.

        Returns:
            JsonResponse: A JsonResponse with the details of the created objects.
        """
        epochs = 80
        try:
            # `transaction.atomic` is important
            # otherwise Django's internal atomic transaction won't be closed if an IntegrityError is raised
            # and `User.objects.get` will raise an TransactionManagementError
            with transaction.atomic():
                user = Dummy.create_user(username="dummy-user", password="secret")
        except IntegrityError:
            self._logger.warning("Dummy User already exists")
            user = User.objects.get(username="dummy-user")

        torch_model = torch.jit.script(torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(9216, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(128, 10),
        ))
        preprocessing_model = torch.nn.Sequential(
            torchvision.transforms.v2.ToImage(),
            torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
            torchvision.transforms.v2.Resize((28, 28)),
            torchvision.transforms.v2.Grayscale(),
            torchvision.transforms.v2.Normalize((0.1307,), (0.3015,))
        )
        model = Dummy.create_model(
            owner=user,
            weights=from_torch_module(torch_model),                # torchscript model
            preprocessing=from_torch_module(preprocessing_model),  # normal torch model
            input_shape=[None, 3, 28, 28],
        )
        participants = [Dummy.create_client() for _ in range(5)]
        training = Dummy.create_training(
            state=TrainingState.COMPLETED,
            target_num_updates=epochs,
            actor=user,
            model=model,
            participants=participants,
        )
        self._fill_metrics(training, epochs)
        self.created = True

        return JsonResponse({
            "message": "Created Dummy Data in Metrics Database!",
            "training_uuid": training.id,
            "user_uuid": user.id,
            "user_credentials": {
                "username": "dummy-user",
                "password": "secret",
                "token": Token.objects.get(user=user).key,
            },
            "participants": [str(p.id) for p in training.participants.all()],
        })

    def _fill_metrics(self, training: Training, epochs: int):
        """
        Create and fill metrics of a training.

        This method creates metrics for each epoch and each participant of the training.
        The metrics include the NLLoss and the Accuracy.

        Args:
            training (Training): The training.
            epochs (int): The number of epochs.
        """
        for epoch in range(epochs):
            for client in training.participants.all():
                Dummy.create_metric(
                    model=training.model,
                    identifier=str(uuid4()).split("-")[0],
                    key="NLLoss",
                    value_float=27.354/(epoch+1) + random.randint(0, 100) / 100,
                    step=epoch,
                    reporter=client
                )
                Dummy.create_metric(
                    model=training.model,
                    identifier=str(uuid4()).split("-")[0],
                    key="Accuracy",
                    value_float=epoch/epochs + random.randint(0, 100) / 1000,
                    step=epoch,
                    reporter=client
                )
