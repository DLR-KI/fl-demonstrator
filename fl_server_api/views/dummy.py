from django.db import IntegrityError, transaction
from django.http import HttpRequest, JsonResponse
from logging import getLogger
import random
from rest_framework.authtoken.models import Token
from rest_framework.viewsets import ViewSet as DjangoViewSet
from uuid import uuid4

from fl_server_core.models import Training, User
from fl_server_core.models.training import TrainingState
from fl_server_core.tests.dummy import Dummy


class DummyView(DjangoViewSet):
    _logger = getLogger("fl.server")
    authentication_classes: list = []
    permission_classes: list = []

    def create_dummy_metrics_and_models(self, _request: HttpRequest):
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

        model = Dummy.create_model(owner=user)
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
