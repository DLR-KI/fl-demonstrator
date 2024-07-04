# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

import base64
import pickle
import random
from logging import getLogger
from typing import Type
from uuid import uuid4

import torch
from django.contrib.auth.models import Group
from django.test import Client
from faker import Faker

from fl_server_core.models import GlobalModel, LocalModel, Metric, Model, Training, User
from fl_server_core.models.training import AggregationMethod, TrainingState, UncertaintyMethod
from fl_server_core.utils.torch_serialization import from_torch_module


BASE_URL: str = "http://127.0.0.1:8000/api"


class Dummy:

    fake = Faker("en_US")
    _logger = getLogger("fl.server.core")

    @classmethod
    def create_model(cls, model_cls: Type[Model] = GlobalModel, **kwargs) -> GlobalModel:
        torchscript_model = torch.jit.script(torch.nn.Sequential(
            torch.nn.Linear(3, 1),
            torch.nn.Sigmoid()
        ))
        args = dict(
            name=f"{cls.fake.company()} Model",
            description=f"Model created for {cls.fake.catch_phrase()}.",
            round=0,
            weights=from_torch_module(torchscript_model),  # torchscript model
            input_shape=None if kwargs.__contains__("weights") else [None, 3],
        )
        args.update(kwargs)
        if "owner" not in args:
            args["owner"] = cls.create_user()
        model = model_cls.objects.create(**args)
        cls._logger.debug(f"Creating Dummy Model with id {model.id}")
        return model

    @classmethod
    def create_model_update(cls, **kwargs) -> LocalModel:
        args = dict(
            round=1,
            weights=from_torch_module(torch.jit.script(torch.nn.Sequential(
                torch.nn.Linear(3, 1),
                torch.nn.Sigmoid()
            ))),
            sample_size=10,
        )
        args.update(kwargs)
        args["base_model"] = args["base_model"] if args.__contains__("base_model") else cls.create_model()
        args["owner"] = args["owner"] if args.__contains__("owner") else cls.create_client()
        model = LocalModel.objects.create(**args)
        cls._logger.debug(f"Creating Dummy Model Update with id {model.id}")
        return model

    @classmethod
    def create_broken_model(cls, **kwargs):
        args = dict(weights=pickle.dumps("I am not a torch.nn.Module!"))
        args.update(kwargs)
        return cls.create_model(**args)

    @classmethod
    def create_training(cls, **kwargs):
        args = dict(
            state=TrainingState.INITIAL,
            target_num_updates=0,
            uncertainty_method=UncertaintyMethod.NONE,
            aggregation_method=AggregationMethod.FED_AVG,
        )
        args.update(kwargs)
        if args.__contains__("actor"):
            model_kwargs = {"owner": args["actor"]}
        else:
            args["actor"] = cls.create_actor()
            model_kwargs = {}

        args["model"] = args["model"] if args.__contains__("model") else cls.create_model(**model_kwargs)
        participants = args.pop("participants", [cls.create_client(), cls.create_client()])
        training = Training.objects.create(**args)
        cls._logger.debug(f"Creating Dummy Training with id {training.id}")
        for participant in participants:
            training.participants.add(participant)
        training.save()
        return training

    @classmethod
    def _create_user(cls, **kwargs) -> User:
        user = User.objects.create(**kwargs)
        user.set_password(kwargs["password"])
        user.save()
        cls._logger.debug(f"Creating Dummy User with id {user.id}")
        return user

    @classmethod
    def create_user(cls, **kwargs):
        args = dict(
            message_endpoint="https://" + cls.fake.safe_email().replace("@", "."),
            actor=False,
            client=False,
            username=cls.fake.user_name(),
            first_name=cls.fake.first_name(),
            last_name=cls.fake.last_name(),
            email=cls.fake.safe_email(),
            password="secret",
        )
        args.update(kwargs)
        return cls._create_user(**args)

    @classmethod
    def create_actor(cls, **kwargs):
        kwargs["actor"] = True
        kwargs["client"] = False
        return cls.create_user(**kwargs)

    @classmethod
    def create_client(cls, **kwargs):
        kwargs["actor"] = False
        kwargs["client"] = True
        return cls.create_user(**kwargs)

    @classmethod
    def create_user_and_authenticate(cls, client: Client, **kwargs):
        args = dict(
            username=cls.fake.user_name(),
            password="secret",
        )
        args.update(kwargs)
        credentials = base64.b64encode(
            f'{args["username"]}:{args["password"]}'.encode("utf-8")
        ).decode("utf-8")
        client.defaults["HTTP_AUTHORIZATION"] = "Basic " + credentials
        return cls._create_user(**args)

    @classmethod
    def create_group(cls, **kwargs):
        args = dict(
            name=cls.fake.catch_phrase(),
        )
        args.update(kwargs)
        group = Group.objects.create(**args)
        cls._logger.debug(f"Creating Dummy Group with id {group.id}")
        return group

    @classmethod
    def create_metric(cls, **kwargs):
        epoch = random.randint(0, 100)
        args = dict(
            identifier=str(uuid4()).split("-")[0],
            key="NLLoss",
            step=epoch,
        )
        args.update(kwargs)
        if not args.__contains__("value_float") and not args.__contains__("value_binary"):
            args["value_float"] = 27.354/(epoch+1) + random.randint(0, 100) / 100
        args["model"] = args["model"] if args.__contains__("model") else cls.create_model()
        args["reporter"] = args["reporter"] if args.__contains__("reporter") else cls.create_client()
        metric = Metric.objects.create(**args)
        cls._logger.debug(f"Creating Dummy Metric with id {metric.id}")
        return metric
