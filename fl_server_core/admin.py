# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from polymorphic.admin import PolymorphicParentModelAdmin, PolymorphicChildModelAdmin, PolymorphicChildModelFilter

from .models import (
    GlobalModel,
    LocalModel,
    Metric,
    Model,
    Training,
    User,
)


admin.site.register(User, UserAdmin)
admin.site.register(Training)
admin.site.register(Metric)


# Flexible and customizable registration

# @admin.register(ModelUpdate)
# class ModelUpdateAdmin(admin.ModelAdmin):
#     pass


class ModelChildAdmin(PolymorphicChildModelAdmin):
    """
    Polymorphic admin model base class.
    """
    base_model = Model


@admin.register(GlobalModel)
class GlobalModelAdmin(ModelChildAdmin):
    """
    Admin interface for the `GlobalModel`.
    """
    base_model = GlobalModel


@admin.register(LocalModel)
class LocalModelAdmin(ModelChildAdmin):
    """
    Admin interface for the `LocalModel`.
    """
    base_model = LocalModel


@admin.register(Model)
class ModelParentAdmin(PolymorphicParentModelAdmin):
    """
    Admin interface for the parent Model class.

    This includes support for `GlobalModel` as well as `LocalModel`.
    """
    base_model = Model
    child_models = (GlobalModel, LocalModel)
    list_filter = (PolymorphicChildModelFilter,)
