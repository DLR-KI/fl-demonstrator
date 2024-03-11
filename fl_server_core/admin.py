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
    base_model = Model


@admin.register(GlobalModel)
class GlobalModelAdmin(ModelChildAdmin):
    base_model = GlobalModel


@admin.register(LocalModel)
class LocalModelAdmin(ModelChildAdmin):
    base_model = LocalModel


@admin.register(Model)
class ModelParentAdmin(PolymorphicParentModelAdmin):
    """The parent model admin"""
    base_model = Model
    child_models = (GlobalModel, LocalModel)
    list_filter = (PolymorphicChildModelFilter,)
