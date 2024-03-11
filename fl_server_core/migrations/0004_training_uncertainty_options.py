# Generated by Django 4.0.10 on 2023-07-24 13:43

import django.core.serializers.json
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('fl_server_core', '0003_alter_training_uncertainty_method'),
    ]

    operations = [
        migrations.AddField(
            model_name='training',
            name='uncertainty_options',
            field=models.JSONField(default=dict, encoder=django.core.serializers.json.DjangoJSONEncoder),
        ),
    ]
