# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

# Generated by Django 4.0.10 on 2024-04-25 07:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('fl_server_core', '0007_globalmodel_input_shape'),
    ]

    operations = [
        migrations.AddField(
            model_name='globalmodel',
            name='preprocessing',
            field=models.BinaryField(null=True),
        ),
    ]
