# Generated by Django 4.0.10 on 2023-09-27 06:18

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('fl_server_core', '0005_alter_training_aggregation_method_and_more'),
    ]

    operations = [
        migrations.RenameField(
            model_name='training',
            old_name='uncertainty_options',
            new_name='options',
        ),
    ]
