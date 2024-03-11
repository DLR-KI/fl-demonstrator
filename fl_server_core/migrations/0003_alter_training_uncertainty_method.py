# Generated by Django 4.0.10 on 2023-07-19 14:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('fl_server_core', '0002_remove_globalmodel_swag_first_moment_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='training',
            name='uncertainty_method',
            field=models.CharField(choices=[('NONE', 'None'), ('SWAG', 'SWAG'), ('ENSEMBLE', 'Ensemble'), ('MC_DROPOUT', 'MC Dropout')], default='NONE', max_length=32),
        ),
    ]
