# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

# Generated by Django 4.0.10 on 2023-07-13 12:46

from django.conf import settings
import django.contrib.auth.models
import django.contrib.auth.validators
from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone
import fl_server_core.models.user
import uuid


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('contenttypes', '0002_remove_content_type_name'),
        ('auth', '0012_alter_user_first_name_max_length'),
    ]

    operations = [
        migrations.CreateModel(
            name='User',
            fields=[
                ('password', models.CharField(max_length=128, verbose_name='password')),
                ('last_login', models.DateTimeField(blank=True, null=True, verbose_name='last login')),
                ('is_superuser', models.BooleanField(default=False, help_text='Designates that this user has all permissions without explicitly assigning them.', verbose_name='superuser status')),
                ('username', models.CharField(error_messages={'unique': 'A user with that username already exists.'}, help_text='Required. 150 characters or fewer. Letters, digits and @/./+/-/_ only.', max_length=150, unique=True, validators=[django.contrib.auth.validators.UnicodeUsernameValidator()], verbose_name='username')),
                ('first_name', models.CharField(blank=True, max_length=150, verbose_name='first name')),
                ('last_name', models.CharField(blank=True, max_length=150, verbose_name='last name')),
                ('email', models.EmailField(blank=True, max_length=254, verbose_name='email address')),
                ('is_staff', models.BooleanField(default=False, help_text='Designates whether the user can log into this admin site.', verbose_name='staff status')),
                ('is_active', models.BooleanField(default=True, help_text='Designates whether this user should be treated as active. Unselect this instead of deleting accounts.', verbose_name='active')),
                ('date_joined', models.DateTimeField(default=django.utils.timezone.now, verbose_name='date joined')),
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('actor', models.BooleanField(default=False)),
                ('client', models.BooleanField(default=False)),
                ('message_endpoint', models.URLField()),
                ('groups', models.ManyToManyField(blank=True, help_text='The groups this user belongs to. A user will get all permissions granted to each of their groups.', related_name='user_set', related_query_name='user', to='auth.group', verbose_name='groups')),
                ('user_permissions', models.ManyToManyField(blank=True, help_text='Specific permissions for this user.', related_name='user_set', related_query_name='user', to='auth.permission', verbose_name='user permissions')),
            ],
            options={
                'verbose_name': 'user',
                'verbose_name_plural': 'users',
                'abstract': False,
            },
            bases=(models.Model, fl_server_core.models.user.NotificationReceiver),
            managers=[
                ('objects', django.contrib.auth.models.UserManager()),
            ],
        ),
        migrations.CreateModel(
            name='Model',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('round', models.IntegerField()),
                ('weights', models.BinaryField()),
                ('owner', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
                ('polymorphic_ctype', models.ForeignKey(editable=False, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='polymorphic_%(app_label)s.%(class)s_set+', to='contenttypes.contenttype')),
            ],
            options={
                'abstract': False,
                'base_manager_name': 'objects',
            },
        ),
        migrations.CreateModel(
            name='GlobalModel',
            fields=[
                ('model_ptr', models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True, primary_key=True, serialize=False, to='fl_server_core.model')),
                ('name', models.CharField(max_length=256)),
                ('description', models.TextField()),
                ('swag_first_moment', models.BinaryField(blank=True, null=True)),
                ('swag_second_moment', models.BinaryField(blank=True, null=True)),
            ],
            options={
                'abstract': False,
                'base_manager_name': 'objects',
            },
            bases=('fl_server_core.model',),
        ),
        migrations.CreateModel(
            name='Training',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('state', models.CharField(choices=[('I', 'Initial'), ('O', 'Ongoing'), ('C', 'Completed'), ('E', 'Error'), ('S', 'SwagRound')], max_length=1)),
                ('target_num_updates', models.IntegerField()),
                ('last_update', models.DateTimeField(auto_now=True)),
                ('uncertainty_method', models.CharField(choices=[('NONE', 'None'), ('SWAG', 'SWAG'), ('ENSEMBLE_LOCAL', 'Ensemble Local'), ('ENSEMBLE_GLOBAL', 'Ensemble Global'), ('MC_DROPOUT', 'MC Dropout')], default='NONE', max_length=32)),
                ('aggregation_method', models.CharField(choices=[('AVG', 'Average')], max_length=3)),
                ('locked', models.BooleanField(default=False)),
                ('actor', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='actors', to=settings.AUTH_USER_MODEL)),
                ('model', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to='fl_server_core.model')),
                ('participants', models.ManyToManyField(to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='Metric',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('identifier', models.CharField(blank=True, max_length=64, null=True)),
                ('key', models.CharField(max_length=32)),
                ('value_float', models.FloatField(blank=True, null=True)),
                ('value_binary', models.BinaryField(blank=True, null=True)),
                ('step', models.IntegerField(blank=True, null=True)),
                ('model', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='fl_server_core.model')),
                ('reporter', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='LocalModel',
            fields=[
                ('model_ptr', models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True, primary_key=True, serialize=False, to='fl_server_core.model')),
                ('sample_size', models.IntegerField()),
                ('base_model', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='fl_server_core.globalmodel')),
            ],
            options={
                'abstract': False,
                'base_manager_name': 'objects',
            },
            bases=('fl_server_core.model',),
        ),
    ]
