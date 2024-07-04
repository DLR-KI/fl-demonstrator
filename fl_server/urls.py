# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

"""
URL configuration for fl_server project.

The `urlpatterns` list routes URLs to views. For more information please see:
<https://docs.djangoproject.com/en/4.2/topics/http/urls>

Examples:
    Function views
    1. Add an import:  `from my_app import views`
    2. Add a URL to urlpatterns:  `path("", views.home, name="home")`

    Class-based views
    1. Add an import:  `from other_app.views import Home`
    2. Add a URL to urlpatterns:  `path("", Home.as_view(), name="home")`

    Including another URLconf
    1. Import the include() function:  `from django.urls import include, path`
    2. Add a URL to urlpatterns:  `path("blog/", include("blog.urls"))`
"""
from django.contrib import admin
from django.urls import path, include

from fl_server_api import urls as api_urls


urlpatterns = [
    path("admin/", admin.site.urls),
    path("api-auth/", include("rest_framework.urls")),
    path("api/", include(api_urls)),
]

handler500 = "rest_framework.exceptions.server_error"
handler400 = "rest_framework.exceptions.bad_request"
