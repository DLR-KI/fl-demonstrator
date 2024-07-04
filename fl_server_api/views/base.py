# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from logging import getLogger
from rest_framework.authentication import BasicAuthentication, SessionAuthentication, TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.viewsets import ViewSet as DjangoViewSet


class BasicAuthAllowingTokenAuthInUrl(BasicAuthentication):
    """
    A class that extends the BasicAuthentication to allow token authentication in the URL.
    """

    def authenticate_credentials(self, userid_or_token, password, request=None):
        """
        Authenticate credentials against username/password or token.

        Basic Authentication:
        Authenticate the userid and password against username and password
        with optional request for context.

        Token Authentication over URL:
        Authenticate the given token against the token in the database.
        """
        # check if special token authentication is used
        if (len(userid_or_token) == 40 and password == ""):
            # tokens are always 40 characters long
            # see: rest_framework.authtoken.models.Token (class method: generate_key)
            #      which uses `binascii.hexlify(os.urandom(20)).decode()`
            return TokenAuthentication().authenticate_credentials(userid_or_token)

        # default Basic Authentication
        return super().authenticate_credentials(userid_or_token, password, request)


class ViewSet(DjangoViewSet):
    """
    A base ViewSet that includes default authentication and permission classes.

    This class allows the authentication and permission classes to be overwritten by the child class or the request
    method. To overwrite the authentication and permission classes for a specific request method, use the
    `@decorators.authentication_classes` and `@decorators.permission_classes` decorators from
    `rest_framework.decorators`.
    """

    _logger = getLogger("fl.server")

    # Note: BasicAuthentication is sensles here since it will and can't never be called due to
    #       BasicAuthAllowingTokenAuthInUrl but is required for OpenAPI to work.
    # Also note that the order of BasicAuthAllowingTokenAuthInUrl and BasicAuthentication is important
    # since if BasicAuthentication is first, Django won't ever call BasicAuthAllowingTokenAuthInUrl!
    authentication_classes = [
        TokenAuthentication,
        BasicAuthAllowingTokenAuthInUrl,
        BasicAuthentication,
        SessionAuthentication,
    ]
    """The authentication classes for the ViewSet."""
    permission_classes = [IsAuthenticated]
    """The permission classes for the ViewSet."""

    def get_authenticators(self):
        """
        Get the authenticators for the ViewSet.

        This method gets the view method and, if it has authentication classes defined via the decorator, returns them.
        Otherwise, it falls back to the default authenticators.

        Returns:
            list: The authenticators for the ViewSet.
        """
        if method := self._get_view_method():
            if hasattr(method, "authentication_classes"):
                return method.authentication_classes
        return super().get_authenticators()

    def get_permissions(self):
        """
        Get the permissions for the ViewSet.

        This method gets the view method and, if it has permission classes defined via the decorator, returns them.
        Otherwise, it falls back to the default permissions.

        Returns:
            list: The permissions for the ViewSet.
        """
        if method := self._get_view_method():
            if hasattr(method, "permission_classes"):
                return method.permission_classes
        return super().get_permissions()

    def _get_view_method(self):
        """
        Get the view method for the ViewSet.

        This method gets the action or the HTTP method of the request and returns the corresponding method of the
        ViewSet, or `None` if no such method is found.

        Returns:
            Callable | None: The view method for the ViewSet, or `None` if no such method is found.
        """
        if hasattr(self, "action") and self.action is not None:
            return self.__getattribute__(self.action)
        if hasattr(self.request, "method") and self.request.method is not None:
            http_method = self.request.method.lower()
            if hasattr(self, http_method):
                return self.__getattribute__(http_method)
        return None
