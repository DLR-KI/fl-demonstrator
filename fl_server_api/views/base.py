from logging import getLogger
from rest_framework.authentication import BasicAuthentication, SessionAuthentication, TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.viewsets import ViewSet as DjangoViewSet


class BasicAuthAllowingTokenAuthInUrl(BasicAuthentication):

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
    DLR Federated Learning base ViewSet including default authentication and permission classes.

    Authentication as well as permission classes can not only be overwritten by the child class, but also by the
    request method. This allows to have different authentication and permission classes for each request method.
    To overwrite the authentication and permission classes for a specific request method, simply add the designated
    decorators: `@decorators.authentication_classes` and `@decorators.permission_classes` from
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
    permission_classes = [IsAuthenticated]

    def get_authenticators(self):
        if method := self._get_view_method():
            if hasattr(method, "authentication_classes"):
                return method.authentication_classes
        return super().get_authenticators()

    def get_permissions(self):
        if method := self._get_view_method():
            if hasattr(method, "permission_classes"):
                return method.permission_classes
        return super().get_permissions()

    def _get_view_method(self):
        if hasattr(self, "action") and self.action is not None:
            return self.__getattribute__(self.action)
        if hasattr(self.request, "method") and self.request.method is not None:
            http_method = self.request.method.lower()
            if hasattr(self, http_method):
                return self.__getattribute__(http_method)
        return None
