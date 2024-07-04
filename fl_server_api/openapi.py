# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

from docstring_parser import Docstring, parse, RenderingStyle
from docstring_parser.google import compose
from drf_spectacular.authentication import BasicScheme
from drf_spectacular.openapi import AutoSchema
from drf_spectacular.utils import OpenApiExample, OpenApiResponse
from inspect import cleandoc
from typing import Callable, List, Optional, Tuple

from .serializers.generic import ErrorSerializer
from .utils import fullname


class BasicAuthAllowingTokenAuthInUrlScheme(BasicScheme):
    """
    A class that extends the BasicScheme to allow token authentication in the URL.
    """

    target_class = "fl_server_api.views.base.BasicAuthAllowingTokenAuthInUrl"
    priority = 0


def create_error_response(
    response_description: Optional[str],
    example_name: str,
    example_details: str,
    example_description: Optional[str],
    **example_kwargs
) -> OpenApiResponse:
    """
    Create an OpenAPI error response.

    Args:
        response_description (Optional[str]): The description of the response.
        example_name (str): The name of the example.
        example_details (str): The details of the example.
        example_description (Optional[str]): The description of the example.
        **example_kwargs: Additional keyword arguments for the example.

    Returns:
        OpenApiResponse: The created OpenAPI response.
    """
    return OpenApiResponse(
        response=ErrorSerializer,
        description=response_description,
        examples=[
            OpenApiExample(
                example_name,
                value={"details": example_details},
                description=example_description,
                **example_kwargs,
            )
        ]
    )


error_response_403 = create_error_response(
    "Unauthorized",
    "Unauthorized",
    "Authentication credentials were not provided.",
    "Do not forget to authorize first!"
)
"""Generic OpenAPI 403 response."""


def custom_preprocessing_hook(endpoints: List[Tuple[str, str, str, Callable]]):
    """
    Hide the "/api/dummy/" endpoint from the OpenAPI schema.

    Args:
        endpoints (List[Tuple[str, str, str, Callable]]): The list of endpoints.

    Returns:
        Iterator: The filtered list of endpoints.
    """
    # your modifications to the list of operations that are exposed in the schema
    # for (path, path_regex, method, callback) in endpoints:
    #     pass
    return filter(lambda endpoint: endpoint[0] != "/api/dummy/", endpoints)


class CustomAutoSchema(AutoSchema):
    """
    A custom AutoSchema that includes the documented examples from the Docstrings in the description.
    """

    show_examples = True
    """Flag to include examples in the description."""
    rendering_style = RenderingStyle.CLEAN
    """Docstring rendering style."""

    def _get_docstring(self):
        """
        Get the docstring of the view.

        This method parses the description of the view.

        Returns:
            Docstring: The parsed docstring.
        """
        return parse(super().get_description())

    def _get_param_docstring(self, docstring: Docstring, argument: str) -> Optional[str]:
        """
        Get the docstring of a parameter.

        This method finds the parameter in the docstring and returns its description.

        Args:
            docstring (Docstring): The docstring.
            argument (str): The name of the argument.

        Returns:
            Optional[str]: The description of the argument, or `None` if the argument is not found.
        """
        params = [p for p in docstring.params if p.arg_name == argument]
        if not params:
            return None
        return params[0].description

    def get_description(self):
        """
        Get the description of the view including its examples (if desired) formatted as markdown.

        Returns:
            str: The description of the view as markdown.
        """
        docstring = self._get_docstring()
        tmp_docstring = Docstring(style=docstring.style)
        tmp_docstring.short_description = docstring.short_description
        tmp_docstring.long_description = docstring.long_description
        if self.show_examples:
            tmp_docstring.meta.extend(docstring.examples)
        desc = compose(tmp_docstring, self.rendering_style, indent="")
        if self.show_examples and desc.__contains__("Examples:"):
            # customize examples section:
            # - examples should be in a new paragraph (not concatenated with the description)
            # - the examples header should be a h3 title
            desc = desc.replace("\nExamples:\n", "\n\n### Examples:\n\n")
        desc = cleandoc(desc)
        return desc

    def _resolve_path_parameters(self, variables: List[str]):
        """
        Resolve the path parameters of the view and set their descriptions if they are missing.

        Args:
            variables (List[str]): The list of variables in the path.

        Returns:
            list: The list of path parameters.
        """
        parameters = super()._resolve_path_parameters(variables)
        docstring = self._get_docstring()
        for parameter in parameters:
            if "description" not in parameter:
                description = self._get_param_docstring(docstring, parameter["name"])
                if description:
                    parameter["description"] = description
        return parameters

    def get_operation_id(self):
        """
        Get the operation ID which is the fully qualified name of the corresponding view/action/method.

        Returns:
            str: The operation ID.
        """
        action_or_method = getattr(self.view, getattr(self.view, 'action', self.method.lower()), None)
        return fullname(action_or_method or self.view.__class__)
