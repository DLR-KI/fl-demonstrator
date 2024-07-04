# SPDX-FileCopyrightText: 2024 Benedikt Franke <benedikt.franke@dlr.de>
# SPDX-FileCopyrightText: 2024 Florian Heinrich <florian.heinrich@dlr.de>
#
# SPDX-License-Identifier: Apache-2.0

class TorchDeserializationException(Exception):
    """
    Exception raised for errors in the deserialization process of PyTorch objects.

    This is a custom exception class that should be raised when there is an error during
    the deserialization of PyTorch objects.
    """
    pass
