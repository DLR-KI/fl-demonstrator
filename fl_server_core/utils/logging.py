# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
import logging
from typing import Optional


@contextmanager
def disable_logger(logger: Optional[logging.Logger] = None):
    """
    Temporary disable the Logger.
    """
    previous = logger.disabled if logger else logging.root.manager.disable
    if logger:
        logger.disabled = True
    else:
        logging.disable()
    try:
        yield
    finally:
        if logger:
            logger.disabled = previous  # type: ignore
        else:
            logging.disable(previous)
