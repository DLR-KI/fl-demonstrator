from contextlib import contextmanager
from django.db import transaction
from django.db.models import Model
from django.db.transaction import get_connection


@contextmanager
def locked_atomic_transaction(*models: Model):
    """
    Create a locked atomic transaction for a model.

    Opens an atomic transaction and locks a specific table until the context is closed.
    Original code taken from: <https://stackoverflow.com/a/54403001>

    Attention: "LOCK TABLE" is not available in SQLite, so this function will not work with SQLite.
               A possible fix would be to skip the "LOCK TABLE" part for SQLite since transactions in SQLite
               lock the whole database anyway. At least if "BEGIN IMMEDIATE TRANSACTION" or
               "BEGIN EXCLUSIVE TRANSACTION" is used. The default "BEGIN TRANSACTION" or "BEGIN DEFERRED TRANSACTION"
               does not lock the database until the first database access.

    Args:
        model (Model): database model
    """
    with transaction.atomic():
        tables = ", ".join(map(lambda model: model._meta.db_table, models))
        cursor = get_connection().cursor()
        cursor.execute(f"LOCK TABLE {tables}")
        try:
            yield
        finally:
            cursor.close()
