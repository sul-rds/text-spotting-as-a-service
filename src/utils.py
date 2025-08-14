import sys
from contextlib import contextmanager
from io import StringIO


@contextmanager
def capture_stdout():
    """
    Context manager encapsulating a pattern for capturing writes to stdout.
    Restores sys.stdout upon exceptions.

    Usage:
    >>> with capture_stdout() as get_value:
    >>>     print("here is a print")
    >>>     captured = get_value()
    >>> print('Gotcha: ' + captured)
    """

    # Redirect sys.stdout
    out = StringIO()
    sys.stdout = out

    # Yield a method clients can use to obtain the value
    try:
        yield out.getvalue
    finally:
        # Restore the stdout when done
        sys.stdout = sys.__stdout__
