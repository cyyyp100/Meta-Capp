from __future__ import annotations

import contextlib
import io
import logging
from collections.abc import Iterator


@contextlib.contextmanager
def capture_noisy_extractor_output(logger: logging.Logger, label: str) -> Iterator[None]:
    """Silence verbose third-party PDF extractors.

    Some backends print raw PDF objects to stdout/stderr for recoverable
    warnings. Keep the UI readable and expose only a short debug summary.
    """
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            yield
    finally:
        output = "\n".join(part.strip() for part in (stdout.getvalue(), stderr.getvalue()) if part.strip())
        if output:
            first_line = output.splitlines()[0][:500]
            logger.debug("%s a produit %s caractere(s) de sortie externe: %s", label, len(output), first_line)
