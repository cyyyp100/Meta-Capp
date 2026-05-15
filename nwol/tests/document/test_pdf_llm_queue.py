from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from llm.pdf_assistant_queue import PDFLLMQueue


def test_pdf_llm_queue_suspends_immediately_after_timeout():
    queue = PDFLLMQueue()

    result = queue.run_sync(
        "layout_visible",
        lambda: time.sleep(0.05) or ["done"],
        timeout=0.01,
    )

    assert result is None

    started = time.monotonic()
    second = queue.run_sync("layout_visible", lambda: ["should_not_run"], timeout=0.5)

    assert second is None
    assert time.monotonic() - started < 0.05
