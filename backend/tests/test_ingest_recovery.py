from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.ingest import process_file


class IngestRecoveryTest(unittest.TestCase):
    def _pdf_path(self) -> str:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "plan.pdf"
        path.write_bytes(b"test pdf")
        return str(path)

    def test_reingests_when_manifest_is_successful_but_chroma_is_empty(self):
        path = self._pdf_path()
        with (
            patch("utils.ingest.is_current_successful_ingestion", return_value=True),
            patch("utils.ingest.get_manifest_status", return_value={"chroma_doc_count": 5}),
            patch("utils.ingest.count_chroma_documents", return_value=0),
            patch("utils.ingest.ingest_pdf_hybrid", return_value=5) as ingest_pdf,
            patch("utils.ingest.upsert_manifest"),
        ):
            process_file(path)

        ingest_pdf.assert_called_once()

    def test_skips_when_manifest_and_chroma_counts_match(self):
        path = self._pdf_path()
        with (
            patch("utils.ingest.is_current_successful_ingestion", return_value=True),
            patch("utils.ingest.get_manifest_status", return_value={"chroma_doc_count": 5}),
            patch("utils.ingest.count_chroma_documents", return_value=5),
            patch("utils.ingest.ingest_pdf_hybrid") as ingest_pdf,
        ):
            process_file(path)

        ingest_pdf.assert_not_called()


if __name__ == "__main__":
    unittest.main()
