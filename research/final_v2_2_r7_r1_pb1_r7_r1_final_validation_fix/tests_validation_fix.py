from __future__ import annotations

import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path

from research.final_v2_2_r7_r1_pb1_r7_r1_final_validation_fix.validator import _verify_archive_members


def entry(path: str, data: bytes) -> dict:
    return {"path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


class FrozenArchiveMemberSetTests(unittest.TestCase):
    def fixture(self):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        scientific_path = "research/results/final/scientific.json"
        payload_path = "research/results/final/frozen_result_payload_manifest.json"
        scientific = b'{"metric":1}\n'
        payload = b'{"files":[]}\n'
        for rel, data in ((scientific_path, scientific), (payload_path, payload)):
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return td, root, scientific_path, payload_path, scientific, payload

    def test_bound_payload_manifest_is_an_expected_archive_member(self):
        td, root, sci_path, payload_path, sci, payload = self.fixture()
        self.addCleanup(td.cleanup)
        archive = root / "archive.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr(sci_path, sci)
            zf.writestr(payload_path, payload)
        result = _verify_archive_members(
            archive, root, [entry(sci_path, sci)], entry(payload_path, payload))
        self.assertEqual(set(result["actual_members"]), {sci_path, payload_path})
        self.assertEqual(result["missing_members"], [])
        self.assertEqual(result["extra_members"], [])

    def test_unbound_extra_member_is_rejected(self):
        td, root, sci_path, payload_path, sci, payload = self.fixture()
        self.addCleanup(td.cleanup)
        extra_path = "research/results/final/unbound.txt"
        (root / extra_path).write_text("x", encoding="utf-8")
        archive = root / "archive.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr(sci_path, sci)
            zf.writestr(payload_path, payload)
            zf.writestr(extra_path, b"x")
        with self.assertRaisesRegex(PermissionError, "extra"):
            _verify_archive_members(
                archive, root, [entry(sci_path, sci)], entry(payload_path, payload))

    def test_missing_scientific_member_is_rejected(self):
        td, root, sci_path, payload_path, sci, payload = self.fixture()
        self.addCleanup(td.cleanup)
        archive = root / "archive.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr(payload_path, payload)
        with self.assertRaisesRegex(PermissionError, "missing"):
            _verify_archive_members(
                archive, root, [entry(sci_path, sci)], entry(payload_path, payload))


if __name__ == "__main__":
    unittest.main()

