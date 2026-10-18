import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from aiden.download_ecd_pbe import (
    Config,
    Downloader,
    StateDB,
    elements_from_entry_name,
    elements_from_local_archive,
)


def make_chgcar_archive(name: str, elements: tuple[str, ...], padding: int = 0) -> bytes:
    header = "\n".join(
        [
            name,
            "1.0",
            "1 0 0",
            "0 1 0",
            "0 0 1",
            " ".join(elements),
            " ".join("1" for _ in elements),
            "Direct",
            "0 0 0",
            "",
        ]
    ).encode()
    payload = header + (b"density-data\n" * padding)
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


class FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, status: int, headers: dict[str, str]):
        super().__init__(body)
        self.status = status
        self.headers = headers

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class InMemoryDownloader(Downloader):
    def __init__(self, config, state):
        super().__init__(config, state)
        self.archives: dict[str, bytes] = {}
        self.ranges: list[tuple[str, str | None]] = []

    def _open(self, name: str, offset: int = 0):
        archive_name = name + ".tar.gz"
        data = self.archives[archive_name]
        range_header = f"bytes={offset}-" if offset else None
        self.ranges.append((archive_name, range_header))
        if offset:
            return FakeResponse(
                data[offset:],
                206,
                {
                    "Content-Length": str(len(data) - offset),
                    "Content-Range": f"bytes {offset}-{len(data)-1}/{len(data)}",
                },
            )
        return FakeResponse(data, 200, {"Content-Length": str(len(data))})


class DownloaderTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temporary.name)
        self.state = StateDB(self.output_dir / "state.sqlite3")
        config = Config(
            output_dir=self.output_dir,
            base_url="https://example.invalid/",
            excluded=frozenset({"Cr", "Mn", "Fe", "Co", "Ni"}),
            timeout=5,
            retries=0,
            retry_delay=0,
            probe_only=False,
            verbose=False,
        )
        self.downloader = InMemoryDownloader(config, self.state)

    def tearDown(self):
        self.state.close()
        self.temporary.cleanup()

    def test_formula_parser_is_element_exact(self):
        self.assertEqual(
            elements_from_entry_name("icsd-1-Li2Fe1O4.chgcar"),
            ("Li", "Fe", "O"),
        )
        self.assertEqual(
            elements_from_entry_name("cod-2-Co1N1.chgcar"), ("Co", "N")
        )
        self.assertIsNone(elements_from_entry_name("hmat_42.chgcar"))

    def test_known_formula_is_excluded_without_request(self):
        result = self.downloader.process("icsd-1-Li2Fe1O4.chgcar")
        self.assertEqual(result, "excluded")
        self.assertEqual(self.downloader.ranges, [])

    def test_hmat_header_probe_excludes_and_removes_partial(self):
        name = "hmat_1.chgcar"
        archive_name = name + ".tar.gz"
        self.downloader.archives[archive_name] = make_chgcar_archive(
            name, ("O", "Fe"), padding=1000
        )
        result = self.downloader.process(name)
        self.assertEqual(result, "excluded")
        self.assertFalse((self.output_dir / archive_name).exists())
        self.assertFalse((self.output_dir / (archive_name + ".part")).exists())
        self.assertEqual(self.state.get_elements(name), ("O", "Fe"))

    def test_hmat_header_probe_continues_same_download(self):
        name = "hmat_2.chgcar"
        archive_name = name + ".tar.gz"
        expected = make_chgcar_archive(name, ("Si", "O"), padding=1000)
        self.downloader.archives[archive_name] = expected
        result = self.downloader.process(name)
        final_path = self.output_dir / archive_name
        self.assertEqual(result, "downloaded")
        self.assertEqual(final_path.read_bytes(), expected)
        self.assertEqual(elements_from_local_archive(final_path), ("Si", "O"))
        self.assertEqual(len(self.downloader.ranges), 1)

    def test_known_formula_resumes_part_file_with_range(self):
        name = "icsd-3-Si1O2.chgcar"
        archive_name = name + ".tar.gz"
        expected = make_chgcar_archive(name, ("Si", "O"), padding=2000)
        self.downloader.archives[archive_name] = expected
        part_path = self.output_dir / (archive_name + ".part")
        offset = len(expected) // 2
        part_path.write_bytes(expected[:offset])

        result = self.downloader.process(name)
        self.assertEqual(result, "downloaded")
        self.assertEqual((self.output_dir / archive_name).read_bytes(), expected)
        self.assertIn((archive_name, f"bytes={offset}-"), self.downloader.ranges)


if __name__ == "__main__":
    unittest.main()
