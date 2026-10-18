#!/usr/bin/env python3
"""Resumable, element-filtered downloader for the ECDBench PBE archives.

ECDBench's ICSD/COD/OQMD filenames contain a chemical formula, so those
entries can be filtered without contacting the archive server.  The 64k
``hmat_*`` filenames do not contain a formula.  For them this program streams
the beginning of the tar.gz archive, reads the VASP 5 CHGCAR element line,
and immediately closes the response when an excluded element is found.

All persistent state is kept in the output directory.  Re-running the same
command resumes ``*.part`` files and skips completed archives.
"""

from __future__ import annotations

import argparse
import io
import itertools
import json
import os
import re
import shutil
import sqlite3
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Sequence


DEFAULT_BASE_URL = "https://matgen.nscc-gz.cn/chgcar/download/pbe/"
DEFAULT_FILELIST = (
    "https://raw.githubusercontent.com/pincher-chen/ECDBench/"
    "main/data/pbe/filelist.txt"
)
DEFAULT_EXCLUDED = "Cr,Mn,Fe,Co,Ni"
USER_AGENT = "ECDBench-filtered-downloader/1.0"

ELEMENT_SYMBOLS = frozenset(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe "
    "Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In "
    "Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf "
    "Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm "
    "Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og".split()
)
ELEMENT_TOKEN_RE = re.compile(r"[A-Z][a-z]?")
FORMULA_SUFFIX_RE = re.compile(r"-((?:[A-Z][a-z]?\d*)+)\.chgcar$")
SAFE_ENTRY_RE = re.compile(r"[A-Za-z0-9_.-]+\.chgcar")


class DownloadError(RuntimeError):
    """A recoverable per-file download or archive error."""


class TeeReader(io.RawIOBase):
    """Copy compressed bytes to disk while tarfile consumes an HTTP stream."""

    def __init__(self, source: BinaryIO, sink: BinaryIO):
        self.source = source
        self.sink = sink

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        data = self.source.read(size)
        if data:
            self.sink.write(data)
        return data


def parse_elements(value: str) -> tuple[str, ...]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(values).difference(ELEMENT_SYMBOLS))
    if invalid:
        raise argparse.ArgumentTypeError(
            "invalid element symbol(s): " + ", ".join(invalid)
        )
    return tuple(sorted(set(values)))


def elements_from_entry_name(name: str) -> tuple[str, ...] | None:
    """Return formula elements encoded in an ECDBench filename, if present."""

    match = FORMULA_SUFFIX_RE.search(name)
    if not match:
        return None
    formula = match.group(1)
    elements = tuple(ELEMENT_TOKEN_RE.findall(formula))
    reconstructed = "".join(
        token.group(0) for token in re.finditer(r"[A-Z][a-z]?\d*", formula)
    )
    if not elements or reconstructed != formula:
        return None
    if any(element not in ELEMENT_SYMBOLS for element in elements):
        return None
    return tuple(dict.fromkeys(elements))


def _valid_element_line(line: str) -> tuple[str, ...] | None:
    tokens = tuple(line.split())
    if tokens and all(token in ELEMENT_SYMBOLS for token in tokens):
        return tuple(dict.fromkeys(tokens))
    return None


def elements_from_chgcar_header(stream: BinaryIO) -> tuple[str, ...]:
    """Read element symbols from the VASP 5 element line of a CHGCAR."""

    lines: list[str] = []
    for _ in range(8):
        raw = stream.readline(64 * 1024)
        if not raw:
            break
        lines.append(raw.decode("utf-8", errors="replace").strip())

    # A conventional VASP 5 CHGCAR has symbols on line 6.  Checking nearby
    # lines tolerates an occasional leading blank/comment line while still
    # requiring a line consisting entirely of valid element symbols.
    for index in (5, 6, 4):
        if index < len(lines):
            elements = _valid_element_line(lines[index])
            if elements:
                return elements
    raise DownloadError("could not find a VASP 5 element line in CHGCAR header")


def elements_from_tar_stream(fileobj: BinaryIO) -> tuple[str, ...]:
    """Find the first CHGCAR member in a sequential tar.gz stream."""

    try:
        with tarfile.open(fileobj=fileobj, mode="r|gz") as archive:
            fallback = None
            for member in archive:
                if not member.isfile():
                    continue
                member_stream = archive.extractfile(member)
                if member_stream is None:
                    continue
                lower_name = member.name.lower()
                if lower_name.endswith(".chgcar") or lower_name.endswith("/chgcar"):
                    return elements_from_chgcar_header(member_stream)
                if fallback is None:
                    try:
                        fallback = elements_from_chgcar_header(member_stream)
                    except DownloadError:
                        pass
            if fallback:
                return fallback
    except (tarfile.TarError, EOFError, OSError) as exc:
        raise DownloadError(f"cannot read tar.gz stream: {exc}") from exc
    raise DownloadError("archive contains no regular CHGCAR member")


def elements_from_local_archive(path: Path) -> tuple[str, ...]:
    with path.open("rb") as handle:
        return elements_from_tar_stream(handle)


def _response_status(response: BinaryIO) -> int:
    return int(getattr(response, "status", response.getcode()))


def _content_total(response: BinaryIO, offset: int) -> int | None:
    content_range = response.headers.get("Content-Range", "")
    match = re.search(r"/([0-9]+)$", content_range)
    if match:
        return int(match.group(1))
    content_length = response.headers.get("Content-Length")
    if content_length and content_length.isdigit():
        length = int(content_length)
        return offset + length if _response_status(response) == 206 else length
    return None


@dataclass(frozen=True)
class Config:
    output_dir: Path
    base_url: str
    excluded: frozenset[str]
    timeout: float
    retries: int
    retry_delay: float
    probe_only: bool
    verbose: bool


class StateDB:
    def __init__(self, path: Path):
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                name TEXT PRIMARY KEY,
                elements TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                bytes INTEGER,
                total_bytes INTEGER,
                error TEXT,
                updated_at REAL NOT NULL
            )
            """
        )
        self.connection.commit()
        self.lock = threading.Lock()

    def get_elements(self, name: str) -> tuple[str, ...] | None:
        with self.lock:
            row = self.connection.execute(
                "SELECT elements FROM files WHERE name = ?", (name,)
            ).fetchone()
        if not row or not row[0]:
            return None
        return tuple(json.loads(row[0]))

    def update(
        self,
        name: str,
        *,
        elements: Sequence[str] | None = None,
        status: str,
        bytes_downloaded: int | None = None,
        total_bytes: int | None = None,
        error: str | None = None,
    ) -> None:
        encoded = json.dumps(list(elements)) if elements is not None else None
        with self.lock:
            self.connection.execute(
                """
                INSERT INTO files
                    (name, elements, status, bytes, total_bytes, error, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    elements = COALESCE(excluded.elements, files.elements),
                    status = excluded.status,
                    bytes = COALESCE(excluded.bytes, files.bytes),
                    total_bytes = COALESCE(excluded.total_bytes, files.total_bytes),
                    error = excluded.error,
                    updated_at = excluded.updated_at
                """,
                (
                    name,
                    encoded,
                    status,
                    bytes_downloaded,
                    total_bytes,
                    error,
                    time.time(),
                ),
            )
            self.connection.commit()

    def export_reports(self, output_dir: Path) -> None:
        with self.lock:
            rows = list(
                self.connection.execute(
                    "SELECT name, elements, status, bytes, total_bytes, error "
                    "FROM files ORDER BY name"
                )
            )
        report_path = output_dir / "download_report.tsv"
        temporary = report_path.with_suffix(".tsv.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write("name\telements\tstatus\tbytes\ttotal_bytes\terror\n")
            for name, elements, status, size, total, error in rows:
                element_text = ",".join(json.loads(elements)) if elements else ""
                clean_error = (error or "").replace("\t", " ").replace("\n", " ")
                handle.write(
                    f"{name}\t{element_text}\t{status}\t{size or ''}\t"
                    f"{total or ''}\t{clean_error}\n"
                )
        os.replace(temporary, report_path)

    def close(self) -> None:
        with self.lock:
            self.connection.close()


class Downloader:
    def __init__(self, config: Config, state: StateDB):
        self.config = config
        self.state = state
        self.print_lock = threading.Lock()

    def log(self, message: str, *, always: bool = False) -> None:
        if not always and not self.config.verbose:
            return
        with self.print_lock:
            print(message, flush=True)

    def _url(self, name: str) -> str:
        archive_name = name + ".tar.gz"
        return urllib.parse.urljoin(
            self.config.base_url.rstrip("/") + "/",
            urllib.parse.quote(archive_name),
        )

    def _open(self, name: str, offset: int = 0):
        headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(self._url(name), headers=headers)
        return urllib.request.urlopen(request, timeout=self.config.timeout)

    def _paths(self, name: str) -> tuple[Path, Path]:
        final_path = self.config.output_dir / (name + ".tar.gz")
        return final_path, final_path.with_suffix(final_path.suffix + ".part")

    def _is_excluded(self, elements: Iterable[str]) -> bool:
        return bool(self.config.excluded.intersection(elements))

    def _finish(self, name: str, part_path: Path, final_path: Path, total: int | None):
        size = part_path.stat().st_size
        if total is not None and size != total:
            raise DownloadError(f"incomplete response: have {size} bytes, expected {total}")
        os.replace(part_path, final_path)
        self.state.update(
            name,
            status="downloaded",
            bytes_downloaded=size,
            total_bytes=total or size,
        )
        self.log(f"downloaded {name} ({size} bytes)")

    def _download_resume(self, name: str) -> str:
        final_path, part_path = self._paths(name)
        if final_path.is_file():
            size = final_path.stat().st_size
            self.state.update(
                name, status="downloaded", bytes_downloaded=size, total_bytes=size
            )
            return "downloaded"

        offset = part_path.stat().st_size if part_path.exists() else 0
        try:
            response = self._open(name, offset)
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and offset:
                content_range = exc.headers.get("Content-Range", "")
                match = re.search(r"\*/([0-9]+)$", content_range)
                if match and int(match.group(1)) == offset:
                    self._finish(name, part_path, final_path, offset)
                    return "downloaded"
            raise

        with response:
            status = _response_status(response)
            if offset and status != 206:
                self.log(f"server ignored Range for {name}; restarting")
                offset = 0
            total = _content_total(response, offset)
            mode = "ab" if offset and status == 206 else "wb"
            self.state.update(
                name,
                status="downloading",
                bytes_downloaded=offset,
                total_bytes=total,
            )
            with part_path.open(mode) as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        self._finish(name, part_path, final_path, total)
        return "downloaded"

    def _probe_hmat_then_download(self, name: str) -> str:
        final_path, part_path = self._paths(name)
        if final_path.is_file():
            elements = elements_from_local_archive(final_path)
            size = final_path.stat().st_size
            self.state.update(
                name,
                elements=elements,
                status="downloaded",
                bytes_downloaded=size,
                total_bytes=size,
            )
            if self._is_excluded(elements):
                self.log(
                    f"warning: completed archive now matches exclusion policy: {name}",
                    always=True,
                )
            return "downloaded"

        known_elements = self.state.get_elements(name)
        if known_elements:
            if self._is_excluded(known_elements):
                self.state.update(name, elements=known_elements, status="excluded")
                return "excluded"
            if self.config.probe_only:
                self.state.update(name, elements=known_elements, status="probed")
                return "probed"
            return self._download_resume(name)

        # A partial hmat archive without a recorded composition cannot be
        # decompressed from its middle.  Restart only this program-owned .part
        # file so the CHGCAR header can be inspected safely.
        if part_path.exists():
            part_path.unlink()

        response = self._open(name)
        with response:
            total = _content_total(response, 0)
            with part_path.open("wb") as output:
                tee = TeeReader(response, output)
                elements = elements_from_tar_stream(tee)
                output.flush()
                self.state.update(
                    name,
                    elements=elements,
                    status="probed",
                    bytes_downloaded=output.tell(),
                    total_bytes=total,
                )
                if self._is_excluded(elements):
                    response.close()
                    output.close()
                    part_path.unlink(missing_ok=True)
                    self.state.update(name, elements=elements, status="excluded")
                    self.log(f"excluded {name}: {','.join(elements)}")
                    return "excluded"
                if self.config.probe_only:
                    response.close()
                    output.close()
                    part_path.unlink(missing_ok=True)
                    self.state.update(name, elements=elements, status="probed")
                    self.log(f"probed {name}: {','.join(elements)}")
                    return "probed"
                shutil.copyfileobj(response, output, length=1024 * 1024)

        self._finish(name, part_path, final_path, total)
        return "downloaded"

    def process_once(self, name: str) -> str:
        elements = elements_from_entry_name(name)
        if elements is not None:
            if self._is_excluded(elements):
                self.state.update(name, elements=elements, status="excluded")
                return "excluded"
            self.state.update(name, elements=elements, status="pending")
            if self.config.probe_only:
                self.state.update(name, elements=elements, status="probed")
                return "probed"
            return self._download_resume(name)
        return self._probe_hmat_then_download(name)

    def process(self, name: str) -> str:
        last_error = ""
        for attempt in range(1, self.config.retries + 2):
            try:
                return self.process_once(name)
            except Exception as exc:  # isolate failures across 140k entries
                last_error = f"{type(exc).__name__}: {exc}"
                self.state.update(name, status="failed", error=last_error)
                if attempt > self.config.retries:
                    self.log(f"failed {name}: {last_error}", always=True)
                    return "failed"
                delay = self.config.retry_delay * (2 ** (attempt - 1))
                self.log(
                    f"retry {attempt}/{self.config.retries} for {name} "
                    f"in {delay:.1f}s: {last_error}",
                    always=True,
                )
                time.sleep(delay)
        raise AssertionError(last_error)


def load_filelist(source: str, output_dir: Path, refresh: bool, timeout: float) -> list[str]:
    cache_path = output_dir / "ecd_pbe_filelist.txt"
    parsed = urllib.parse.urlparse(source)
    is_url = parsed.scheme in {"http", "https"}

    if is_url and cache_path.exists() and not refresh:
        text = cache_path.read_text(encoding="utf-8")
    elif is_url:
        request = urllib.request.Request(source, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8")
        temporary = cache_path.with_suffix(".txt.tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, cache_path)
    else:
        text = Path(source).read_text(encoding="utf-8")

    entries: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        name = raw_line.strip()
        if not name:
            continue
        if not SAFE_ENTRY_RE.fullmatch(name):
            raise DownloadError(f"unsafe or unexpected filelist entry: {name!r}")
        if name not in seen:
            entries.append(name)
            seen.add(name)
    if not entries:
        raise DownloadError("filelist is empty")
    return entries


def dry_run(entries: Sequence[str], excluded: frozenset[str]) -> None:
    known_kept = known_excluded = unknown = 0
    for name in entries:
        elements = elements_from_entry_name(name)
        if elements is None:
            unknown += 1
        elif excluded.intersection(elements):
            known_excluded += 1
        else:
            known_kept += 1
    print(f"entries:                 {len(entries):>10}")
    print(f"known formula, kept:     {known_kept:>10}")
    print(f"known formula, excluded: {known_excluded:>10}")
    print(f"hmat requiring probe:    {unknown:>10}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/ecd-pbe"),
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--filelist", default=DEFAULT_FILELIST)
    parser.add_argument(
        "--exclude-elements",
        type=parse_elements,
        default=parse_elements(DEFAULT_EXCLUDED),
        help=f"comma-separated exact symbols (default: {DEFAULT_EXCLUDED})",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--max-files", type=int)
    parser.add_argument(
        "--only-prefix",
        action="append",
        help="restrict to a source prefix such as hmat or icsd; repeatable",
    )
    parser.add_argument("--refresh-filelist", action="store_true")
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="record hmat compositions without retaining any new archives",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="classify filenames only; do not contact the archive server",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.retries < 0:
        raise SystemExit("--retries cannot be negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    entries = load_filelist(
        args.filelist, args.output_dir, args.refresh_filelist, args.timeout
    )
    if args.only_prefix:
        prefixes = tuple(prefix + ("_" if prefix == "hmat" else "-") for prefix in args.only_prefix)
        entries = [name for name in entries if name.startswith(prefixes)]
    if args.max_files is not None:
        entries = entries[: args.max_files]

    excluded = frozenset(args.exclude_elements)
    print("excluded elements: " + ",".join(sorted(excluded)), flush=True)
    if args.dry_run:
        dry_run(entries, excluded)
        return 0

    config = Config(
        output_dir=args.output_dir,
        base_url=args.base_url,
        excluded=excluded,
        timeout=args.timeout,
        retries=args.retries,
        retry_delay=args.retry_delay,
        probe_only=args.probe_only,
        verbose=args.verbose,
    )
    state = StateDB(args.output_dir / "download_state.sqlite3")
    downloader = Downloader(config, state)
    counts = {"downloaded": 0, "excluded": 0, "probed": 0, "failed": 0}

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            entry_iterator = iter(entries)
            futures = {
                executor.submit(downloader.process, name): name
                for name in itertools.islice(entry_iterator, args.workers * 2)
            }
            completed = 0
            while futures:
                future = next(as_completed(futures))
                futures.pop(future)
                result = future.result()
                counts[result] = counts.get(result, 0) + 1
                completed += 1
                if completed % 100 == 0 or completed == len(entries):
                    print(
                        f"progress {completed}/{len(entries)} | "
                        + " ".join(f"{key}={value}" for key, value in counts.items()),
                        flush=True,
                    )
                try:
                    name = next(entry_iterator)
                except StopIteration:
                    pass
                else:
                    futures[executor.submit(downloader.process, name)] = name
    except KeyboardInterrupt:
        print("interrupted; partial files and state are resumable", file=sys.stderr)
        return_code = 130
    else:
        return_code = 1 if counts["failed"] else 0
    finally:
        state.export_reports(args.output_dir)
        state.close()

    print("summary: " + " ".join(f"{k}={v}" for k, v in counts.items()))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
