"""SOCOFing dataset parsing and gallery/probe construction.

Filename convention (confirmed against on-disk files by scripts/verify_dataset.py
BEFORE any model runs -- do not assume):

    Real:    <id>__<gender>_<hand>_<finger>_finger.BMP
    Altered: <id>__<gender>_<hand>_<finger>_finger_<ALT>.BMP   ALT in {Obl, CR, Zcut}

Identity = (subject_id, hand, finger)  ->  6000 finger-identities (600 x 10).
Gender is parsed as metadata but excluded from the identity key (it is constant
per subject, so probe<->real joins are unaffected).

All parsing is case-insensitive; the raw casing of the alteration token is
retained so verify_dataset.py can report a casing histogram.
"""
from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

FINGERS: Tuple[str, ...] = ("thumb", "index", "middle", "ring", "little")
HANDS: Tuple[str, ...] = ("Left", "Right")
ALTS: Tuple[str, ...] = ("Obl", "CR", "Zcut")

_ALT_CANON: Dict[str, str] = {a.lower(): a for a in ALTS}

# Strict parser: only matches the four known alteration tokens (or none).
FILENAME_RE = re.compile(
    r"^(?P<id>\d+)__(?P<gender>[MF])_(?P<hand>Left|Right)_"
    r"(?P<finger>thumb|index|middle|ring|little)_finger"
    r"(?:_(?P<alt>Obl|CR|Zcut))?\.bmp$",
    re.IGNORECASE,
)

# Loose parser: captures ANY token after `_finger_` so the casing histogram can
# also surface unexpected/unknown suffixes (not just Obl/CR/Zcut).
LOOSE_ALT_RE = re.compile(r"_finger_(?P<tok>[^.]+)\.bmp$", re.IGNORECASE)

Identity = Tuple[int, str, str]  # (subject_id, hand, finger)


@dataclass(frozen=True)
class FpRecord:
    path: str
    filename: str
    subject_id: int
    gender: str            # canonical 'M'/'F'
    hand: str              # canonical 'Left'/'Right'
    finger: str            # canonical lowercase
    alt: Optional[str]     # canonical 'Obl'/'CR'/'Zcut' or None (real)
    alt_raw: Optional[str] # as-seen casing, for the histogram

    @property
    def identity(self) -> Identity:
        return (self.subject_id, self.hand, self.finger)

    @property
    def identity_str(self) -> str:
        return f"{self.subject_id}_{self.hand}_{self.finger}"

    @property
    def stem(self) -> str:
        return self.filename[:-4] if self.filename.lower().endswith(".bmp") else self.filename


def parse_filename(filename: str, path: Optional[str] = None) -> Optional[FpRecord]:
    """Parse one filename into an FpRecord, or None if it does not match."""
    match = FILENAME_RE.match(filename)
    if match is None:
        return None
    fields = match.groupdict()
    alt_raw = fields["alt"]
    alt = _ALT_CANON[alt_raw.lower()] if alt_raw else None
    return FpRecord(
        path=path if path is not None else filename,
        filename=filename,
        subject_id=int(fields["id"]),
        gender=fields["gender"].upper(),
        hand=fields["hand"].capitalize(),
        finger=fields["finger"].lower(),
        alt=alt,
        alt_raw=alt_raw,
    )


def scan_dir(directory: os.PathLike) -> Tuple[List[FpRecord], List[str]]:
    """Parse every file in a directory. Returns (records, unparsed_filenames).

    Filenames are processed in sorted order for deterministic downstream sampling.
    """
    directory = Path(directory)
    records: List[FpRecord] = []
    skipped: List[str] = []
    for name in sorted(os.listdir(directory)):
        full = directory / name
        if not full.is_file():
            continue
        rec = parse_filename(name, str(full))
        if rec is None:
            skipped.append(name)
        else:
            records.append(rec)
    return records, skipped


def build_gallery(real_dir: os.PathLike) -> Tuple[Dict[Identity, FpRecord], List[str], List[tuple], List[FpRecord]]:
    """Build the gallery index from the Real folder.

    Returns (gallery, skipped, collisions, unexpected_altered):
      gallery            : identity -> FpRecord (one template per finger)
      skipped            : filenames that did not parse
      collisions         : (identity, existing_filename, duplicate_filename)
      unexpected_altered : real-folder files that carry an alteration suffix
    """
    records, skipped = scan_dir(real_dir)
    gallery: Dict[Identity, FpRecord] = {}
    collisions: List[tuple] = []
    unexpected_altered: List[FpRecord] = []
    for rec in records:
        if rec.alt is not None:
            unexpected_altered.append(rec)
            continue
        if rec.identity in gallery:
            collisions.append((rec.identity, gallery[rec.identity].filename, rec.filename))
        else:
            gallery[rec.identity] = rec
    return gallery, skipped, collisions, unexpected_altered


def build_probes(level_dir: os.PathLike,
                 gallery: Dict[Identity, FpRecord]
                 ) -> Tuple[List[FpRecord], List[str], List[str], List[str]]:
    """Build the probe list for one altered level.

    Returns (probes, skipped, orphans, non_altered):
      probes      : altered records whose true identity exists in the gallery
      skipped     : filenames that did not parse
      orphans     : altered filenames whose identity is NOT in the gallery
                    (breaks the closed-set guarantee -> logged, dropped)
      non_altered : files in an altered folder that carry no alteration suffix
    """
    records, skipped = scan_dir(level_dir)
    probes: List[FpRecord] = []
    orphans: List[str] = []
    non_altered: List[str] = []
    for rec in records:
        if rec.alt is None:
            non_altered.append(rec.filename)
        elif rec.identity not in gallery:
            orphans.append(rec.filename)
        else:
            probes.append(rec)
    return probes, skipped, orphans, non_altered


def alt_casing_histogram(level_dir: os.PathLike, limit: int = 100
                         ) -> Tuple[Counter, Counter, int]:
    """Tally the raw casing of alteration tokens over the first `limit` altered files.

    Returns (casing_counts, unexpected_counts, n_examined):
      casing_counts    : Counter of the exact token strings seen (e.g. 'Obl', 'obl')
      unexpected_counts: subset whose lowercase form is not a known alteration
      n_examined       : how many altered files were inspected
    """
    casing: Counter = Counter()
    unexpected: Counter = Counter()
    examined = 0
    for name in sorted(os.listdir(level_dir)):
        if examined >= limit:
            break
        match = LOOSE_ALT_RE.search(name)
        if match is None:
            continue
        token = match.group("tok")
        casing[token] += 1
        if token.lower() not in _ALT_CANON:
            unexpected[token] += 1
        examined += 1
    return casing, unexpected, examined


def breakdown_counts(records: List[FpRecord]) -> Dict[str, Counter]:
    """Per-alteration / per-hand / per-finger counts for a record list."""
    counts: Dict[str, Counter] = {
        "alt": Counter(),
        "hand": Counter(),
        "finger": Counter(),
    }
    for rec in records:
        counts["alt"][rec.alt or "Real"] += 1
        counts["hand"][rec.hand] += 1
        counts["finger"][rec.finger] += 1
    return counts


def walk_summary(root: os.PathLike, max_depth: int = 2, max_examples: int = 3
                 ) -> List[Dict[str, object]]:
    """Depth-limited directory summary for the Kaggle dataset-discovery cell.

    Returns one entry per directory (up to max_depth below root) with the number
    of subdirectories, number of files, and a few example filenames.
    """
    root = Path(root)
    summary: List[Dict[str, object]] = []
    if not root.exists():
        return summary
    root_depth = len(root.parts)
    for current, dirs, files in os.walk(root):
        depth = len(Path(current).parts) - root_depth
        if depth > max_depth:
            dirs[:] = []  # prune deeper traversal
            continue
        dirs.sort()
        files_sorted = sorted(files)
        summary.append({
            "path": str(current),
            "depth": depth,
            "n_subdirs": len(dirs),
            "n_files": len(files),
            "examples": files_sorted[:max_examples],
        })
    return summary


def find_dataset_root(search_root: os.PathLike, real_name: str = "Real",
                      max_depth: int = 6) -> "Path | None":
    """Locate the directory that holds the gallery folder (`real_name` with BMPs).

    Robust to how the dataset is mounted, e.g. any of:
        /kaggle/input/socofing/SOCOFing
        /kaggle/input/datasets/<owner>/socofing/SOCOFing
        /kaggle/input/<slug>
    Returns the parent directory of `real_name`, or None if not found.
    """
    search_root = Path(search_root)
    if not search_root.exists():
        return None
    root_depth = len(search_root.parts)
    for current, dirs, files in os.walk(search_root):
        depth = len(Path(current).parts) - root_depth
        if depth > max_depth:
            dirs[:] = []  # prune deeper traversal
            continue
        dirs.sort()
        if real_name in dirs:
            real_path = Path(current) / real_name
            try:
                entries = os.listdir(real_path)
            except OSError:
                entries = []
            if any(e.lower().endswith(".bmp") for e in entries[:100]):
                return Path(current)
    return None
