"""Read and write JSON Lines files, one JSON object per line."""

from __future__ import annotations

import json
from collections.abc import Hashable, Iterable
from pathlib import Path
from typing import Any

Record = dict[str, Any]


def read_jsonl(path: str | Path) -> list[Record]:
    """Read every nonblank line of a JSON Lines file.

    Parameters
    ----------
    path : str or Path
        The file to read.

    Returns
    -------
    list[dict]
        One record per nonblank line.

    Raises
    ------
    ValueError
        If a line is not valid JSON. The message names the file and line.
    """
    records = []
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON") from exc
    return records


def write_jsonl(path: str | Path, records: Iterable[Record]) -> None:
    """Write records to a JSON Lines file, creating its folder if needed.

    Parameters
    ----------
    path : str or Path
        The file to write. An existing file is replaced.
    records : iterable of dict
        The records to write, one per line.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        for record in records:
            line = json.dumps(record, ensure_ascii=False, allow_nan=False)
            stream.write(line + "\n")


def unique_by(records: Iterable[Record], key: str) -> dict[Hashable, Record]:
    """Index records by one field, rejecting duplicate values.

    Parameters
    ----------
    records : iterable of dict
        The records to index.
    key : str
        The field whose value must be unique.

    Returns
    -------
    dict
        Each record, keyed by its value of ``key``.

    Raises
    ------
    ValueError
        If two records share a value.
    """
    result: dict[Hashable, Record] = {}
    for record in records:
        value = record[key]
        if value in result:
            raise ValueError(f"duplicate {key}: {value}")
        result[value] = record
    return result
