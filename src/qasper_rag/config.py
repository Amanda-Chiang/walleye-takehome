"""Load and check the JSON settings file, e.g. ``config/baseline.json``."""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qasper_rag.validation.contracts import MAX_RETRIEVED_PASSAGES


class ConfigurationError(ValueError):
    """The settings file is missing or holds an invalid value."""


@dataclass(frozen=True)
class AppConfig:
    """Settings for one run of the pipeline.

    Attributes
    ----------
    top_k : int
        How many passages to retrieve and give the model, at most 5.
    retriever : dict
        Settings for the retriever. ``embedding`` holds the embedding model
        settings, and an optional ``class`` names a custom retriever.
    generator : dict
        Settings for the generator. An optional ``class`` names a custom
        generator; otherwise the chat model settings are used.
    service : dict
        Optional. ``class`` names a ``RAGService`` subclass, and the other
        keys are passed to it as ``settings``.
    """

    top_k: int
    retriever: dict[str, Any]
    generator: dict[str, Any]
    service: dict[str, Any] = field(default_factory=dict)

    @property
    def embedding(self) -> dict[str, Any] | None:
        """The embedding settings, or None if the retriever needs none."""
        return self.retriever.get("embedding")


def parse_config(raw: Any) -> AppConfig:
    """Check a decoded settings object and build an ``AppConfig``.

    Parameters
    ----------
    raw : Any
        The decoded JSON.

    Returns
    -------
    AppConfig
        The checked settings.

    Raises
    ------
    ConfigurationError
        If a required section is missing or ``top_k`` is out of range.
    """
    if not isinstance(raw, dict):
        raise ConfigurationError("configuration root must be a JSON object")
    top_k = raw.get("top_k", 5)
    if type(top_k) is not int or not 1 <= top_k <= MAX_RETRIEVED_PASSAGES:
        raise ConfigurationError(
            f"top_k must be an integer from 1 to {MAX_RETRIEVED_PASSAGES}"
        )
    retriever = raw.get("retriever")
    generator = raw.get("generator")
    service = raw.get("service", {})
    sections = (
        ("retriever", retriever),
        ("generator", generator),
        ("service", service),
    )
    for name, section in sections:
        if not isinstance(section, dict):
            raise ConfigurationError(f"{name} must be a JSON object")
        if not isinstance(section.get("class", ""), str):
            raise ConfigurationError(f"{name}.class must be a string")
    embedding = retriever.get("embedding")
    if "class" not in retriever and not isinstance(embedding, dict):
        raise ConfigurationError("retriever.embedding must be a JSON object")
    return AppConfig(
        top_k=top_k, retriever=retriever, generator=generator, service=service
    )


def load_class(section: dict[str, Any], default: type) -> type:
    """Return the class named by ``section["class"]``, or ``default``.

    Parameters
    ----------
    section : dict
        A ``retriever``, ``generator`` or ``service`` settings section.
        ``class`` has the form ``"package.module:ClassName"``. The module is
        looked up in the current folder first, then in installed packages,
        so a ``my_rag/`` folder in the project root works.
    default : type
        The built-in class to use when ``class`` is missing.

    Returns
    -------
    type
        The class to build the stage with.

    Raises
    ------
    ConfigurationError
        If the name is malformed or the class cannot be found.
    """
    name = section.get("class")
    if not name:
        return default
    module_name, _, class_name = name.partition(":")
    if not module_name or not class_name:
        raise ConfigurationError(f"class must look like module:Class: {name}")
    # The installed qasper-rag command does not search the current folder.
    if "" not in sys.path and str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    try:
        return getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError) as exc:
        raise ConfigurationError(f"cannot load class {name}: {exc}") from exc


def load_config(path: str | Path) -> AppConfig:
    """Read and check a settings file.

    Parameters
    ----------
    path : str or Path
        The JSON settings file.

    Returns
    -------
    AppConfig
        The checked settings.

    Raises
    ------
    ConfigurationError
        If the file cannot be read, is not JSON, or holds an invalid value.
    """
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"cannot read config {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            f"invalid JSON in config {source}:{exc.lineno}:{exc.colno}"
        ) from exc
    try:
        return parse_config(raw)
    except ConfigurationError as exc:
        raise ConfigurationError(f"{source}: {exc}") from exc
