"""Build the swappable parts named in a config section.

Every part follows the starter's convention: a config section names its
class as ``"class": "module:ClassName"``, and the whole section is passed to
that class as its ``settings``.
"""

from __future__ import annotations

from typing import Any

from qasper_rag.config import ConfigurationError, load_class


def build_component(
    role: str, section: dict[str, Any] | None, default: type | None, *args: Any
) -> Any:
    """Build the class named in ``section``, or ``default`` if none is named.

    Parameters
    ----------
    role : str
        What the part does, e.g. ``"chunker"``. Used in error messages.
    section : dict or None
        The part's config section. ``None`` means use ``default``.
    default : type or None
        The class to use when ``section`` names none. ``None`` makes the
        ``class`` key required.
    *args : Any
        Extra constructor arguments, passed after ``settings``.

    Returns
    -------
    Any
        The built part.

    Raises
    ------
    ConfigurationError
        If the section is not an object, or no class is named and there is
        no default.
    """
    section = {} if section is None else section
    if not isinstance(section, dict):
        raise ConfigurationError(f"{role} settings must be a JSON object")
    part_class = load_class(section, default)
    if part_class is None:
        raise ConfigurationError(f"{role} settings must name a class")
    return part_class(section, *args)
