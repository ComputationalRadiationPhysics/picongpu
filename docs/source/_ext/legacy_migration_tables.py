"""Sphinx extension: build-time legacy ``.param``/``.cfg`` migration tables.

The tables answer one question for users migrating a legacy C++ setup:

    If I changed legacy file *A*, which PICMI input classes can currently
    produce content in it?

Instead of maintaining that mapping by hand, it is derived at documentation
build time from the two sources of truth in this repository:

1. **The mustache templates** in ``lib/python/picongpu/templates`` say which
   rendering-context paths end up in which generated file.
2. **The PyPIConGPU models** in ``lib/python/picongpu/pypicongpu`` define the
   rendering context. Starting from ``pypicongpu.simulation.Simulation``, the
   extension resolves every template path against the pydantic model graph. This
   yields the concrete backend classes that contribute to each file.
3. **The PICMI frontend** in ``lib/python/picongpu/picmi`` converts user input
   into those backend classes. The extension reads the conversion targets from
   the ``@converts_to`` decorators (via their closure) and from the bodies of
   hand written ``get_as_pypicongpu`` methods (AST), including PICMI classes
   that are themselves converted further. Frontend subclasses of backend models
   are picked up as well.
4. A small, documented ownership map (``MIGRATION_OWNERS``) covers backend
   objects that the frontend assembles indirectly (species init operations, the
   openPMD plugin built from dump diagnostics), so that they still point at the
   PICMI class a user actually writes.

The tables are emitted only into the doctree/HTML; nothing is written to disk.
Three directives are registered:

``.. legacy-migration-table:: files``
    The per-file table (``include/picongpu/param/*.param``), columns legacy file
    and the PICMI classes that can produce its content.

``.. legacy-migration-table:: cfg``
    A separate table with one row per PICMI class that renders into
    ``etc/picongpu/*.cfg``.

``.. legacy-migration-table:: missing``
    The legacy parameter files that have no dedicated template.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import re
import sys
import textwrap
import types
import typing
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "lib" / "python"

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from docutils import nodes  # noqa: E402
from docutils.parsers.rst import Directive  # noqa: E402
from docutils.statemachine import StringList  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import picongpu.picmi as picmi_root  # noqa: E402
import picongpu.pypicongpu as pypicongpu_root  # noqa: E402
from picongpu.picmi.diagnostics.field_dump import DerivedFieldDump, NativeFieldDump  # noqa: E402
from picongpu.picmi.diagnostics.particle_dump import ParticleDump  # noqa: E402
from picongpu.picmi.species import Species as PicmiSpecies  # noqa: E402
from picongpu.pypicongpu.output.openpmd_plugin import OpenPMDPlugin  # noqa: E402
from picongpu.pypicongpu.simulation import Simulation  # noqa: E402
from picongpu.pypicongpu.species.operation.momentum.drift import Drift  # noqa: E402
from picongpu.pypicongpu.species.operation.momentum.temperature import Temperature  # noqa: E402
from picongpu.pypicongpu.species.operation.setchargestate import SetChargeState  # noqa: E402
from picongpu.pypicongpu.species.operation.simpledensity import SimpleDensity  # noqa: E402
from picongpu.pypicongpu.species.operation.simplemomentum import SimpleMomentum  # noqa: E402

TEMPLATE_ROOT = REPO_ROOT / "lib/python/picongpu/templates"
PARAM_ROOT = REPO_ROOT / "include/picongpu/param"

_CFG_TEMPLATE_PREFIX = ("etc", "picongpu")
_CFG_CAPTION = "PICMI input classes that render into ``etc/picongpu/*.cfg``"

_SCALAR = object()

# Backend objects that the PICMI frontend creates indirectly, i.e. without a
# one-to-one ``get_as_pypicongpu`` converter. Map each to the PICMI class(es) a
# user would edit instead. The generator asserts that every key really occurs in
# a template mapping, so stale entries are noticed.
MIGRATION_OWNERS = {
    SimpleDensity: {PicmiSpecies},
    SimpleMomentum: {PicmiSpecies},
    SetChargeState: {PicmiSpecies},
    Drift: {PicmiSpecies},
    Temperature: {PicmiSpecies},
    OpenPMDPlugin: {NativeFieldDump, DerivedFieldDump, ParticleDump},
}


@dataclass(frozen=True)
class Contribution:
    """One class/field lookup while resolving a template path."""

    model: type
    chain: tuple[type, ...]
    result: frozenset


@dataclass(frozen=True)
class Row:
    """One generated table row."""

    label: str
    picmi_classes: tuple[str, ...]


def _iter_classes(package) -> typing.Iterator[type]:
    """Yield classes defined directly in every module below *package*."""

    for module_info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        try:
            module = importlib.import_module(module_info.name)
        except Exception:
            # Optional dependencies or environment-specific modules are not
            # relevant for the mapping.
            continue
        for name, value in vars(module).items():
            if inspect.isclass(value) and value.__module__ == module_info.name and not name.startswith("_"):
                yield value


def _resolve_expression(node: ast.AST, module, depth: int = 0) -> object | None:
    """Resolve a simple ``Name``/``Attribute`` chain in *module*'s namespace."""

    if depth > 16:
        return None
    if isinstance(node, ast.Name):
        return vars(module).get(node.id)
    if isinstance(node, ast.Attribute):
        base = _resolve_expression(node.value, module, depth + 1)
        if base is not None:
            return getattr(base, node.attr, None)
    return None


def _class_body(cls: type) -> list[ast.stmt]:
    """Return the AST body of a class definition, or an empty list."""

    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
    except (OSError, SyntaxError):
        return []
    if not tree.body or not isinstance(tree.body[0], ast.ClassDef):
        return []
    return tree.body[0].body


def _conversion_targets(cls: type) -> set:
    """Return the types that ``cls.get_as_pypicongpu`` constructs or returns."""

    method = cls.__dict__.get("get_as_pypicongpu")
    if method is not None and getattr(method, "__closure__", None):
        # Classes decorated with ``@converts_to``/``@default_converts_to``
        # capture their target class in the closure.
        for name, cell in zip(method.__code__.co_freevars, method.__closure__):
            if name == "to_class":
                return {cell.cell_contents}

    module = inspect.getmodule(cls)
    targets = set()
    for node in _class_body(cls):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != "get_as_pypicongpu":
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                target = _resolve_expression(child.func, module)
                if inspect.isclass(target):
                    targets.add(target)
            elif isinstance(child, ast.Return) and isinstance(child.value, ast.Attribute):
                # ``return self.<field>``: use the field annotation.
                base = child.value.value
                if isinstance(base, ast.Name) and base.id == "self" and hasattr(cls, "model_fields"):
                    field = cls.model_fields.get(child.value.attr)
                    if field is not None:
                        targets.add(field.annotation)
    return targets


def _normalize_targets(targets, seen: set | None = None) -> set:
    """Flatten unions/annotations and follow conversions through PICMI classes."""

    result: set = set()
    seen = seen if seen is not None else set()
    for target in targets:
        origin = typing.get_origin(target)
        if origin is typing.Union or origin is types.UnionType:
            result |= _normalize_targets(typing.get_args(target), seen)
            continue
        if origin is typing.Annotated:
            result |= _normalize_targets((typing.get_args(target)[0],), seen)
            continue
        if not isinstance(target, type) or target in seen:
            continue
        seen.add(target)
        if target.__module__.startswith("picongpu.pypicongpu"):
            if issubclass(target, BaseModel):
                result.add(target)
        elif target.__module__.startswith("picongpu.picmi"):
            # A PICMI class converting into another PICMI class (for example
            # GaussianBunchDistribution -> AnalyticDistribution): keep following.
            result |= _normalize_targets(_conversion_targets(target), seen)
    return result


def _build_producers(picmi_classes: set, pypicongpu_classes: set) -> dict:
    """Map backend class -> PICMI classes that can construct it."""

    producers: dict[type, set[type]] = {}
    for cls in picmi_classes:
        for target in _normalize_targets(_conversion_targets(cls)):
            if target not in pypicongpu_classes:
                continue
            if cls is picmi_root.Simulation and target is not Simulation:
                # The PICMI Simulation assembles every sub-model, but a user
                # editing e.g. ``collision.param`` writes a Collision, not a
                # Simulation. Simulation is attributed only where a template
                # consumes a scalar field defined directly on it (see
                # _picmi_classes_for).
                continue
            producers.setdefault(target, set()).add(cls)

    # Frontend homes that directly subclass a backend model (e.g.
    # ``picmi.interaction.ConstLogCollision``) are producers of that model too.
    for cls in picmi_classes:
        for base in cls.__mro__[1:]:
            if base in pypicongpu_classes:
                producers.setdefault(base, set()).add(cls)

    # Propagate to backend base classes so that template references to a base
    # (e.g. IonizationModel) are attributed to the concrete frontend classes.
    for cls, cls_producers in list(producers.items()):
        for base in cls.__mro__[1:]:
            if base in pypicongpu_classes:
                producers.setdefault(base, set()).update(cls_producers)
    return producers


def _model_fields(cls: type) -> dict[str, object]:
    """Serialized field/annotation map of a pydantic model."""

    fields: dict[str, object] = {}
    for name, field in cls.model_fields.items():
        fields[name] = field.annotation
    for name, field in getattr(cls, "model_computed_fields", {}).items():
        fields[name] = getattr(field, "return_type", None) or getattr(field, "annotation", None)
    return fields


def _annotation_members(annotation, depth: int = 0) -> set:
    """Possible values of an annotation as model classes and/or ``_SCALAR``."""

    if annotation is None or depth > 16:
        return set()
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        members: set = set()
        for argument in typing.get_args(annotation):
            members |= _annotation_members(argument, depth + 1)
        return members
    if origin is typing.Annotated:
        return _annotation_members(typing.get_args(annotation)[0], depth + 1)
    if origin in (list, set, tuple, frozenset):
        members = set()
        for argument in typing.get_args(annotation):
            members |= _annotation_members(argument, depth + 1)
        return members
    if origin is not None:
        return {_SCALAR}
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return {annotation}
    return {_SCALAR}


def _navigate(context: dict, parts: list[str]) -> tuple[dict, list[Contribution]]:
    """Resolve a dotted mustache path through the model graph."""

    contributions: list[Contribution] = []
    current = dict(context)
    for part in parts:
        following: dict = {}
        for model, chain in current.items():
            if model is _SCALAR or not (isinstance(model, type) and issubclass(model, BaseModel)):
                continue
            fields = _model_fields(model)
            if part not in fields:
                continue
            result = frozenset(_annotation_members(fields[part]))
            contributions.append(Contribution(model, chain, result))
            for member in result:
                if member is _SCALAR:
                    following.setdefault(_SCALAR, chain)
                else:
                    following.setdefault(member, chain + (member,))
        current = following
        if not current:
            break
    return current, contributions


_MUSTACHE_TAG = re.compile(r"\{\{\{?\s*([#^/&]?)\s*([^}]+?)\s*\}?\}\}")


def _template_contributions(text: str) -> list[Contribution]:
    """Interpret a mustache template and collect its context lookups."""

    stack: list[dict] = [{Simulation: (Simulation,)}]
    contributions: list[Contribution] = []
    for match in _MUSTACHE_TAG.finditer(text):
        sigil, path = match.group(1), match.group(2).strip()
        if not path or path.startswith("!"):
            continue
        if sigil == "/":
            if len(stack) > 1:
                stack.pop()
            continue

        parts = path.split(".")
        if path == "." or any(part.startswith("_") for part in parts):
            results = {_SCALAR: ()}
        else:
            results, found = _navigate(stack[-1], parts)
            contributions.extend(found)

        if sigil in ("#", "^"):
            if path.startswith("type_"):
                # ``type_*`` fields are literal discriminators. Inverted
                # sections select the models that do not carry the marker.
                if sigil == "#":
                    section = {
                        model: chain
                        for model, chain in stack[-1].items()
                        if model is not _SCALAR and path in _model_fields(model)
                    }
                else:
                    section = {
                        model: chain
                        for model, chain in stack[-1].items()
                        if model is not _SCALAR and path not in _model_fields(model)
                    }
            else:
                # Scalar sections (booleans, numbers) do not change the
                # mustache context; model sections do.
                section = {model: chain for model, chain in results.items() if model is not _SCALAR}
            stack.append(section if section else stack[-1])
    return contributions


def _picmi_classes_for(
    contribution: Contribution,
    producers: dict,
    owners: dict,
) -> set:
    """Return the PICMI classes attributed to one contribution."""

    model = contribution.model
    mapped: set = set()

    if model is Simulation:
        # PIConGPU's ``Simulation`` is the root of every rendering context. Only
        # count it when a template consumes a plain scalar field defined directly
        # on it (e.g. ``delta_t_si``), not when it merely reaches an optional or
        # nested sub-model such as ``output`` or ``laser``.
        if contribution.result == {_SCALAR}:
            mapped.add(picmi_root.Simulation)
        return mapped

    mapped |= producers.get(model, set())
    for ancestor in reversed(contribution.chain[:-1]):
        if ancestor is Simulation:
            continue
        if ancestor in producers:
            mapped |= producers[ancestor]
            break
    mapped |= owners.get(model, set())
    return mapped


def _display_name(cls: type) -> str:
    """Shortest import path under ``picmi`` for a class."""

    if getattr(picmi_root, cls.__name__, None) is cls:
        return f"picmi.{cls.__name__}"
    for submodule_name in (
        "diagnostics",
        "interaction",
        "lasers",
        "distribution",
        "particle_functor",
        "layout",
        "grid",
        "solver",
        "species",
        "multi_species",
        "applied_field",
        "memory_config",
        "precision_config",
        "unit_dimension",
    ):
        submodule = getattr(picmi_root, submodule_name, None)
        if submodule is not None and getattr(submodule, cls.__name__, None) is cls:
            return f"picmi.{submodule_name}.{cls.__name__}"
    relative = cls.__module__.removeprefix("picongpu.picmi.")
    return f"picmi.{relative}.{cls.__name__}"


def _template_relative_path(template: Path) -> Path:
    return template.relative_to(TEMPLATE_ROOT)


def _legacy_label(relative: Path) -> str:
    """Human-readable legacy path for a template file."""

    path = relative.with_suffix("")  # drop ".mustache"
    parts = path.parts
    if parts[:2] == _CFG_TEMPLATE_PREFIX:
        # The filename "N" is the per-setup number used by pic-create.
        return "``etc/picongpu/*.cfg`` (TBG macros)"
    if parts[:3] == ("include", "picongpu", "param"):
        return f"``include/picongpu/param/{parts[3]}``"
    return "``" + str(path) + "``"


def _build_rows(picmi_classes: set, pypicongpu_classes: set) -> tuple[list[Row], list[str], list[str]]:
    """Return ``(file rows, cfg PICMI classes, untemplated param files)``."""

    producers = _build_producers(picmi_classes, pypicongpu_classes)
    owners = MIGRATION_OWNERS

    file_rows: list[Row] = []
    cfg_classes: set[str] = set()
    used_owner_keys: set = set()
    for template in sorted(TEMPLATE_ROOT.rglob("*.mustache")):
        relative = _template_relative_path(template)
        contributions = _template_contributions(template.read_text())
        classes: set = set()
        for contribution in contributions:
            used_owner_keys.add(contribution.model)
            if contribution.model.__name__.startswith("_"):
                continue
            classes |= _picmi_classes_for(contribution, producers, owners)
        display_names = tuple(sorted({_display_name(cls) for cls in classes}))
        if relative.parts[:2] == _CFG_TEMPLATE_PREFIX:
            # The per-setup .cfg file is presented as its own table, one row per
            # PICMI class, so drop the repeated legacy file label.
            cfg_classes.update(display_names)
            continue
        file_rows.append(Row(_legacy_label(relative), display_names))

    stale = sorted(cls.__name__ for cls in MIGRATION_OWNERS if cls not in used_owner_keys)
    if stale:
        raise RuntimeError("MIGRATION_OWNERS entries no longer occur in any template: " + ", ".join(stale))

    templated = {
        relative.with_suffix("").parts[-1]
        for relative in (_template_relative_path(t) for t in TEMPLATE_ROOT.rglob("*.mustache"))
        if relative.parts[:3] == ("include", "picongpu", "param")
    }
    unsupported = sorted(f"include/picongpu/param/{path.name}" for path in PARAM_ROOT.glob("*.param"))
    unsupported = [path for path in unsupported if Path(path).name not in templated]
    return file_rows, sorted(cfg_classes), unsupported


@lru_cache(maxsize=1)
def _build_tables() -> tuple[list[Row], list[str], list[str]]:
    """Compute the tables once per build."""

    picmi_classes = {cls for cls in _iter_classes(picmi_root) if cls.__module__.startswith("picongpu.picmi")}
    pypicongpu_classes = {
        cls for cls in _iter_classes(pypicongpu_root) if cls.__module__.startswith("picongpu.pypicongpu")
    }
    return _build_rows(picmi_classes, pypicongpu_classes)


def _list_table(
    caption: str,
    header: tuple[str, ...],
    rows: list[tuple[str, ...]],
    widths: str | None = None,
) -> list[str]:
    """Render a ``list-table`` directive as RST lines."""

    lines = [f".. list-table:: {caption}", "   :header-rows: 1"]
    if widths is not None:
        lines.append(f"   :widths: {widths}")
    lines.append("")
    for cells in (header, *rows):
        for index, cell in enumerate(cells):
            prefix = "   * - " if index == 0 else "     - "
            lines.append(prefix + cell)
    lines.append("")
    return lines


def _files_table() -> list[str]:
    file_rows, _, _ = _build_tables()
    rows = [(row.label, ", ".join(f"``{name}``" for name in row.picmi_classes) or "*(none)*") for row in file_rows]
    return _list_table(
        "Low-level files that the Python package renders",
        ("Low-level file", "PICMI input classes that can produce its content"),
        rows,
        widths="38 62",
    )


def _cfg_table() -> list[str]:
    _, cfg_classes, _ = _build_tables()
    rows = [(f"``{name}``",) for name in cfg_classes]
    return _list_table(_CFG_CAPTION, ("PICMI input class",), rows)


def _missing_table() -> list[str]:
    _, _, unsupported = _build_tables()
    rows = [(f"``{path}``", "no dedicated template") for path in unsupported]
    return _list_table(
        "Low-level parameter files without a dedicated template",
        ("Low-level file", "Status"),
        rows,
        widths="38 62",
    )


class LegacyMigrationTableDirective(Directive):
    """Render one of the generated legacy-migration tables."""

    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False

    _TABLES = {
        "files": _files_table,
        "cfg": _cfg_table,
        "missing": _missing_table,
    }

    def run(self):
        kind = self.arguments[0].strip()
        if kind not in self._TABLES:
            raise self.error(
                f'invalid argument "{kind}" to "legacy-migration-table"; '
                f"expected one of: {', '.join(sorted(self._TABLES))}"
            )
        container = nodes.Element()
        self.state.nested_parse(StringList(self._TABLES[kind]()), 0, container)
        return container.children


def setup(app):
    """Register the ``legacy-migration-table`` directive."""

    app.add_directive("legacy-migration-table", LegacyMigrationTableDirective)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
