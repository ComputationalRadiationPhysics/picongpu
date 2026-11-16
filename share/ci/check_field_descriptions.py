#!/usr/bin/env python3
"""
Check that the pydantic PICMI models under ``lib/python/picongpu/picmi/`` keep
their field descriptions and do not re-list their fields in class docstrings.

This is the offline, dependency-free guard agreed for
https://github.com/chillenzer-agents/picongpu/issues/194. It is intentionally
stdlib-only (``ast``) and must run under Python >= 3.11 without importing
``pydantic`` or ``picongpu`` -- so it is safe to run as a pre-commit hook
without the project environment.

It reports three classes of regression:

(i)   a ``Field(...)`` call on a model field without a ``description=``
      argument (the (b-iii)/field-prose regression). A baseline of the
      pre-existing gaps is accepted until the prose PR fills them; any *new*
      bare ``Field(...)`` outside that baseline fails.
(ii)  a field that is redeclared in a subclass (shadowing a field of a PICMI
      standard base class or of a local base class) without its own
      ``description=`` (the (a1) regression of issue #194).
(iii) a class docstring that re-lists the class' own fields in numpydoc
      ``name : type``, Google-style ``name (type):`` / ``name:`` or bullet
      ``- name : type`` form (the (c) duplication regression).
(iv)  a parameter-list-looking string constant in a class body that is *not*
      the docstring (for example a list placed after a ``model_config``
      assignment). ``ast.get_docstring`` and autodoc ignore such a string, so
      its parameter list silently escapes the docstring check.

Exit status is non-zero if any finding is reported.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PICMI_DIR = REPO_ROOT / "lib" / "python" / "picongpu" / "picmi"

# Real model fields declared by the pinned picmistandard base classes
# (chillenzer/picmi@c1d66223, installed as `picmistandard`). Embedded verbatim
# so this script stays stdlib-only and offline: a redeclaration of any of these
# names in a PIConGPU subclass shadows the standard description and must carry
# its own. Only names present in the standard's ``model_fields`` belong here --
# ``ClassVar`` attributes (e.g. ``number_of_dimensions``) and properties (e.g.
# ``PICMI_MagnetostaticSolver.methods_list``) can never take a ``Field``
# description and are deliberately excluded; (ii) additionally skips
# ``ClassVar``-annotated targets as a safety net.
STANDARD_FIELDS: dict[str, frozenset[str]] = {
    "PICMI_AnalyticDistribution": frozenset(
        {
            "density_expression",
            "directed_velocity",
            "fill_in",
            "lower_bound",
            "momentum_expressions",
            "momentum_spread_expressions",
            "rms_velocity",
            "upper_bound",
            "user_defined_kw",
        }
    ),
    "PICMI_BinomialSmoother": frozenset({"alpha", "compensation", "n_pass", "stride"}),
    "PICMI_Cartesian1DGrid": frozenset(
        {
            "bc_xmax",
            "bc_xmax_particles",
            "bc_xmin",
            "bc_xmin_particles",
            "guard_cells",
            "lower_bound",
            "lower_bound_particles",
            "lower_boundary_conditions",
            "lower_boundary_conditions_particles",
            "moving_window_velocity",
            "number_of_cells",
            "nx",
            "pml_cells",
            "refined_regions",
            "upper_bound",
            "upper_bound_particles",
            "upper_boundary_conditions",
            "upper_boundary_conditions_particles",
            "xmax",
            "xmax_particles",
            "xmin",
            "xmin_particles",
        }
    ),
    "PICMI_Cartesian2DGrid": frozenset(
        {
            "bc_xmax",
            "bc_xmax_particles",
            "bc_xmin",
            "bc_xmin_particles",
            "bc_ymax",
            "bc_ymax_particles",
            "bc_ymin",
            "bc_ymin_particles",
            "guard_cells",
            "lower_bound",
            "lower_bound_particles",
            "lower_boundary_conditions",
            "lower_boundary_conditions_particles",
            "moving_window_velocity",
            "number_of_cells",
            "nx",
            "ny",
            "pml_cells",
            "refined_regions",
            "upper_bound",
            "upper_bound_particles",
            "upper_boundary_conditions",
            "upper_boundary_conditions_particles",
            "xmax",
            "xmax_particles",
            "xmin",
            "xmin_particles",
            "ymax",
            "ymax_particles",
            "ymin",
            "ymin_particles",
        }
    ),
    "PICMI_Cartesian3DGrid": frozenset(
        {
            "bc_xmax",
            "bc_xmax_particles",
            "bc_xmin",
            "bc_xmin_particles",
            "bc_ymax",
            "bc_ymax_particles",
            "bc_ymin",
            "bc_ymin_particles",
            "bc_zmax",
            "bc_zmax_particles",
            "bc_zmin",
            "bc_zmin_particles",
            "guard_cells",
            "lower_bound",
            "lower_bound_particles",
            "lower_boundary_conditions",
            "lower_boundary_conditions_particles",
            "moving_window_velocity",
            "number_of_cells",
            "nx",
            "ny",
            "nz",
            "pml_cells",
            "refined_regions",
            "upper_bound",
            "upper_bound_particles",
            "upper_boundary_conditions",
            "upper_boundary_conditions_particles",
            "xmax",
            "xmax_particles",
            "xmin",
            "xmin_particles",
            "ymax",
            "ymax_particles",
            "ymin",
            "ymin_particles",
            "zmax",
            "zmax_particles",
            "zmin",
            "zmin_particles",
        }
    ),
    "PICMI_CylindricalGrid": frozenset(
        {
            "bc_rmax",
            "bc_rmax_particles",
            "bc_rmin",
            "bc_rmin_particles",
            "bc_zmax",
            "bc_zmax_particles",
            "bc_zmin",
            "bc_zmin_particles",
            "guard_cells",
            "lower_bound",
            "lower_bound_particles",
            "lower_boundary_conditions",
            "lower_boundary_conditions_particles",
            "moving_window_velocity",
            "n_azimuthal_modes",
            "nr",
            "number_of_cells",
            "nz",
            "pml_cells",
            "refined_regions",
            "rmax",
            "rmax_particles",
            "rmin",
            "rmin_particles",
            "upper_bound",
            "upper_bound_particles",
            "upper_boundary_conditions",
            "upper_boundary_conditions_particles",
            "zmax",
            "zmax_particles",
            "zmin",
            "zmin_particles",
        }
    ),
    "PICMI_ElectromagneticSolver": frozenset(
        {
            "cfl",
            "divB_cleaning",
            "divE_cleaning",
            "field_smoother",
            "galilean_velocity",
            "grid",
            "method",
            "pml_divB_cleaning",
            "pml_divE_cleaning",
            "source_smoother",
            "stencil_order",
            "subcycling",
        }
    ),
    "PICMI_FieldDiagnostic": frozenset(
        {
            "data_list",
            "grid",
            "lower_bound",
            "name",
            "number_of_cells",
            "parallelio",
            "period",
            "step_max",
            "step_min",
            "upper_bound",
            "write_dir",
        }
    ),
    "PICMI_FoilDistribution": frozenset(
        {
            "density",
            "directed_velocity",
            "exponential_post_plasma_cutoff",
            "exponential_post_plasma_length",
            "exponential_pre_plasma_cutoff",
            "exponential_pre_plasma_length",
            "fill_in",
            "front",
            "lower_bound",
            "rms_velocity",
            "thickness",
            "upper_bound",
        }
    ),
    "PICMI_GaussianLaser": frozenset(
        {
            "E0",
            "a0",
            "beta",
            "centroid_position",
            "duration",
            "fill_in",
            "focal_position",
            "k0",
            "name",
            "phi0",
            "phi2",
            "polarization_direction",
            "propagation_direction",
            "waist",
            "wavelength",
            "zeta",
        }
    ),
    "PICMI_GriddedLayout": frozenset({"grid", "n_macroparticles_per_cell"}),
    "PICMI_ParticleDiagnostic": frozenset(
        {
            "data_list",
            "name",
            "parallelio",
            "period",
            "species",
            "step_max",
            "step_min",
            "write_dir",
        }
    ),
    "PICMI_PseudoRandomLayout": frozenset({"grid", "n_macroparticles", "n_macroparticles_per_cell", "seed"}),
    "PICMI_Simulation": frozenset(
        {
            "applied_fields",
            "diagnostics",
            "gamma_boost",
            "initialize_self_fields",
            "injection_plane_normal_vectors",
            "injection_plane_positions",
            "interactions",
            "laser_injection_methods",
            "lasers",
            "layouts",
            "load_balancing",
            "max_steps",
            "max_time",
            "particle_shape",
            "solver",
            "species",
            "time_step_size",
            "verbose",
        }
    ),
    "PICMI_Species": frozenset(
        {
            "charge",
            "charge_state",
            "density_scale",
            "initial_distribution",
            "interactions",
            "mass",
            "method",
            "methods_list",
            "name",
            "particle_shape",
            "particle_type",
        }
    ),
    "PICMI_UniformDistribution": frozenset(
        {"density", "directed_velocity", "fill_in", "lower_bound", "rms_velocity", "upper_bound"}
    ),
}

# Pre-existing ``Field(...)`` calls without ``description=`` (file, class, field),
# keyed relative to ``lib/python/picongpu/picmi``. These are the field-prose
# backlog of issue #194 item (b-iii), deliberately fixed in the follow-up prose
# PR, not here. The guard is baseline-locked: it fails on any *new* bare
# ``Field(...)`` that is not listed below. Remove entries as they are documented.
BARE_FIELD_BASELINE: frozenset[tuple[str, str, str]] = frozenset(
    {
        ("distribution/GaussianBunchDistribution.py", "GaussianBunchDistribution", "velocity_divergence"),
        ("simulation.py", "Simulation", "picongpu_min_weighting"),
        ("grid.py", "Cartesian3DGrid", "picongpu_n_gpus"),
        ("grid.py", "Cartesian3DGrid", "picongpu_grid_dist"),
        ("grid.py", "Cartesian3DGrid", "picongpu_super_cell_size"),
        ("grid.py", "Cartesian2DGrid", "picongpu_n_gpus"),
        ("grid.py", "Cartesian2DGrid", "picongpu_grid_dist"),
        ("grid.py", "Cartesian2DGrid", "picongpu_super_cell_size"),
        ("grid.py", "Cartesian2DGrid", "picongpu_cell_depth_si"),
        ("interaction/collision.py", "CollisionalPhysicsSetup", "collisions"),
        ("interaction/collision.py", "CollisionalPhysicsSetup", "screening_species"),
        ("lasers/from_openpmd_pulse_laser.py", "FromOpenPMDPulseLaser", "picongpu_huygens_surface_positions"),
        ("lasers/plane_wave_laser.py", "PlaneWaveLaser", "picongpu_huygens_surface_positions"),
        ("lasers/twts_laser.py", "TWTSLaser", "picongpu_huygens_surface_positions"),
        ("memory_config.py", "MemoryConfig", "reserved_gpu_memory_size"),
        ("memory_config.py", "MemoryConfig", "bytes_exchange_x"),
        ("memory_config.py", "MemoryConfig", "bytes_exchange_y"),
        ("memory_config.py", "MemoryConfig", "bytes_exchange_z"),
        ("memory_config.py", "MemoryConfig", "bytes_edges"),
        ("memory_config.py", "MemoryConfig", "bytes_corner"),
        ("simulation.py", "Simulation", "picongpu_custom_user_input"),
        ("simulation.py", "Simulation", "picongpu_typical_ppc"),
        ("simulation.py", "Simulation", "picongpu_template_dir"),
        ("simulation.py", "Simulation", "picongpu_moving_window_move_point"),
        ("simulation.py", "Simulation", "picongpu_moving_window_stop_iteration"),
        ("simulation.py", "Simulation", "picongpu_base_density"),
        ("simulation.py", "Simulation", "picongpu_precision"),
        ("simulation.py", "Simulation", "picongpu_precision_config"),
        ("simulation.py", "Simulation", "picongpu_memory_config"),
        ("simulation.py", "Simulation", "picongpu_walltime"),
    }
)

# The (a1) redeclaration set and the (c) docstring duplication set are fixed in
# this change, so their baselines are intentionally empty: a new occurrence is
# always a failure.
# Parameter-list sections come in two styles: numpydoc uses a bare (or dashed)
# header with ``name : type`` entries, Google style a header with an optional
# trailing colon and ``name (type):`` / ``name:`` entries. ``Args`` is the
# Google spelling used by free functions.
NUMDOC_PARAM_HEADER = re.compile(r"^[ \t]*(?:Parameters|Other Parameters|Attributes|Args)[ \t]*:?[ \t]*$", re.MULTILINE)
NUMDOC_ENTRY = re.compile(r"^\s*(\w+)\s*:\s*\S", re.MULTILINE)
GOOGLE_ENTRY = re.compile(r"^\s*([A-Za-z_]\w*)\s*(?:\([^)]*\))?\s*:(?:\s.*)?$", re.MULTILINE)
BULLET_ENTRY = re.compile(r"^\s*[-*]\s*(\w+)\s*:\s*\S", re.MULTILINE)


def _call_name(node: ast.expr) -> str | None:
    """Return the terminal name of a call target (``Field``, ``pydantic.Field``, ...)."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_field_call(node: ast.expr) -> bool:
    return isinstance(node, ast.Call) and _call_name(node.func) == "Field"


def _has_description(call: ast.Call) -> bool:
    return any(keyword.arg == "description" for keyword in call.keywords)


def _field_calls(node: ast.AnnAssign) -> list[ast.Call]:
    """All ``Field(...)`` calls attached to an annotated assignment."""
    calls: list[ast.Call] = []
    for part in (node.annotation, node.value):
        if part is not None:
            calls.extend(n for n in ast.walk(part) if _is_field_call(n))
    return calls


def _annotated_field_name(node: ast.AnnAssign) -> str | None:
    target = node.target
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return None


def _is_classvar(node: ast.AnnAssign) -> bool:
    """True if the annotation's outer subscript is ``ClassVar``/``typing.ClassVar``.

    ``ClassVar`` attributes are not pydantic model fields and can never take a
    ``Field(description=...)``; flagging them as undescribed shadowed fields
    would be a false positive (e.g. the standard grid's
    ``number_of_dimensions: ClassVar[int]``).
    """
    annotation = node.annotation
    if isinstance(annotation, ast.Subscript):
        annotation = annotation.value
    return _call_name(annotation) == "ClassVar"


def _base_names(node: ast.ClassDef) -> list[str]:
    names = []
    for base in node.bases:
        name = _call_name(base)
        if name is not None:
            names.append(name)
    return names


class _ClassInfo:
    def __init__(self, node: ast.ClassDef):
        self.node = node
        self.name = node.name
        self.fields: dict[str, ast.AnnAssign] = {}
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and not _is_classvar(stmt):
                field_name = _annotated_field_name(stmt)
                if field_name is not None:
                    self.fields[field_name] = stmt


def _collect_file(path: Path, source: str) -> list[_ClassInfo]:
    tree = ast.parse(source, filename=str(path))
    return [_ClassInfo(node) for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]


def _local_ancestor_fields(
    cls: _ClassInfo, registry: dict[str, _ClassInfo], seen: frozenset[str] = frozenset()
) -> set[str]:
    """Names of all fields declared by local (same-file) ancestors of ``cls``."""
    inherited: set[str] = set()
    for base_name in _base_names(cls.node):
        if base_name in seen or base_name not in registry:
            continue
        base = registry[base_name]
        inherited.update(base.fields)
        inherited.update(_local_ancestor_fields(base, registry, seen | {base_name}))
    return inherited


def _section_listed_names(text: str) -> set[str]:
    """Names listed as entries of a numpydoc or Google-style parameter section."""
    names: set[str] = set()
    header = NUMDOC_PARAM_HEADER.search(text)
    if header:
        lines = text[header.end() :].splitlines()
        start = 0
        while start < len(lines) and not lines[start].strip():
            start += 1
        if start < len(lines) and set(lines[start].strip()) == {"-"}:
            start += 1
        # A numpydoc section lists "name : type" entries, a Google-style section
        # "name (type):" / "name:" entries; entries may be indented or not and are
        # followed by (possibly indented) prose. The section ends at the first
        # unindented, non-blank line that is not itself an entry.
        for line in lines[start:]:
            entry = NUMDOC_ENTRY.match(line) or GOOGLE_ENTRY.match(line)
            if entry:
                names.add(entry.group(1))
                continue
            if line.strip() and line[0] not in " \t":
                break
    names.update(BULLET_ENTRY.findall(text))
    return names


def _docstring_listed_names(node: ast.ClassDef) -> set[str]:
    doc = ast.get_docstring(node)
    return _section_listed_names(doc) if doc else set()


def _has_parameter_list(text: str) -> bool:
    """True if ``text`` looks like a parameter-list section (header plus entries)."""
    return NUMDOC_PARAM_HEADER.search(text) is not None and bool(_section_listed_names(text))


def _class_body_parameter_lists(node: ast.ClassDef) -> list[int]:
    """Line numbers of class-body strings that look like parameter lists.

    Only the first statement can be the class docstring; any other standalone
    string is inert (``ast.get_docstring`` ignores it and autodoc renders
    nothing), so a parameter list hidden there silently escapes the docstring
    check -- the ``checkpoint.py`` bug this guard exists to catch.
    """
    lines: list[int] = []
    for index, stmt in enumerate(node.body):
        if index == 0:
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            if _has_parameter_list(stmt.value.value):
                lines.append(stmt.value.lineno)
    return lines


def check(directory: Path) -> list[str]:
    findings: list[str] = []
    for path in sorted(directory.rglob("*.py")):
        rel = path.relative_to(directory).as_posix()
        source = path.read_text(encoding="utf-8")
        classes = _collect_file(path, source)
        registry = {cls.name: cls for cls in classes}
        for cls in classes:
            if cls.name.startswith("_"):
                continue
            local_ancestors = _local_ancestor_fields(cls, registry, frozenset({cls.name}))
            for field_name, stmt in cls.fields.items():
                calls = _field_calls(stmt)
                described = any(_has_description(call) for call in calls)
                # (i) bare Field(...) without description
                for call in calls:
                    if not _has_description(call) and (rel, cls.name, field_name) not in BARE_FIELD_BASELINE:
                        findings.append(
                            f"{rel}:{call.lineno}: {cls.name}.{field_name}: Field(...) is missing description="
                        )
                # (ii) redeclaration of a standard or local-base field without description
                inherited = field_name in STANDARD_FIELDS.get(
                    next((b for b in _base_names(cls.node) if b in STANDARD_FIELDS), ""), frozenset()
                )
                inherited = inherited or field_name in local_ancestors
                if inherited and not described:
                    findings.append(
                        f"{rel}:{stmt.lineno}: {cls.name}.{field_name}: "
                        f"redeclares an inherited field without its own description="
                    )
            # (iii) class docstring re-lists own fields
            listed = _docstring_listed_names(cls.node) & set(cls.fields)
            if listed:
                findings.append(
                    f"{rel}:{cls.node.lineno}: {cls.name}: class docstring re-lists its own "
                    f"fields {sorted(listed)}; move the prose to Field(description=...)"
                )
            # (iv) parameter list in a class-body string that is not the docstring
            for lineno in _class_body_parameter_lists(cls.node):
                findings.append(
                    f"{rel}:{lineno}: {cls.name}: parameter list in a class-body string that is not "
                    f"the docstring; make it the docstring or move the prose to Field(description=...)"
                )
    return findings


def _validate_baseline(directory: Path) -> list[str]:
    """Check that every baseline entry names a real, still-bare own ``Field``.

    A baseline tuple is only meaningful while its field genuinely lacks a
    ``description=``. This self-check makes stale tuples (a field that was
    documented, removed, renamed or never declared locally) impossible to keep:
    they fail the guard at startup instead of silently widening the baseline.
    """
    problems: list[str] = []
    registries: dict[str, dict[str, _ClassInfo]] = {}
    for rel, cls_name, field_name in sorted(BARE_FIELD_BASELINE):
        if rel not in registries:
            path = directory / rel
            registries[rel] = (
                {cls.name: cls for cls in _collect_file(path, path.read_text(encoding="utf-8"))}
                if path.is_file()
                else {}
            )
        cls = registries[rel].get(cls_name)
        if cls is None:
            problems.append(f"{rel}: baseline entry {cls_name}.{field_name}: file or class not found")
            continue
        stmt = cls.fields.get(field_name)
        if stmt is None:
            problems.append(f"{rel}: baseline entry {cls_name}.{field_name}: no such own field")
            continue
        calls = _field_calls(stmt)
        if not calls:
            problems.append(f"{rel}: baseline entry {cls_name}.{field_name}: field has no Field(...) call")
        elif any(_has_description(call) for call in calls):
            problems.append(
                f"{rel}: baseline entry {cls_name}.{field_name}: field now has description=; drop the entry"
            )
    return problems


def main(argv: list[str]) -> int:
    directory = Path(argv[1]) if len(argv) > 1 else PICMI_DIR
    if not directory.is_dir():
        print(f"check_field_descriptions: not a directory: {directory}", file=sys.stderr)
        return 2
    problems = _validate_baseline(directory)
    if problems:
        print(
            "check_field_descriptions: baseline self-check failed (stale BARE_FIELD_BASELINE entries):",
            file=sys.stderr,
        )
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    findings = check(directory)
    if findings:
        print("check_field_descriptions: field-description guard failed:", file=sys.stderr)
        for finding in findings:
            print(f"  {finding}", file=sys.stderr)
        return 1
    print("check_field_descriptions: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
