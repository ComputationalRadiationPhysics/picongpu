# /// script
# dependencies = [
#   "picongpu @ git+https://github.com/ComputationalRadiationPhysics/picongpu@dev#subdirectory=lib/python"
# ]
# ///
"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+

Best-effort dependency auto-installation for a PIConGPU preset.

``pic-deps`` is a thin, preset-agnostic driver over the preset's own
``dependencies_autoinstall.sh`` script. It reads everything from the standard
``picongpu.rc_params`` discovery mechanism (no ``--preset`` argument), renders a
sourceable profile, and either runs the script (``install``) or verifies the
dependency directories the script owns (``check``).

A user-supplied script can be selected instead of the preset's own with the
``pic_deps_script`` runtime configuration parameter or the ``--script``
command-line option. It then works without a preset as long as a profile is
configured.
"""

import argparse
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

from picongpu import rc_params as _rc_params_default
from picongpu._rc_params import PRESET_STORAGE_PATH, get_preset_path
from picongpu.pypicongpu.runner import generate_bare_profile

__all__ = [
    "CUSTOM_SCRIPT_KEY",
    "main",
    "resolve_preset_script",
    "parse_guard_roots",
    "render_profile",
    "check_dependencies",
    "install_dependencies",
]

CUSTOM_SCRIPT_KEY = "pic_deps_script"

_DESC = (
    "pic-deps -- best-effort dependency auto-installation for a PIConGPU preset\n"
    "\n"
    "Installs (or checks) the compile-time dependencies of the preset named in"
    " the current .picongpurc.toml by running the preset's own"
    " dependencies_autoinstall.sh against a rendered profile.\n"
    "A custom script can be selected with --script or the `pic_deps_script`"
    " rc parameter.\n"
    "\n"
    "Subcommands:\n"
    "  install   run the preset's dependencies_autoinstall.sh (best-effort)\n"
    "  check     verify each dependency directory the script owns\n"
)

# Matches the idempotency guard `if [ ! -d "$X_ROOT" ];` (and tolerates the
# historical buggy form that omitted the `$`, e.g. `if [ ! -d "X_ROOT" ];`).
_GUARD_RE = re.compile(r'! -d\s+"?\$?([A-Za-z0-9_]+_ROOT)"?\s*\]')


def _backend_suffix(profile: Path) -> str:
    """Return the dependency-backend suffix encoded in a profile file name.

    ``spack_picongpu.profile.example`` -> ``spack``,
    ``conda_picongpu.profile.example`` -> ``conda``, and the default
    ``bash_picongpu.profile.example`` -> ``""``.
    """
    stem = profile.name
    for suffix in (".example", ".profile"):
        stem = stem.removesuffix(suffix)
    stem = stem.removesuffix("_picongpu")
    return "" if stem == "bash" else stem


def _resolve_custom_script(rcp, script):
    """Return the validated user-supplied installer path, or ``None`` if none is configured.

    ``script`` (the ``--script`` option) wins over the ``pic_deps_script`` rc
    parameter. A relative path is resolved against the current working directory.
    """
    requested = script if script is not None else rcp.get(CUSTOM_SCRIPT_KEY)
    if not requested:
        return None
    path = Path(str(requested)).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_file():
        origin = "the --script option" if script is not None else f"`{CUSTOM_SCRIPT_KEY}`"
        raise SystemExit(
            f"error: {origin} points to '{requested}', which is not a file.\n"
            f"       Set `{CUSTOM_SCRIPT_KEY}` in your .picongpurc.toml or pass --script PATH."
        )
    return path


def resolve_preset_script(rcp, script=None):
    """Fail-fast validation: return (preset_dir, script_path) or raise SystemExit.

    A user-supplied script (``script`` or the ``pic_deps_script`` rc parameter)
    wins over the preset's own installer and does not require a preset. Without
    one, errors when (a) no preset is set in ``rc_params``, (b) the preset name
    does not resolve, or (c) the resolved preset has no dependency installer
    script. A variant profile (``spack_picongpu``/``conda_picongpu``) chooses
    ``dependencies_autoinstall_<suffix>.sh``; otherwise the shared
    ``dependencies_autoinstall.sh`` is used.
    """
    custom = _resolve_custom_script(rcp, script)
    if custom is not None:
        preset_dir = str(rcp.preset).split("/")[0] if rcp.preset else "custom"
        return preset_dir, custom
    preset = rcp.preset
    if not preset:
        raise SystemExit(
            "error: no preset is set in rc_params and no custom dependency installer is configured.\n"
            "       Run 'picrc-builder' (or set `preset` in your .picongpurc.toml) first,\n"
            f"       or set `{CUSTOM_SCRIPT_KEY}` to your own dependencies_autoinstall*.sh."
        )
    try:
        profile = get_preset_path(preset)
    except ValueError as error:
        raise SystemExit(f"error: could not resolve preset '{preset}': {error}") from error
    preset_dir = profile.parent.relative_to(PRESET_STORAGE_PATH).as_posix()
    suffix = _backend_suffix(profile)
    preferred = profile.parent / f"dependencies_autoinstall_{suffix}.sh" if suffix else None
    fallback = profile.parent / "dependencies_autoinstall.sh"
    script = preferred if preferred is not None and preferred.is_file() else fallback
    if not script.is_file():
        available = sorted(p.name for p in profile.parent.glob("dependencies_autoinstall*.sh"))
        available_text = ", ".join(available) if available else "(none)"
        tried = [preferred.name] if preferred is not None else []
        tried.append(fallback.name)
        raise SystemExit(
            f"error: preset '{preset_dir}' has no dependency installer script "
            f"(tried {', '.join(tried)}).\n"
            f"       Available scripts: {available_text}.\n"
            "       This preset is not supported by pic-deps."
        )
    return preset_dir, script


def parse_guard_roots(script: Path) -> list[str]:
    """Return the ordered, de-duplicated ``*_ROOT`` vars the autoinstall script guards on."""
    seen: set[str] = set()
    roots: list[str] = []
    for root in _GUARD_RE.findall(script.read_text()):
        if root not in seen:
            seen.add(root)
            roots.append(root)
    return roots


def render_profile(rcp, dir_path: Path) -> Path:
    """Render a sourceable profile from *rcp* into *dir_path* and return its path."""
    profile = dir_path / "picongpu.profile"
    generate_bare_profile(path=profile, rc_params=rcp)
    return profile


def _run_bash(snippet: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-lc", snippet], capture_output=True, text=True, env=env)


def check_dependencies(rcp=None, script=None) -> list[tuple[str, str, bool]]:
    """Return [(root, resolved_path, present), ...] for each dependency the script owns.

    Renders + sources a profile, then tests each guard ``*_ROOT`` directory.
    *rcp* defaults to the discovered module-level ``picongpu.rc_params``;
    *script* overrides the preset's installer (see :func:`resolve_preset_script`).
    """
    rcp = _rc_params_default if rcp is None else rcp
    _, script = resolve_preset_script(rcp, script)
    roots = parse_guard_roots(script)
    with tempfile.TemporaryDirectory() as td:
        profile = render_profile(rcp, Path(td))
        # For each root: report its resolved value and whether it is an existing dir.
        # The root name is embedded literally ($r would be unset in the shell).
        body = "\n".join(
            f'v=${{{r}}} ; if [ -n "$v" ] && [ -d "$v" ]; then echo "OK {r} $v"; else echo "MISSING {r} $v"; fi'
            for r in roots
        )
        snippet = f'source "{profile}"\n{body}'
        env = dict(os.environ)
        env["PIC_PROFILE"] = str(profile)
        proc = _run_bash(snippet, env)

    results: list[tuple[str, str, bool]] = []
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in ("OK", "MISSING"):
            present = parts[0] == "OK"
            root = parts[1]
            resolved = parts[2] if len(parts) > 2 else ""
            results.append((root, resolved, present))
    # Preserve the guard order (and surface any roots bash did not echo).
    order = {r: i for i, r in enumerate(roots)}
    results.sort(key=lambda t: order.get(t[0], 1 << 30))
    return results


def _print_check(results: list[tuple[str, str, bool]]) -> int:
    missing = [r for r, _, ok in results if not ok]
    print("Dependency status:")
    for root, resolved, ok in results:
        marker = "ok" if ok else "MISSING"
        where = resolved if resolved else "(not set)"
        print(f"  [{marker:7}] {root} = {where}")
    if missing:
        print(f"\n{len(missing)} of {len(results)} dependencies are missing: {', '.join(missing)}")
        print("Run 'pic-deps install' to (re)build them, then re-check.")
        return 1
    print(f"\nAll {len(results)} dependencies are present.")
    return 0


def check(script=None) -> int:
    return _print_check(check_dependencies(script=script))


def install_dependencies(rcp=None, script=None) -> int:
    """Best-effort: run the autoinstall script, then report the final state."""
    rcp = _rc_params_default if rcp is None else rcp
    _, script = resolve_preset_script(rcp, script)
    with tempfile.TemporaryDirectory() as td:
        profile = render_profile(rcp, Path(td))
        env = dict(os.environ)
        env["PIC_PROFILE"] = str(profile)
        print(f"Installing dependencies using '{script}' (best-effort) ...")
        # Invoke through bash so a user-supplied script does not need the
        # executable bit (preset scripts are executable, copied ones may not be).
        proc = subprocess.run(["bash", "-lc", f"bash {shlex.quote(str(script))}"], env=env)
        rc = proc.returncode

    if rc != 0:
        print(f"\nwarning: '{script.name}' exited with code {rc} (best-effort; continuing).")
        print("The module/toolchain environment may be incomplete on this machine.")
    print("\nFinal dependency status:")
    return _print_check(check_dependencies(rcp, script))


def install(script=None) -> int:
    return install_dependencies(script=script)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="pic-deps",
        description=_DESC,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    script_help = (
        "use this dependency installer instead of the preset's dependencies_autoinstall*.sh "
        f"(same as the `{CUSTOM_SCRIPT_KEY}` rc parameter)"
    )
    for name, help_text in (
        ("install", "run the preset's dependencies_autoinstall.sh (best-effort)"),
        ("check", "verify each dependency directory the script owns"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--script", metavar="PATH", default=None, help=script_help)
    args = parser.parse_args(argv)
    if args.command == "install":
        return install(script=args.script)
    return check(script=args.script)


if __name__ == "__main__":
    sys.exit(main())
