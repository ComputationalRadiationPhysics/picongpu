# /// script
# requires-python = ">=3.11"
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
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from picongpu import rc_params as _rc_params_default
from picongpu._rc_params import PRESET_STORAGE_PATH
from picongpu.pypicongpu.runner import generate_bare_profile

__all__ = [
    "main",
    "resolve_preset_script",
    "parse_guard_roots",
    "render_profile",
    "check_dependencies",
    "install_dependencies",
]

_DESC = (
    "pic-deps -- best-effort dependency auto-installation for a PIConGPU preset\n"
    "\n"
    "Installs (or checks) the compile-time dependencies of the preset named in"
    " the current .picongpurc.toml by running the preset's own"
    " dependencies_autoinstall.sh against a rendered profile.\n"
    "\n"
    "Subcommands:\n"
    "  install   run the preset's dependencies_autoinstall.sh (best-effort)\n"
    "  check     verify each dependency directory the script owns\n"
)

# Matches the idempotency guard `if [ ! -d "$X_ROOT" ];` (and tolerates the
# historical buggy form that omitted the `$`, e.g. `if [ ! -d "X_ROOT" ];`).
_GUARD_RE = re.compile(r'! -d\s+"?\$?([A-Za-z0-9_]+_ROOT)"?\s*\]')


def resolve_preset_script(rcp):
    """Fail-fast validation: return (preset_dir, script_path) or raise SystemExit.

    Errors when (a) no preset is set in ``rc_params``, or (b) the resolved
    preset has no ``dependencies_autoinstall.sh``.
    """
    preset = rcp.preset
    if not preset:
        raise SystemExit(
            "error: no preset is set in rc_params.\n"
            "       Run 'picrc-builder' (or set `preset` in your .picongpurc.toml) first."
        )
    preset_dir = rcp.preset_dir
    if not preset_dir:
        raise SystemExit(f"error: could not resolve the preset directory from preset '{preset}'.")
    script = PRESET_STORAGE_PATH / preset_dir / "dependencies_autoinstall.sh"
    if not script.is_file():
        raise SystemExit(
            f"error: preset '{preset_dir}' has no dependencies_autoinstall.sh "
            f"(expected {script}).\n"
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


def check_dependencies(rcp=None) -> list[tuple[str, str, bool]]:
    """Return [(root, resolved_path, present), ...] for each dependency the preset owns.

    Renders + sources a profile, then tests each guard ``*_ROOT`` directory.
    *rcp* defaults to the discovered module-level ``picongpu.rc_params``.
    """
    rcp = _rc_params_default if rcp is None else rcp
    _, script = resolve_preset_script(rcp)
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


def check(argv: list[str] | None = None) -> int:
    return _print_check(check_dependencies())


def install_dependencies(rcp=None) -> int:
    """Best-effort: run the preset's autoinstall script, then report final state."""
    rcp = _rc_params_default if rcp is None else rcp
    preset_dir, script = resolve_preset_script(rcp)
    with tempfile.TemporaryDirectory() as td:
        profile = render_profile(rcp, Path(td))
        env = dict(os.environ)
        env["PIC_PROFILE"] = str(profile)
        print(f"Installing dependencies for preset '{preset_dir}' (best-effort) ...")
        proc = subprocess.run(["bash", "-lc", str(script)], env=env)
        rc = proc.returncode

    if rc != 0:
        print(f"\nwarning: dependencies_autoinstall.sh exited with code {rc} (best-effort; continuing).")
        print("The preset's module/toolchain environment may be incomplete on this machine.")
    print("\nFinal dependency status:")
    return _print_check(check_dependencies(rcp))


def install(argv: list[str] | None = None) -> int:
    return install_dependencies()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="pic-deps",
        description=_DESC,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("install", help="run the preset's dependencies_autoinstall.sh (best-effort)")
    sub.add_parser("check", help="verify each dependency directory the script owns")
    args = parser.parse_args(argv)
    if args.command == "install":
        return install()
    return check()


if __name__ == "__main__":
    sys.exit(main())
