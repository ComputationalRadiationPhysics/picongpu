"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import sys
from pathlib import Path

import tomli_w

from moosetash import MissingVariable

from picongpu._rc_params import RCParams


def _prompt(prompt_text):
    return input(prompt_text).strip()


def _gather_missing(p):
    while True:
        try:
            _ = p.profile_content
            break
        except MissingVariable as e:
            var = e.__cause__.args[0] if e.__cause__ else e.args[0]
            print(f'Found missing variable "{var}". Please provide a value:')
            p[var] = _prompt(f'{var} = ')


def _toml_serialize(value):
    """Serialize a Python value as a TOML scalar string."""
    s = tomli_w.dumps({"_": value})
    return s.split("=", 1)[1].strip()


def _display_toml(output):
    for key, value in output.items():
        print(f'{key} = {_toml_serialize(value)}')


def main():
    path = None
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if not path.is_file():
            print(f"Error: {path} does not exist or is not a file.", file=sys.stderr)
            sys.exit(1)

    if path is not None:
        p = RCParams(picongpurc_path=path)
    else:
        preset = _prompt("preset = ")
        p = RCParams(preset=preset)

    _gather_missing(p)

    data = p.model_dump()
    internal_keys = {
        "picongpurc_path",
        "preset_dir",
        "profile_template_content",
        "dirty_reset_policy",
        "missing_variable_policy",
        "required_information",
        "module_section",
        "spack_section",
        "pic_backend",
        "tbg_submit",
        "tbg_tpl_file",
        "tbg_partition",
    }
    # Build ordered output: preset first, then the rest sorted
    output = {}
    for key in ("preset",):
        if key not in internal_keys and data.get(key) is not None:
            output[key] = data[key]
    for key in sorted(k for k in data if k not in internal_keys and data[k] is not None and k != "preset"):
        output[key] = data[key]

    print("\nInformation is complete. Here is the full file:\n")
    _display_toml(output)

    print("\nDo you want to write this? If so, where?")
    if path is not None:
        print("(1) No.")
        print(f"(2) Yes, to {path}.")
        print("(3) Yes, but ask for a new path.")
        choice = _prompt("Please choose [1]")
    else:
        print("(1) No.")
        print("(2) Yes, but ask for a path.")
        choice = _prompt("Please choose [1]")

    if choice == "2" and path is not None:
        with path.open("wb") as f:
            tomli_w.dump(output, f)
        print(f"Written to {path}")
    elif choice in ("2", "3"):
        new_path = _prompt("Path to write: ")
        out = Path(new_path)
        with out.open("wb") as f:
            tomli_w.dump(output, f)
        print(f"Written to {out}")
    else:
        print("Aborted.")


if __name__ == "__main__":
    main()
