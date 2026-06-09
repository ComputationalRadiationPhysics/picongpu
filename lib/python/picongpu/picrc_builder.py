"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import sys
from pathlib import Path

import questionary
import tomli_w

from moosetash import MissingVariable

from picongpu._rc_params import RCParams


def _gather_missing(p):
    while True:
        try:
            _ = p.profile_content
            break
        except MissingVariable as e:
            var = e.__cause__.args[0] if e.__cause__ else e.args[0]
            questionary.print(f'Found missing variable "{var}". Please provide a value:')
            p[var] = questionary.text(f'{var} = ').ask()


def _toml_serialize(value):
    """Serialize a Python value as a TOML scalar string."""
    s = tomli_w.dumps({"_": value})
    return s.split("=", 1)[1].strip()


def _display_toml(output):
    for key, value in output.items():
        questionary.print(f'{key} = {_toml_serialize(value)}')


def main():
    path = None
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if not path.is_file():
            print(f"Error: {path} does not exist or is not a file.", file=sys.stderr)
            sys.exit(1)

    questionary.print(
        "Welcome to picrc-builder!\n"
        "This tool helps you create or complete a .picongpurc.toml configuration "
        "file for your PIConGPU simulation setup.\n"
        "You will be asked to fill in any missing values required by your chosen preset.\n"
    )

    if path is not None:
        questionary.print(f"Loading existing configuration from {path}:")
        p = RCParams(picongpurc_path=path)
    else:
        questionary.print("First, let's choose a preset.")
        preset = questionary.text("preset = ").ask()
        questionary.print(f"Using preset '{preset}'.")
        p = RCParams(preset=preset)

    questionary.print("\nGathering missing information:")
    _gather_missing(p)

    questionary.print("\nAll done collecting values. Here is what will be written:")

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
        "pic_src_path",
        "pic_backend",
        "tbg_submit",
        "tbg_tpl_file",
        "tbg_partition",
    }
    output = {}
    for key in ("preset",):
        if key not in internal_keys and data.get(key) is not None:
            output[key] = data[key]
    for key in sorted(k for k in data if k not in internal_keys and data[k] is not None and k != "preset"):
        output[key] = data[key]

    questionary.print("")
    _display_toml(output)
    questionary.print("")
    questionary.print("Do you want to write this configuration?")

    if path is not None:
        choice = questionary.select(
            "Please choose",
            choices=[
                "Don't write.",
                f"Yes, to {path}.",
                "Yes, but ask for a new path.",
            ],
        ).ask()
        choice = choice if choice is not None else "Don't write."
    else:
        choice = questionary.select(
            "Please choose",
            choices=[
                "Don't write.",
                "Yes, but ask for a path.",
            ],
        ).ask()
        choice = choice if choice is not None else "Don't write."

    if choice == f"Yes, to {path}.":
        with path.open("wb") as f:
            tomli_w.dump(output, f)
        questionary.print(f"Written to {path}.")
        questionary.print("You can start your simulation now.")
    elif choice in ("Yes, but ask for a path.", "Yes, but ask for a new path."):
        new_path = Path(questionary.path("Path to write: ").ask())
        if new_path.exists():
            if not questionary.confirm(
                f"{new_path} already exists. Overwrite?",
                default=False,
            ).ask():
                questionary.print("Aborted. Nothing was written.")
                sys.exit(0)
        with new_path.open("wb") as f:
            tomli_w.dump(output, f)
        questionary.print(f"Written to {new_path}.")
        questionary.print("You can start your simulation now.")
    else:
        questionary.print("Aborted. Nothing was written.")


if __name__ == "__main__":
    main()
