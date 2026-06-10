"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+

Interactive builder for .picongpurc.toml configuration files.

Guides the user through filling in missing variables required by a
PIConGPU preset, then writes the resulting TOML file to disk.
"""

import argparse
from pathlib import Path

import questionary
import tomli_w

from moosetash import MissingVariable

from picongpu._rc_params import RCParams

__all__ = ["main"]

_DESC = (
    "picrc-builder -- interactive .picongpurc.toml configuration builder\n"
    "\n"
    "Guides you through creating or completing a PIConGPU configuration file."
    " If a path to an existing .picongpurc.toml is given, the tool loads it"
    " and only asks for missing values. Otherwise it starts from scratch by"
    " asking for a preset first."
)


def _gather_missing(p):
    """Ask the user for every template variable the preset requires.

    Repeatedly accesses ``p.profile_content`` until no ``MissingVariable``
    is raised, prompting the user for each missing key.
    """
    while True:
        try:
            _ = p.profile_content
            break
        except MissingVariable as e:
            var = e.__cause__.args[0] if e.__cause__ else e.args[0]
            questionary.print(f'Found missing variable "{var}". Please provide a value:')
            p[var] = questionary.text(f"{var} = ").ask()


def _toml_serialize(value):
    """Return a TOML scalar representation of *value*."""
    s = tomli_w.dumps({"_": value})
    return s.split("=", 1)[1].strip()


def _display_toml(output):
    """Print each key-value pair in *output* as a TOML line."""
    for key, value in output.items():
        questionary.print(f"{key} = {_toml_serialize(value)}")


def _filter_user_keys(data):
    """Return dict of user-facing keys (preset first, then sorted).

    Internal / preset-derived keys are excluded so the preview stays clean.
    """
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
    return output


def write_output(output, path):
    """Write *output* as a TOML file to *path*."""
    with path.open("wb") as f:
        tomli_w.dump(output, f)


def main(argv=None):
    """Entry point for the picrc-builder CLI.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments (defaults to ``sys.argv[1:]``).
    """
    parser = argparse.ArgumentParser(
        prog="picrc-builder",
        description=_DESC,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        type=Path,
        help="Path to an existing .picongpurc.toml to load and complete.",
    )
    args = parser.parse_args(argv)

    path = args.config
    if path is not None and not path.is_file():
        parser.error(f"{path} does not exist or is not a file.")

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

    output = _filter_user_keys(p.model_dump())

    questionary.print("\nAll done collecting values. Here is what will be written:")
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
        write_output(output, path)
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
                return
        write_output(output, new_path)
        questionary.print(f"Written to {new_path}.")
        questionary.print("You can start your simulation now.")
    else:
        questionary.print("Aborted. Nothing was written.")


if __name__ == "__main__":
    main()
