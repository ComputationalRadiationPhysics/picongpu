# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "picongpu @ git+https://github.com/ComputationalRadiationPhysics/picongpu@dev#subdirectory=lib/python"
# ]
# ///
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

from picongpu._rc_params import RCParams, get_available_presets

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


_MULTI_LINE_KEYS = {"module_section", "profile_content", "profile_template_content"}


def _offer_param_edits(p):
    """Show current parameters (excluding large multi-line content) and let the user edit any.

    Multi-line fields (module_section, profile_content, profile_template_content)
    are shown only if the user requests them via an expand option.

    Returns
    -------
    set[str]
        Keys that the user actually changed.
    """
    data = p.model_dump()
    all_entries = [(k, v) for k, v in data.items() if v is not None and k != "preset"]
    if not all_entries:
        return set()

    if not questionary.confirm("\nWant to see all set parameters to make edits?").ask():
        return set()

    short_entries = sorted((k, v) for k, v in all_entries if k not in _MULTI_LINE_KEYS)
    multi_entries = sorted((k, v) for k, v in all_entries if k in _MULTI_LINE_KEYS)

    questionary.print("\nYour configuration contains the following parameters:")
    for key, value in short_entries:
        questionary.print(f"  {key} = {_toml_serialize(value)}")

    if multi_entries:
        questionary.print(
            "  (<multi-line content hidden for module_section, profile_content, profile_template_content>)"
        )
        show = questionary.confirm("Show multi-line content?").ask()
        if show:
            for key, value in multi_entries:
                questionary.print(f"\n--- {key} ---")
                questionary.print(f"{value}\n")

    if not questionary.confirm("\nWant to change any of these parameters?").ask():
        return set()

    all_keys = [k for k, _ in all_entries]
    keys_to_edit = questionary.checkbox("Which parameters would you like to change?", choices=all_keys).ask() or []

    overridden = set()
    for key in keys_to_edit:
        if key in _MULTI_LINE_KEYS:
            questionary.print(f"\nCurrent value of {key}:")
            questionary.print(f"{p[key]}\n")
        else:
            questionary.print(f"\nCurrent value: {key} = {_toml_serialize(p[key])}")
        new_value = questionary.text(f"New value for {key}: ").ask()
        if new_value is not None:
            p[key] = new_value
            overridden.add(key)
    return overridden


def _offer_custom_params(p):
    """Allow the user to add arbitrary custom key-value pairs."""
    if not questionary.confirm("\nWant to add custom parameters?").ask():
        return

    questionary.print("Enter key names and values. Leave key empty to finish.")
    while True:
        key = questionary.text("Key name (empty to finish): ").ask()
        if not key or key == "":
            break
        value = questionary.text(f"Value for {key}: ").ask()
        if value is not None:
            p[key] = value


def _toml_serialize(value):
    """Return a TOML scalar representation of *value*."""
    s = tomli_w.dumps({"_": value})
    return s.split("=", 1)[1].strip()


def _display_toml(output):
    """Print each key-value pair in *output* as a TOML line."""
    for key, value in output.items():
        questionary.print(f"{key} = {_toml_serialize(value)}")


def _filter_user_keys(data, /, original_data):
    """Return dict of user-facing keys (preset first, then sorted).

    Internal / preset-derived keys are excluded so the preview stays clean.
    A default parameter only appears in the output if its value differs from
    the snapshot taken before the user was allowed to edit it.
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
        if data.get(key) is not None:
            output[key] = data[key]
    for key in sorted(k for k in data if k != "preset" and data[k] is not None):
        if key not in internal_keys or data.get(key) != original_data.get(key):
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

        available_presets = [
            name.removesuffix(".example").removesuffix(".profile").removesuffix("_picongpu")
            for name in get_available_presets()
        ]
        if not available_presets:
            print("Error: No presets found in etc/picongpu.")
            return

        preset = questionary.select("Select a preset:", choices=available_presets).ask()

        if preset is None:
            print("Aborted.")
            return

        questionary.print(f"Using preset '{preset}'.")
        p = RCParams(preset=preset)

    questionary.print("\nGathering missing information:")
    _gather_missing(p)
    original_data = p.model_dump()

    _offer_param_edits(p)
    _offer_custom_params(p)

    output = _filter_user_keys(p.model_dump(), original_data)

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
        default_path = Path("./.picongpurc.toml")
        new_path_str = questionary.path("Path to write:", default=str(default_path)).ask()

        if new_path_str is None:
            print("Aborted.")
            return

        new_path = Path(new_path_str)

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
