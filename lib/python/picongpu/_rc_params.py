"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import tomllib
from contextlib import contextmanager
from copy import deepcopy
from itertools import chain
from operator import methodcaller
from os import environ
from pathlib import Path
from warnings import warn

from moosetash import MissingVariable, missing_partial_default, missing_variable_keep, missing_variable_raise, render
from pydantic import BaseModel, ConfigDict

from picongpu import core


class DirtyResetError(Exception):
    pass


def _parse_export(lines, variable):
    try:
        return (
            next(filter(lambda line: line.startswith(f"export {variable}="), lines))
            .split("=", maxsplit=1)[1]
            .split("#", maxsplit=1)[0]
            .strip()
            .strip('"')
        )
    except StopIteration:
        return None


def _drop_nones(dictionary):
    return {key: value for key, value in dictionary.items() if value is not None}


_POTENTIAL_EXPORTS = {
    "author": "MY_NAME",
    "email": "MY_MAIL",
    "pic_src_path": "PICSRC",
    "pic_backend": "PIC_BACKEND",
    "tbg_submit": "TBG_SUBMIT",
    "tbg_tpl_file": "TBG_TPLFILE",
    "tbg_partition": "TBG_partition",
    "pic_libs": "PIC_LIBS",
    "account": "account",
    "qos": "qos",
    "disco_partition": "disco_partition",
    "scratch_dir": "SCRATCH",
    "project_id": "PROJID",
    "pic_node_oversubscription_pt": "PIC_NODE_OVERSUBSCRIPTION_PT",
    "project_name": "PROJECT_NAME",
}

_KEEP_AS_DEFAULT = [
    "module_section",
    "spack_section",
    "pic_backend",
    "tbg_submit",
    "tbg_partition",
    "tbg_tpl_file",
    "account",
    "qos",
    "disco_partition",
    "pic_node_oversubscription_pt",
]


def _parse_example_content(example_content):
    lines = example_content.split("\n")
    return _drop_nones(
        {
            "module_section": "\n".join(filter(_is_module_line, lines)),
            "spack_section": "\n".join(filter(_is_spack_line, lines)),
        }
        | {key: _parse_export(lines, value) for key, value in _POTENTIAL_EXPORTS.items()}
    )


def _split_into_default_and_required(parsed_example):
    return {k: v for k, v in parsed_example.items() if k in _KEEP_AS_DEFAULT} | {
        "required_information": [k for k in parsed_example.keys() if k not in _KEEP_AS_DEFAULT]
    }


def _parse_example_into_preset(preset):
    if preset is None:
        return {}
    return _split_into_default_and_required(_parse_example_content(_read_preset(preset))) | {
        "profile_template_content": _generate_profile_template_content(preset)
    }


def _is_module_line(line):
    return line.startswith("module ")


def _is_spack_line(line):
    return line.startswith("spack ")


def _find_first_index(predicate, iterable):
    try:
        return next(filter(lambda x: predicate(x[1]), enumerate(iterable)))[0]
    except StopIteration:
        return None


def _replace_export_with_template(line, replacements):
    for key, variable in replacements.items():
        if line.startswith(f"export {variable}="):
            # This is not a mistake:
            # Escaping { in f-strings is done by {{,
            # so this is {{{<value of key>}}}.
            return f'export {variable}="{{{{{{{key}}}}}}}"'
    return line


def _replace_exports_with_templates(lines, replacements):
    return [_replace_export_with_template(line, replacements) for line in lines]


def _make_template_from_example(profile_content):
    lines = profile_content.split("\n")
    lines = _replace_exports_with_templates(lines, _POTENTIAL_EXPORTS)
    module_section_start = _find_first_index(_is_module_line, lines)
    if module_section_start is not None:
        lines = [
            *lines[:module_section_start],
            "{{{module_section}}}",
            *filter(lambda x: not _is_module_line(x), lines[module_section_start:]),
        ]
    spack_section_start = _find_first_index(_is_spack_line, lines)
    if spack_section_start is not None:
        lines = [
            *lines[:spack_section_start],
            "{{{spack_section}}}",
            *filter(lambda x: not _is_spack_line(x), lines[spack_section_start:]),
        ]
    return "\n".join(lines)


def _read_preset(preset):
    etc_path = core.path("etc") / "picongpu" / preset
    if etc_path.is_dir():
        candidates = list(etc_path.glob("*.profile.example"))
        if len(candidates) == 0:
            raise ValueError(f"{preset=} not found in {etc_path=}.")
        if len(candidates) > 1:
            raise ValueError(
                f"{preset=} is ambiguous. Please use one of the following instead: {[f'{preset}/{c.name}' for c in candidates]}."
            )
        etc_path = candidates[0]
    if not etc_path.is_file() and not preset.endswith(".profile.example"):
        etc_path = Path(str(etc_path) + ".profile.example")
    if not etc_path.is_file():
        raise ValueError(f"{preset=} not found in {etc_path=}.")
    with etc_path.open("r") as file:
        return file.read()


def _generate_profile_template_content(preset):
    if preset is None:
        return ""
    return _make_template_from_example(_read_preset(preset))


def _dirty_reset_handler_raise(trigger_key, trigger_value, previous, current):
    message = (
        f"Setting {trigger_key=} to {trigger_value=} triggered resetting rc_params "
        "while it contained non-default content.\n"
        f"{previous=}\n"
        f"{current=}\n\n"
        "You can set 'dirty_reset_policy' to 'ignore', 'warn' or a custom handler "
        "to stop this exception from being raised."
    )
    raise DirtyResetError(message)


def _dirty_reset_handler_warn(trigger_key, trigger_value, previous, current):
    message = (
        f"Setting {trigger_key=} to {trigger_value=} triggered resetting rc_params "
        "while it contained non-default content.\n"
        f"{previous=}\n"
        f"{current=}\n\n"
        "You can set 'dirty_reset_policy' to 'ignore', 'raise' or a custom handler "
        "to stop this warning from being raised."
    )
    warn(message)
    return current


def _dirty_reset_handler_ignore(*args):
    return args[3]


def _interpret_dirty_reset_handler(value):
    if value == "raise":
        return _dirty_reset_handler_raise
    if value == "warn":
        return _dirty_reset_handler_warn
    if value == "ignore":
        return _dirty_reset_handler_ignore
    return value


def _missing_variable_warn(variable, *_):
    warn(
        f"Found a missing {variable=} while rendering your profile template. It is unlikely that the code will run. Check your rc_params!"
    )
    return ""


def _interpret_missing_variable_handler(value):
    if value == "raise":
        return missing_variable_raise
    if value == "warn":
        return _missing_variable_warn
    if value == "keep":
        return missing_variable_keep
    if value == "ignore":
        return missing_partial_default
    return value


def _path_to_str(value):
    if isinstance(value, str):
        return value
    if isinstance(value, Path):
        return str(value)
    try:
        return {k: _path_to_str(v) for k, v in value.items()}
    except AttributeError:
        try:
            return type(value)(_path_to_str(v) for v in value)
        except TypeError:
            return value


_RETAINED_CONTENT = {"dirty_reset_policy": "raise", "missing_variable_policy": "raise"}
_DEFAULT_CONTENT = _RETAINED_CONTENT | {"required_information": tuple(), "pic_src_path": core.path()}


def _read_picongpurc(path):
    if path is None:
        return {}
    if not isinstance(path, Path) or not path.is_absolute():
        return _read_picongpurc(Path(path).absolute())
    with path.open("rb") as file:
        return tomllib.load(file)


_SET_VIA_PROPERTY = ["picongpurc_path", "preset"]


class _Serializer(BaseModel):
    model_config = ConfigDict(extra="allow")


class RCParams:
    def __init__(self, *args, **kwargs):
        self._init_args = args
        self._init_kwargs = kwargs
        direct_init = dict(*args, **kwargs)
        rc_config = _read_picongpurc(direct_init.get("picongpurc_path", None))
        preset = _parse_example_into_preset(direct_init.get("preset", None) or rc_config.get("preset", None))
        self._data = _DEFAULT_CONTENT | preset | rc_config | direct_init

    def model_dump(self, mode="json"):
        return _Serializer(**self._data).model_dump(mode=mode)

    @property
    def preset(self):
        return self._data.get("preset", None)

    @preset.setter
    def preset(self, value):
        new_data = (
            _DEFAULT_CONTENT
            | _parse_example_into_preset(value)
            | {key: self._data.get(key, value) for key, value in _RETAINED_CONTENT.items()}
        )

        # adding _RETAINED_CONTENT shadows potential deviations from the original state
        # which is what we want because this content is retained and not lost during reset
        if (self._data | _RETAINED_CONTENT) != (
            RCParams(*self._init_args, **self._init_kwargs)._data | _RETAINED_CONTENT
        ):
            new_data = _interpret_dirty_reset_handler(self._data.get("dirty_reset_policy", "raise"))(
                "preset", value, self._data, new_data
            )
        self._data = new_data

    @property
    def picongpurc_path(self):
        return self._data.get("picongpurc_path", None)

    @picongpurc_path.setter
    def picongpurc_path(self, value):
        new_data = RCParams(picongpurc_path=value)._data | {
            key: self._data.get(key, value) for key, value in _RETAINED_CONTENT.items()
        }
        # adding _RETAINED_CONTENT shadows potential deviations from the original state
        # which is what we want because this content is retained and not lost during reset
        if (self._data | _RETAINED_CONTENT) != (
            RCParams(*self._init_args, **self._init_kwargs)._data | _RETAINED_CONTENT
        ):
            new_data = _interpret_dirty_reset_handler(self._data.get("dirty_reset_policy", "raise"))(
                "preset", value, self._data, new_data
            )
        self._data = new_data

    @property
    def preset_dir(self):
        if "preset" in self._data:
            return self._data["preset"].split("/")[0]
        elif "export PIC_SYSTEM_TEMPLATE_PATH=" in (txt := self.profile_content):
            return txt.split('PIC_SYSTEM_TEMPLATE_PATH:-"')[1].split('"}')[0].strip().split("/picongpu/")[1]
        else:
            return ""

    def __getitem__(self, *args, **kwargs):
        return self._data.__getitem__(*args, **kwargs)

    def __contains__(self, *args, **kwargs):
        return self._data.__contains__(*args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        if args[0] in _SET_VIA_PROPERTY:
            setattr(self, args[0], args[1])
        return self._data.__setitem__(*args, **kwargs)

    def __ior__(self, other):
        if isinstance(other, RCParams):
            return self.__ior__(other._data)
        if not isinstance(other, dict):
            raise TypeError(f"Unknown operator {type(self)} |= {type(other)}.")
        other = deepcopy(other)
        for key in _SET_VIA_PROPERTY:
            if (value := other.pop(key, NotImplemented)) is not NotImplemented:
                setattr(self, key, value)
        self._data = self._data.__ior__(other)
        return self

    def __or__(self, other):
        if isinstance(other, RCParams):
            return self.__or__(other._data)
        if not isinstance(other, dict):
            raise TypeError(f"Unknown operator {type(self)} | {type(other)}.")
        return RCParams(self._data.__or__(other))

    @contextmanager
    def set_temporarily(self, /, override_existing=True, **kwargs):
        setter = self._data.__setitem__ if override_existing else self._data.setdefault
        previous = {k: self._data.get(k, NotImplemented) for k in kwargs}
        try:
            for k, v in kwargs.items():
                setter(k, v)
            yield self
        finally:
            for k, v in previous.items():
                if v is NotImplemented:
                    self._data.pop(k, NotImplemented)
                else:
                    setter(k, v)

    @property
    def rendering_context(self):
        return _path_to_str(self._data)

    @property
    def profile_template_content(self):
        if "profile_template_content" in self._data:
            return self._data["profile_template_content"]
        elif "profile_template_path" in self._data:
            with Path(self._data["profile_template_path"]).open("r") as file:
                return file.read()
        else:
            return ""

    @property
    def profile_content(self):
        if "profile_content" in self._data:
            return self._data["profile_content"]
        elif "profile_path" in self._data:
            with Path(self._data["profile_path"]).open("r") as file:
                return file.read()
        elif template := self.profile_template_content:
            try:
                return render(
                    template=template,
                    context=self.rendering_context,
                    missing_variable_handler=_interpret_missing_variable_handler(self._data["missing_variable_policy"]),
                )
            except MissingVariable as error:
                message = (
                    "Rendering your profile template encountered a missing variable. "
                    f"The following variables are expected from your preset: {self._data.get('required_information', [])}. "
                    "You can query this via rc_params['required_information']."
                )
                raise MissingVariable(message) from error
        else:
            return f'export PATH="{str(core.path("bin"))}:$PATH"'

    @property
    def shebang(self):
        if "shebang" in self._data:
            return self._data["shebang"]
        elif (shebang := self.profile_content.split("\n", maxsplit=1)[0]).startswith("#!"):
            return shebang
        else:
            return "#!/bin/bash"

    @property
    def preamble(self):
        if "preamble" in self._data:
            return self._data["preamble"]
        else:
            return f"""
set -euxo pipefail
export PATH="{str(core.path("bin"))}:$PATH"
"""


def search_for_in_parents(filename, start_path):
    if not isinstance(filename, Path):
        return search_for_in_parents(filename=Path(filename), start_path=start_path)
    if not isinstance(start_path, Path) or not start_path.is_absolute():
        return search_for_in_parents(filename=filename, start_path=Path(start_path).absolute())
    try:
        return next(chain(*map(methodcaller("glob", filename.name), [start_path, *start_path.parents])))
    except StopIteration:
        return None


def search_in_environment_variables():
    path = environ.get("PIC_RC", None)
    if path is None:
        return None
    path = Path(path)
    if path.is_file():
        return path
    if path.is_dir():
        try:
            return next(path.glob("picongpurc.toml"))
        except StopIteration:
            return None
    return None


def search_in_user_config():
    path = environ.get("XDG_CONFIG_HOME", None)
    if path is None:
        return None
    path = Path(path) / "picongpu" / "picongpurc.toml"
    if path.is_file():
        return path
    return None


_DEFAULT_PICONGPURC_PATH = None


def generate_default_rc_params():
    picongpurc_path = (
        search_for_in_parents("[.]*picongpurc.toml", Path())
        or search_in_environment_variables()
        or search_in_user_config()
        or _DEFAULT_PICONGPURC_PATH
    )
    result = RCParams()
    if picongpurc_path is not None:
        result["picongpurc_path"] = picongpurc_path
    return result


rc_params = generate_default_rc_params()
