"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import tomllib
from itertools import chain
from operator import methodcaller
from pathlib import Path

from moosetash import render

from picongpu import core


class RCParams(dict):
    def _handle_setting_picongpurc_path(self, to):
        if to is None:
            return
        self.clear()
        with Path(to).open("rb") as file:
            self |= tomllib.load(file).items()

    def __setitem__(self, *args, **kwargs):
        if args[0] == "picongpurc_path":
            self._handle_setting_picongpurc_path(args[1])
        return super().__setitem__(*args, **kwargs)

    @property
    def rendering_context(self):
        return self

    @property
    def profile_template_content(self):
        if "profile_template_content" in self:
            return self["profile_template_content"]
        elif "profile_template_path" in self:
            with Path(self["profile_template_path"]).open("r") as file:
                return file.read()
        else:
            return ""

    @property
    def profile_content(self):
        if "profile_content" in self:
            return self["profile_content"]
        elif "profile_path" in self:
            with Path(self["profile_path"]).open("r") as file:
                return file.read()
        elif template := self.profile_template_content:
            return render(
                template=template, context=self.rendering_context, missing_variable_handler=lambda *_: str(1 / 0)
            )
        else:
            return ""

    @property
    def shebang(self):
        if "shebang" in self:
            return self["shebang"]
        elif (shebang := self.profile_content.split("\n", maxsplit=1)[0]).startswith("#!"):
            return shebang
        else:
            return "#!/bin/bash"

    @property
    def preamble(self):
        if "preamble" in self:
            return self["preamble"]
        else:
            return f"""
set -euxo pipefail
export PATH="{str(core.path("bin"))}:$PATH"
"""


def search_for_in_parents(filename, start_path=Path()):
    if not isinstance(filename, Path):
        return search_for_in_parents(filename=Path(filename), start_path=start_path)
    if not isinstance(start_path, Path) or not start_path.is_absolute():
        return search_for_in_parents(filename=filename, start_path=Path(start_path).absolute())
    try:
        return next(chain(*map(methodcaller("glob", filename.name), [start_path, *start_path.parents])))
    except StopIteration:
        return None


rc_params = RCParams()
