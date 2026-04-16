"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from itertools import chain, groupby
from operator import attrgetter
from pathlib import Path
from re import Pattern, compile
from typing import Iterable, Literal

from pydantic import BaseModel, PrivateAttr


def from_line(cls, statement, regex):
    return cls(**regex.match(statement).groupdict())


class Statement(BaseModel):
    _format_pattern: str = PrivateAttr()
    _regex: Pattern = PrivateAttr()

    def to_bash(self) -> str:
        return self._format_pattern.format(**self.model_dump(mode="json"))

    def __str__(self) -> str:
        return self.to_bash()

    @classmethod
    def from_line(cls, statement):
        return from_line(cls, statement, cls._regex.default)


class ModuleStatement(Statement):
    _format_pattern: str = "{engine} {cmd} {package}"
    _regex: Pattern = compile(r"(?P<engine>module|spack)\s+(?P<cmd>purge|load|unload)\s*(?P<package>.*)")
    engine: str
    cmd: str
    package: str = ""


class SourceStatement(Statement):
    _format_pattern: str = "source {path}"
    _regex: Pattern = compile(r"source\s+(?P<path>.*)")
    path: Path


class LiteralCodeBlock(Statement):
    _format_pattern: str = "{code}"
    _regex: Pattern = compile("(?P<code>.*)")
    code: str


def _parse_export(variable):
    result = compile(r"(?P<export>export\s|)\s*(?P<variable>.*)").match(variable).groupdict()
    return result["export"] != "", result["variable"]


def _parse_mode(value, variable):
    match value.strip('"').split(":"):
        case [var, value] if var == f"${variable}":
            return "append", value
        case [value, var] if var == f"${variable}":
            return "prepend", value
        case [value]:
            return "literal", value
        case _:
            raise ValueError(f"Unrecognised assignement after = sign: {value=}.")


class AssignmentStatement(Statement):
    _format_pattern: str = '{variable}="{value}"'
    _regex: Pattern = compile(r"(?P<variable>.*)=(?P<value>.*)")
    variable: str
    value: str
    mode: Literal["literal", "prepend", "append"] = "literal"
    export: bool = False

    def to_bash(self) -> str:
        if self.mode == "literal" and not self.export:
            return super().to_bash()
        value = self.value
        if self.mode == "append":
            value = f"${self.variable}:{self.value}"
        elif self.mode == "prepend":
            value = f"{self.value}:${self.variable}"
        variable = f"export {self.variable}" if self.export else self.variable
        return AssignmentStatement(variable=variable, value=value, mode="literal", export=False).to_bash()

    @classmethod
    def from_line(cls, statement):
        tmp = from_line(cls, statement, cls._regex.default)
        export, variable = _parse_export(tmp.variable)
        mode, value = _parse_mode(tmp.value, variable)
        return AssignmentStatement(export=export, variable=variable, mode=mode, value=value)


AnyStatement = AssignmentStatement | SourceStatement | LiteralCodeBlock | ModuleStatement


def parse_line(statement: str) -> AnyStatement | None:
    statement = statement.split("#", maxsplit=1)[0].strip()
    if statement == "":
        return None
    errors = {}
    for StatementType in [AssignmentStatement, ModuleStatement, SourceStatement, LiteralCodeBlock]:
        try:
            return StatementType.from_line(statement)
        except Exception as error:
            errors[StatementType] = error
    # Not sure why but for some reason, ruff (during pre-commit) doesn't pick up that we're on python>=3.11 already.
    # It complains that ExceptionGroup is not known.
    raise ExceptionGroup(f"Couldn't parse {statement=}.", errors)  # noqa: F821


def _merge_literal_code_blocks(statements: Iterable[AnyStatement]) -> Iterable[AnyStatement]:
    return chain(
        *(
            [label(code="\n".join(map(attrgetter("code"), list(group))))]
            if issubclass(label, LiteralCodeBlock)
            else list(group)
            for label, group in groupby(statements, type)
        )
    )


def parse_script(script: str) -> list[AnyStatement]:
    return list(_merge_literal_code_blocks(filter(lambda s: s is not None, map(parse_line, script.split("\n")))))


def generate_script(statements: Iterable[AnyStatement]) -> str:
    return "\n".join(map(str, statements))
