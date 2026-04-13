"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from pathlib import Path

from picongpu.presets.statements import (
    AssignmentStatement,
    LiteralCodeBlock,
    ModuleStatement,
    SourceStatement,
    make_statement,
)
from pytest import mark

VALID_STATEMENTS = {
    "module purge": ModuleStatement(engine="module", cmd="purge"),
    "module load gcc/14.1": ModuleStatement(engine="module", cmd="load", package="gcc/14.1"),
    "spack unload gcc/14.1": ModuleStatement(engine="spack", cmd="unload", package="gcc/14.1"),
    'VARIABLE="value"': AssignmentStatement(variable="VARIABLE", value="value"),
    'VARIABLE="value:$VARIABLE"': AssignmentStatement(variable="VARIABLE", value="value", mode="prepend"),
    'VARIABLE="$VARIABLE:value"': AssignmentStatement(variable="VARIABLE", value="value", mode="append"),
    'export VARIABLE="value"': AssignmentStatement(variable="VARIABLE", value="value", export=True),
    "source /path/to/file": SourceStatement(path=Path("/path/to/file")),
    "literal code block": LiteralCodeBlock(code="literal code block"),
}


@mark.parametrize(["bash_representation", "python_representation"], VALID_STATEMENTS.items())
def test_to_bash(bash_representation, python_representation):
    assert python_representation.to_bash().strip() == bash_representation


@mark.parametrize(["bash_representation", "python_representation"], VALID_STATEMENTS.items())
def test_from_bash(bash_representation, python_representation):
    assert python_representation == make_statement(bash_representation)


@mark.parametrize("python_representation", VALID_STATEMENTS.values())
def test_roundtrip_to_bash(python_representation):
    assert make_statement(python_representation.to_bash()) == python_representation


@mark.parametrize("bash_representation", VALID_STATEMENTS.keys())
def test_roundtrip_from_bash(bash_representation):
    assert make_statement(bash_representation).to_bash().strip() == bash_representation


def test_blank_line():
    assert not make_statement("")


def test_pure_comment():
    assert not make_statement("# this is a comment")


@mark.parametrize(["bash_representation", "python_representation"], VALID_STATEMENTS.items())
def test_with_weird_spacing_and_comment(bash_representation, python_representation):
    if not isinstance(python_representation, LiteralCodeBlock):
        bash_representation = bash_representation.replace(" ", " \t  ")
    assert python_representation == make_statement(f"  {bash_representation} # this is a comment  ")
