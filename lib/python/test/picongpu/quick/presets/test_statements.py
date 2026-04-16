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
    parse_line,
    parse_script,
    generate_script,
)
from pytest import mark

VALID_STATEMENTS = {
    "module purge ": ModuleStatement(engine="module", cmd="purge"),
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
def test_statement_to_str(bash_representation, python_representation):
    assert str(python_representation) == bash_representation


@mark.parametrize(["bash_representation", "python_representation"], VALID_STATEMENTS.items())
def test_from_line(bash_representation, python_representation):
    assert python_representation == parse_line(bash_representation)


@mark.parametrize("python_representation", VALID_STATEMENTS.values())
def test_roundtrip_to_bash(python_representation):
    assert parse_line(str(python_representation)) == python_representation


@mark.parametrize("bash_representation", VALID_STATEMENTS.keys())
def test_roundtrip_from_line(bash_representation):
    assert str(parse_line(bash_representation)) == bash_representation


def test_blank_line():
    assert not parse_line("")


def test_pure_comment():
    assert not parse_line("# this is a comment")


@mark.parametrize(["bash_representation", "python_representation"], VALID_STATEMENTS.items())
def test_with_weird_spacing_and_comment(bash_representation, python_representation):
    if not isinstance(python_representation, LiteralCodeBlock):
        bash_representation = bash_representation.replace(" ", " \t  ")
    assert python_representation == parse_line(f"  {bash_representation} # this is a comment  ")


def test_parse_script():
    script = "\n".join(VALID_STATEMENTS.keys())
    statements = list(VALID_STATEMENTS.values())
    assert parse_script(script) == statements


def test_generate_script():
    assert generate_script(VALID_STATEMENTS.values()) == "\n".join(VALID_STATEMENTS.keys())


def test_roundtrip_parse_script():
    script = "\n".join(VALID_STATEMENTS.keys())
    assert generate_script(parse_script(script)) == script


def test_roundtrip_generate_script():
    statements = list(VALID_STATEMENTS.values())
    assert parse_script(generate_script(statements)) == statements


def test_parse_script_merges_literal_code_blocks():
    script = """
first literal code block
module purge
second literal code block starts here

and ends here
"""
    assert parse_script(script) == [
        LiteralCodeBlock(code="first literal code block"),
        ModuleStatement(engine="module", cmd="purge"),
        LiteralCodeBlock(code="second literal code block starts here\nand ends here"),
    ]
