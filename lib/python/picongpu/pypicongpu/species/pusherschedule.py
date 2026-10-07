"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+

Backend model for step-dependent (composite) particle pushers.

A :class:`PusherSchedule` holds the pre-rendered C++ for one species'
``particles::pusher::...`` type.  It is built from a list of step-slices per
pusher stage (first match wins), which the PICMI layer resolves from its
``CompositePusher`` mapping.  The rendering is closed-form (a predicate per
stage), so it stays valid if the user changes ``max_steps`` afterwards.
"""

from uuid import uuid4

from pydantic import BaseModel


def _slice_membership(start: int, stop: int, step: int, variable: str = "currentStep") -> str:
    """C++ boolean expression true iff ``variable`` is in the (inclusive) slice.

    ``stop == -1`` denotes an open upper end.  ``start`` is always non-negative
    (negatives have been resolved against the step count before this point).
    """
    terms = [f"({variable} >= {start})"]
    if stop != -1:
        terms.append(f"({variable} <= {stop})")
    if step != 1:
        terms.append(f"((({variable} - {start}) % {step}) == 0)")
    return " && ".join(terms)


def _entry_predicate(slices, variable: str = "currentStep") -> str:
    """C++ boolean expression for a pusher stage covering a union of slices."""
    if not slices:
        return "false"
    return " || ".join("(" + _slice_membership(*one, variable=variable) + ")" for one in slices)


def _activation_predicate(entries, stage: int, variable: str = "currentStep") -> str:
    """Predicate for activation functor ``stage`` (0-based).

    It is true iff ``variable`` is claimed by stage ``stage`` and by no earlier
    stage: ``AND_{j < stage} NOT(spec_j) AND spec_stage``.
    """
    terms = [f"!({_entry_predicate(entries[j], variable=variable)})" for j in range(stage)]
    terms.append(f"({_entry_predicate(entries[stage], variable=variable)})")
    return " && ".join(terms)


def build_pusher_schedule(entries, pusher_cpp_names) -> "PusherSchedule":
    """Build the C++ for a step-dependent pusher.

    :param entries: list (one per stage, in first-match order) of lists of
        ``(start, stop, step)`` slices, all resolved to steps.
    :param pusher_cpp_names: C++ struct names (``particles::pusher::<name>``),
        aligned with ``entries``.  May also carry already-namespaced types when
        a nested ``Composite`` is used, but this builder always takes plain
        struct names.
    """
    if len(entries) != len(pusher_cpp_names):
        raise ValueError(f"Mismatching stage count: {len(entries)=} vs {len(pusher_cpp_names)=}.")
    if not entries:
        raise ValueError("A pusher schedule needs at least one stage.")

    qualified = [f"particles::pusher::{name}" for name in pusher_cpp_names]

    if len(entries) == 1:
        return PusherSchedule(pusher_cpp=qualified[0], activation_declaration="")

    activation_names = [f"PusherActivation_{uuid4().hex}" for _ in range(len(entries) - 1)]
    declarations = []
    for stage, name in enumerate(activation_names):
        predicate = _activation_predicate(entries, stage)
        declarations.append(
            "struct {name}\n"
            "{{\n"
            "    HDINLINE constexpr uint32_t operator()(uint32_t const currentStep) const\n"
            "    {{\n"
            "        return ({predicate}) ? 1u : 2u;\n"
            "    }}\n"
            "}};".format(name=name, predicate=predicate)
        )

    composite = f"particles::pusher::Composite<{qualified[-2]}, {qualified[-1]}, {activation_names[-1]}>"
    for stage in range(len(entries) - 3, -1, -1):
        composite = f"particles::pusher::Composite<{qualified[stage]}, {composite}, {activation_names[stage]}>"

    return PusherSchedule(pusher_cpp=composite, activation_declaration="\n\n".join(declarations))


class PusherSchedule(BaseModel):
    """Rendered C++ for a step-dependent pusher.

    ``pusher_cpp`` is the full ``particles::pusher::...`` type referenced by
    ``particles::pusher`` in ``speciesDefinition.param``.
    ``activation_declaration`` contains the standalone activation-functor
    structs it refers to (empty for a single-stage schedule, which needs none).
    """

    pusher_cpp: str
    activation_declaration: str = ""
