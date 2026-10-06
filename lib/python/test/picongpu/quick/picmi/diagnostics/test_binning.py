"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from unittest import TestCase

from picongpu import templates
from picongpu.picmi import Species
from picongpu.picmi.diagnostics import BinSpec, Binning, BinningAxis, TS
from picongpu.picmi.diagnostics.binning import BinningFunctor
from picongpu.pypicongpu.rendering import Renderer

TEMPLATE = (templates.path() / "include" / "picongpu" / "param" / "binningSetup.param.mustache").read_text()


def _binning(period=TS[::10], **kwargs):
    return Binning(
        name="electron_density",
        deposition_functor=BinningFunctor(name="weighting", functor=lambda p: p.get("weighting")),
        axes=[
            BinningAxis(
                functor=BinningFunctor(name="position0", functor=lambda p: 0.0),
                bin_spec=BinSpec(kind="linear", start=0, stop=1, nsteps=2),
            )
        ],
        species=Species(particle_type="electron"),
        period=period,
        **kwargs,
    )


def _rendered(binning, time_step_size=1.0, num_steps=100):
    converted = binning.get_as_pypicongpu(time_step_size=time_step_size, num_steps=num_steps)
    context = {"output": [converted.model_dump()]}
    return Renderer.get_rendered_template(Renderer.get_context_preprocessed(context), TEMPLATE)


class TestAccumulationPeriod(TestCase):
    def test_defaults_to_one(self):
        assert _binning().accumulation_period == 1

    def test_renders_set_dump_period(self):
        rendered = _rendered(_binning(accumulation_period=5))
        assert ".setDumpPeriod(5)" in rendered

    def test_renders_default_set_dump_period(self):
        assert ".setDumpPeriod(1)" in _rendered(_binning())

    def test_translated_to_pypicongpu(self):
        converted = _binning(accumulation_period=7).get_as_pypicongpu(time_step_size=1.0, num_steps=10)
        assert converted.accumulation_period == 7

    def test_old_name_is_gone(self):
        with self.assertRaises(ValueError):
            _binning(dumpPeriod=5)
        # the misleading name must no longer exist on the output model either
        b = _binning()
        assert not hasattr(b, "dumpPeriod")
        converted = b.get_as_pypicongpu(time_step_size=1.0, num_steps=10)
        assert not hasattr(converted, "dumpPeriod")

    def test_must_accept_integers(self):
        # the C++ argument is an integer; a float is not a valid accumulation period
        with self.assertRaises(ValueError):
            _binning(accumulation_period=2.5)


class TestBinningPeriodRendering(TestCase):
    def test_unshifted_period_renders_byte_identical(self):
        """An unshifted period still renders the same integer start:stop:step token."""
        assert '.setNotifyPeriod("0:-1:10")' in _rendered(_binning(period=TS[::10]))

    def test_shifted_period_renders_shifted_token(self):
        shifted = TS[::10]("steps") + 5 * TS.steps
        assert '.setNotifyPeriod("5:-1:10")' in _rendered(_binning(period=shifted))

    def test_seconds_period_renders_resolved_token(self):
        # dt = 1e-6 s: 1e-5 s -> step 10, every 1e-5 s -> step 10, stop rounded up
        period = TS[1e-5:2e-5:1e-5]("seconds")
        rendered = _rendered(_binning(period=period), time_step_size=1e-6, num_steps=100)
        assert '.setNotifyPeriod("10:21:10")' in rendered
