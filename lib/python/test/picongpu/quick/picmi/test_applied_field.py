"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from pathlib import Path
from unittest import TestCase

import pytest
import sympy
from picongpu import picmi
from picongpu.pypicongpu.backgroundfield import BackgroundField
from picongpu.pypicongpu.util import UnsupportedFeatureError
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[6]
STATIC_FIELDBACKGROUND_PARAM = REPO_ROOT / "include" / "picongpu" / "param" / "fieldBackground.param"


def _get_sim():
    grid = picmi.Cartesian3DGrid(
        number_of_cells=[16, 16, 16],
        lower_bound=[0, 0, 0],
        upper_bound=[16e-6, 16e-6, 16e-6],
        lower_boundary_conditions=["open", "open", "periodic"],
        upper_boundary_conditions=["open", "open", "periodic"],
    )
    solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
    return picmi.Simulation(time_step_size=1e-14, max_steps=4, solver=solver)


def _nonblank_lines(text: str):
    return [line for line in text.splitlines() if line.strip()]


class TestConstantAppliedField(TestCase):
    def test_translation(self):
        applied_field = picmi.ConstantAppliedField(Ex=1e6, By=0.5)
        background = applied_field.get_as_pypicongpu()

        assert isinstance(background, BackgroundField)
        assert background.ex == "1000000.0"
        assert background.ey == "0"
        assert background.ez == "0"
        assert background.bx == "0"
        assert background.by == "0.5"
        assert background.bz == "0"
        assert background.user_defined_kw == []

    def test_expression_rendering(self):
        background = picmi.ConstantAppliedField(Ez=2.5).get_as_pypicongpu()
        assert background.ez == "2.5"

    def test_influence_defaults(self):
        background = picmi.ConstantAppliedField(Ex=1e6).get_as_pypicongpu()
        assert background.influences_plugins is True
        assert background.influences_dumps is True

    def test_influence_knobs_forwarded(self):
        applied_field = picmi.ConstantAppliedField(
            Ex=1e6,
            picongpu_influences_plugins=False,
            picongpu_influences_dumps=True,
        )
        background = applied_field.get_as_pypicongpu()
        assert background.influences_plugins is False
        assert background.influences_dumps is True

    def test_no_particle_pusher_influence_knob(self):
        # The pusher scope is not a meaningful knob: the background is always
        # applied around the push. The extension keyword is rejected.
        with pytest.raises(ValidationError):
            picmi.ConstantAppliedField(Ex=1e6, picongpu_influence_particle_pusher=False)


class TestAnalyticAppliedField(TestCase):
    def test_translation_renders_expression_via_pmaccprinter(self):
        applied_field = picmi.AnalyticAppliedField(Ex_expression="sin(x)*cos(t)")
        background = applied_field.get_as_pypicongpu()

        assert isinstance(background, BackgroundField)
        assert "pmacc::math::sin(x)" in background.ex
        assert "pmacc::math::cos(t)" in background.ex
        assert background.ey == "0"
        assert background.ez == "0"

    def test_user_defined_kw(self):
        applied_field = picmi.AnalyticAppliedField(Ex_expression="b0*sin(2*pi*y/wl)", b0=1e5, wl=800e-9)
        background = applied_field.get_as_pypicongpu()

        params = {p.name: p.value for p in background.user_defined_kw}
        assert params == {"b0": 1e5, "wl": 800e-9}
        # parameters are resolved inside the rendered expression
        assert "b0" in background.ex
        assert "wl" in background.ex

    def test_influence_knobs_are_not_expression_parameters(self):
        # the picongpu_* extension kwargs must be intercepted before the standard
        # base class funnels unknown kwargs into user_defined_kw
        applied_field = picmi.AnalyticAppliedField(
            Ex_expression="b0*x",
            b0=2.0,
            picongpu_influences_plugins=False,
            picongpu_influences_dumps=False,
        )
        background = applied_field.get_as_pypicongpu()
        assert [p.name for p in background.user_defined_kw] == ["b0"]
        assert background.influences_plugins is False
        assert background.influences_dumps is False

    def test_undefined_symbol_rejected(self):
        with pytest.raises(ValueError, match="wl"):
            picmi.AnalyticAppliedField(Ex_expression="wl*sin(x)")

    def test_colliding_parameter_name_rejected(self):
        applied_field = picmi.AnalyticAppliedField(Ex_expression="x/L", x=2.0, L=3.0)
        with pytest.raises(ValueError, match="collides"):
            applied_field.get_as_pypicongpu()

    def test_cpp_keyword_parameter_name_is_escaped(self):
        # the PMAccPrinter is the single source of truth for identifier spelling:
        # a keyword parameter is escaped in both the declaration and the expression
        applied_field = picmi.AnalyticAppliedField(Ex_expression="float*x", float=2.0)
        background = applied_field.get_as_pypicongpu()
        assert "float_" in background.ex
        assert "float*" not in background.ex
        assert [p.name for p in background.user_defined_kw] == ["float_"]

    def test_cxx20_keyword_parameter_name_is_escaped(self):
        # C++20 keywords beyond sympy's built-in C++17 set (requires, concept, ...)
        # must be escaped too, since PMacc compiles with C++20
        for keyword in ("requires", "concept", "co_await", "char8_t", "consteval"):
            with self.subTest(keyword=keyword):
                applied_field = picmi.AnalyticAppliedField(**{"Ex_expression": f"{keyword}*x", keyword: 2.0})
                background = applied_field.get_as_pypicongpu()
                assert f"{keyword}_" in background.ex
                assert f"{keyword}*" not in background.ex

    def test_lower_upper_bound_none_accepted(self):
        # the whole-domain case is the default (all-None bounds)
        applied_field = picmi.AnalyticAppliedField(Ex_expression="x")
        background = applied_field.get_as_pypicongpu()
        assert background.ex == "x"


class TestAnalyticAppliedFieldFunctionInterface(TestCase):
    """The AnalyticDistribution-equivalent ``*_function`` spelling for all six components."""

    CASES = [
        ("Ex", lambda x, y, z, t: sympy.sin(x)),
        ("Ey", lambda x, y, z, t: sympy.cos(y)),
        ("Ez", lambda x, y, z, t: x + y + z),
        ("Bx", lambda x, y, z, t: sympy.exp(-t)),
        ("By", lambda x, y, z, t: sympy.Abs(z)),
        ("Bz", lambda x, y, z, t: 2.0 * t),
    ]

    def test_all_components_accept_functions(self):
        for component, function in self.CASES:
            with self.subTest(component=component):
                applied_field = picmi.AnalyticAppliedField(**{f"{component}_function": function})
                background = applied_field.get_as_pypicongpu()
                assert getattr(background, component.lower()) != "0"

    def test_function_and_expression_are_equivalent(self):
        for component, function in self.CASES:
            with self.subTest(component=component):
                x, y, z, t = sympy.symbols("x y z t")
                via_function = picmi.AnalyticAppliedField(**{f"{component}_function": function}).get_as_pypicongpu()
                expected = picmi.AnalyticAppliedField(
                    **{f"{component}_expression": str(function(x, y, z, t))}
                ).get_as_pypicongpu()
                assert getattr(via_function, component.lower()) == getattr(expected, component.lower())

    def test_function_extra_kwargs_become_parameters(self):
        applied_field = picmi.AnalyticAppliedField(
            Ex_function=lambda x, y, z, t, E0, wl: E0 * sympy.sin(2 * sympy.pi * y / wl),
            E0=1.0e5,
            wl=800e-9,
        )
        background = applied_field.get_as_pypicongpu()
        params = {p.name: p.value for p in background.user_defined_kw}
        assert params == {"E0": 1.0e5, "wl": 800e-9}
        assert "E0" in background.ex
        assert "wl" in background.ex

    def test_expression_and_function_for_same_component_must_agree(self):
        # both spellings for one component are accepted as long as they agree
        agreed = picmi.AnalyticAppliedField(Ex_expression="1.0", Ex_function=lambda x, y, z, t: sympy.Integer(1))
        assert agreed.Ex_sympy == sympy.Float(1)
        # a genuine disagreement is rejected
        with pytest.raises(ValueError, match="disagree"):
            picmi.AnalyticAppliedField(Ex_expression="1.0", Ex_function=lambda x, y, z, t: sympy.Integer(2))

    def test_component_triple_is_consistent(self):
        # every spelling is backed by the same _FieldFunctor and translates
        x, y, z, t = sympy.symbols("x y z t")
        applied_field = picmi.AnalyticAppliedField(Ex_function=lambda x, y, z, t: sympy.sin(x) + t)
        assert applied_field.Ex_sympy == sympy.sin(x) + t
        assert sympy.sympify(applied_field.Ex_expression) == sympy.sin(x) + t
        assert applied_field.Ex_function(x, y, z, t) == sympy.sin(x) + t
        assert applied_field.Ey_sympy is None
        assert applied_field.Ey_expression is None

    def test_all_three_spellings_are_owned_by_the_subclass(self):
        # the standard base class already declares ``*_expression``; the PIConGPU
        # subclass must declare the full triple itself so that every spelling is
        # exposed on this class and backed by the shared _FieldFunctor
        annotations = picmi.AnalyticAppliedField.__annotations__
        for component in ("Ex", "Ey", "Ez", "Bx", "By", "Bz"):
            for spelling in ("expression", "function", "sympy"):
                assert f"{component}_{spelling}" in annotations, f"{component}_{spelling} not declared"
                assert f"{component}_{spelling}" in picmi.AnalyticAppliedField.model_fields

    def test_expression_spelling_supplied_as_sympy_expression(self):
        x, y, z, t = sympy.symbols("x y z t")
        applied_field = picmi.AnalyticAppliedField(Ex_sympy=sympy.sin(x) + t)
        assert applied_field.Ex_sympy == sympy.sin(x) + t
        assert sympy.sympify(applied_field.Ex_expression) == sympy.sin(x) + t
        assert applied_field.Ex_function(x, y, z, t) == sympy.sin(x) + t

    def test_expression_spelling_supplied_as_number(self):
        applied_field = picmi.AnalyticAppliedField(Ex_expression=3.5)
        assert applied_field.Ex_sympy == sympy.Float(3.5)
        # the canonical string is lossless: exactly-representable floats render
        # in their shortest round-tripping form
        assert applied_field.Ex_expression == "3.5"
        assert sympy.sympify(applied_field.Ex_expression) == sympy.Float(3.5)

    def test_call_operator_evaluates_all_components(self):
        applied_field = picmi.AnalyticAppliedField(
            Ex_expression="E0*x", Ey_function=lambda x, y, z, t: sympy.sin(t), E0=2.0
        )
        values = applied_field(3.0, 0.0, 0.0, 0.0)
        assert set(values) == {"Ex", "Ey", "Ez", "Bx", "By", "Bz"}
        assert values["Ex"] == 6.0
        assert values["Ey"] == 0.0
        assert values["Ez"] is None

    def test_call_operator_broadcasts_over_arrays(self):
        import numpy as np

        applied_field = picmi.AnalyticAppliedField(Ex_expression="2*x")
        values = applied_field(np.array([0.0, 1.0, 2.0]), 0.0, 0.0, 0.0)
        np.testing.assert_allclose(values["Ex"], [0.0, 2.0, 4.0])

    def test_constant_call_operator_is_broadcastable(self):
        import numpy as np

        applied_field = picmi.ConstantAppliedField(Ex=1.0e6, Bz=0.5)
        x = np.zeros((2, 3))
        values = applied_field(x, x, x, 0.0)
        assert values["Ex"] == 1.0e6
        np.testing.assert_allclose(values["Bz"], 0.5)
        assert values["Ey"] is None

    def test_function_undefined_symbol_rejected(self):
        unknown = sympy.Symbol("unknown")
        with pytest.raises(ValueError, match="unknown"):
            picmi.AnalyticAppliedField(Ex_function=lambda x, y, z, t: unknown * x)

    def test_expression_only_unreferenced_kwarg_still_rejected(self):
        # the standard collector stays in charge for pure-expression inputs
        with pytest.raises(Exception, match="bogus"):
            picmi.AnalyticAppliedField(Ex_expression="x", bogus=3.0)

    def test_function_only_unreferenced_kwarg_still_rejected(self):
        # a function input must use the #97 mechanism: only kwargs named like the
        # callable's extra parameters are collected, unknown ones are rejected
        with pytest.raises(Exception, match="bogus"):
            picmi.AnalyticAppliedField(Ex_function=lambda x, y, z, t: sympy.Integer(5), bogus=3.0)

    def test_type_error_inside_callable_is_not_masked(self):
        # a genuine error raised *inside* the user callable must propagate, not
        # be swallowed and retried positionally by the fallback
        def broken(x, y, z, t):
            raise TypeError("inside user callable")

        with pytest.raises(TypeError, match="inside user callable"):
            picmi.AnalyticAppliedField(Ex_function=broken).get_as_pypicongpu()

    def test_mixed_expression_and_function_parameters(self):
        applied_field = picmi.AnalyticAppliedField(
            Ex_expression="q*x",
            Ey_function=lambda x, y, z, t, r: r * y,
            q=1.0,
            r=2.0,
        )
        background = applied_field.get_as_pypicongpu()
        params = {p.name: p.value for p in background.user_defined_kw}
        assert params == {"q": 1.0, "r": 2.0}
        assert "q" in background.ex
        assert "r" in background.ey

    def test_unrelated_parameter_is_not_forced_into_function(self):
        # a function that does not use an expression's parameter must still work
        applied_field = picmi.AnalyticAppliedField(
            Ex_expression="q*x",
            Ey_function=lambda x, y, z, t: sympy.Integer(5),
            q=1.0,
        )
        background = applied_field.get_as_pypicongpu()
        assert background.ey == "5"


class TestBackgroundFieldRoundTrip(TestCase):
    def test_json_roundtrip_idempotent(self):
        background = picmi.AnalyticAppliedField(Ex_expression="sin(x)*cos(t)").get_as_pypicongpu()
        restored = BackgroundField.model_validate_json(background.model_dump_json())
        assert restored.ex == background.ex
        assert restored.bz == "0"
        assert restored.user_defined_kw == background.user_defined_kw


class TestSimulationBackgroundField(TestCase):
    def test_no_applied_field(self):
        sim = _get_sim()
        assert sim.get_as_pypicongpu().background_field is None

    def test_single_constant_applied_field(self):
        sim = _get_sim()
        sim.add_applied_field(picmi.ConstantAppliedField(Ez=3e6))
        background = sim.get_as_pypicongpu().background_field
        assert isinstance(background, BackgroundField)
        assert background.ez == "3000000.0"

    def test_single_analytic_applied_field(self):
        sim = _get_sim()
        sim.add_applied_field(picmi.AnalyticAppliedField(Bx_expression="0.1*x/L", L=1e-3))
        background = sim.get_as_pypicongpu().background_field
        assert "x/L" in background.bx
        assert "0.1" in background.bx
        assert "L" in background.bx

    def test_multiple_constant_applied_fields_are_summed(self):
        sim = _get_sim()
        sim.add_applied_field(picmi.ConstantAppliedField(Ez=1.0))
        sim.add_applied_field(picmi.ConstantAppliedField(Ez=2.0, Bz=3.0))
        background = sim.get_as_pypicongpu().background_field
        assert isinstance(background, BackgroundField)
        assert background.ez == "3.0"
        assert background.bz == "3.0"
        assert background.ex == "0"

    def test_constant_and_analytic_applied_fields_are_summed(self):
        sim = _get_sim()
        sim.add_applied_field(picmi.ConstantAppliedField(Ez=1.0, By=2.0))
        sim.add_applied_field(
            picmi.AnalyticAppliedField(Ez_expression="3.0", Bx_function=lambda x, y, z, t: sympy.sin(x))
        )
        background = sim.get_as_pypicongpu().background_field
        assert background.ez == "4.0"
        assert background.by == "2.0"
        assert background.bx == "pmacc::math::sin(x)"

    def test_combined_parameters_are_merged(self):
        sim = _get_sim()
        sim.add_applied_field(picmi.AnalyticAppliedField(Ex_expression="a*x", a=2.0))
        sim.add_applied_field(picmi.AnalyticAppliedField(Ey_expression="b*y", b=3.0))
        background = sim.get_as_pypicongpu().background_field
        params = {p.name: p.value for p in background.user_defined_kw}
        assert params == {"a": 2.0, "b": 3.0}
        assert "a*x" in background.ex
        assert "b*y" in background.ey

    def test_conflicting_parameter_values_rejected(self):
        sim = _get_sim()
        sim.add_applied_field(picmi.AnalyticAppliedField(Ex_expression="a*x", a=2.0))
        sim.add_applied_field(picmi.AnalyticAppliedField(Ey_expression="a*y", a=3.0))
        with pytest.raises(UnsupportedFeatureError):
            sim.get_as_pypicongpu()

    def test_undefined_symbol_rejected_through_simulation(self):
        # the same expression validation runs at construction, so an undefined
        # symbol can never reach the combine path and emit invalid C++
        sim = _get_sim()
        with pytest.raises(ValueError, match="wl"):
            sim.add_applied_field(picmi.AnalyticAppliedField(Ex_expression="wl*sin(x)"))

    def test_function_undefined_symbol_rejected_through_simulation(self):
        sim = _get_sim()
        with pytest.raises(ValueError, match="typo"):
            sim.add_applied_field(picmi.AnalyticAppliedField(Ex_function=lambda x, y, z, t: sympy.Symbol("typo") * x))

    def test_keyword_parameter_name_escaped_through_simulation(self):
        sim = _get_sim()
        sim.add_applied_field(picmi.AnalyticAppliedField(Ex_expression="x/float", float=2.0))
        background = sim.get_as_pypicongpu().background_field
        assert "float_" in background.ex
        assert "x/float*" not in background.ex

    def test_conflicting_influence_knobs_rejected(self):
        sim = _get_sim()
        sim.add_applied_field(picmi.ConstantAppliedField(Ez=1.0))
        sim.add_applied_field(picmi.ConstantAppliedField(Bz=1.0, picongpu_influences_dumps=False))
        with pytest.raises(UnsupportedFeatureError):
            sim.get_as_pypicongpu()

    def test_unsupported_applied_field_type_rejected(self):
        from picmistandard import PICMI_LoadGriddedField

        sim = _get_sim()
        # Standard PICMI applied fields that map to other C++ mechanisms
        # (injection/initialization) are not supported as background fields yet.
        sim.add_applied_field(picmi.AnalyticAppliedField(Ex_expression="1.0"))
        sim.add_applied_field(PICMI_LoadGriddedField(read_fields_from_path="/tmp/dummy.h5"))
        with pytest.raises(UnsupportedFeatureError):
            sim.get_as_pypicongpu()

    def test_region_bounds_rejected(self):
        sim = _get_sim()
        sim.add_applied_field(picmi.ConstantAppliedField(Ez=1.0, lower_bound=[0, 0, 0], upper_bound=[1e-6, 1e-6, 1e-6]))
        with pytest.raises(UnsupportedFeatureError):
            sim.get_as_pypicongpu()

    def test_render_context_is_none_without_applied_field(self):
        sim = _get_sim()
        rendered = sim.get_as_pypicongpu().get_rendering_context()
        assert "background_field" in rendered
        assert rendered["background_field"] is None

    def test_render_context_contains_background_field(self):
        sim = _get_sim()
        sim.add_applied_field(picmi.ConstantAppliedField(Ey=1e6))
        context = sim.get_as_pypicongpu().get_rendering_context()
        assert context["background_field"] is not None
        for key in (
            "ex",
            "ey",
            "ez",
            "bx",
            "by",
            "bz",
            "influences_plugins",
            "influences_dumps",
        ):
            assert key in context["background_field"]
        # the renderer only accepts the standard leaf types
        assert isinstance(context["background_field"]["ey"], str)
        assert isinstance(context["background_field"]["influences_plugins"], bool)

    def test_applied_field_from_constructor(self):
        grid = picmi.Cartesian3DGrid(
            number_of_cells=[16, 16, 16],
            lower_bound=[0, 0, 0],
            upper_bound=[16e-6, 16e-6, 16e-6],
            lower_boundary_conditions=["open", "open", "periodic"],
            upper_boundary_conditions=["open", "open", "periodic"],
        )
        solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
        sim = picmi.Simulation(
            time_step_size=1e-14,
            max_steps=4,
            solver=solver,
            applied_fields=[picmi.ConstantAppliedField(Ez=3e6)],
        )
        background = sim.get_as_pypicongpu().background_field
        assert isinstance(background, BackgroundField)
        assert background.ez == "3000000.0"

    def test_constant_lower_upper_bound_none_accepted(self):
        # the whole-domain case is the default (all-None bounds)
        applied_field = picmi.ConstantAppliedField(Ez=1.0)
        background = applied_field.get_as_pypicongpu()
        assert background.ez == "1.0"


class TestRenderedParamFunctionallyEqual(TestCase):
    """Render an input setup and compare the generated fieldBackground.param to the static one.

    These are rendering-level checks rather than pinning of exact output, so
    they stay robust against formatting changes: the generated file must be
    build-relevant-identical when no background field is configured."""

    def _render_setup(self, applied_field=None):
        import tempfile

        sim = _get_sim()
        if applied_field is not None:
            sim.add_applied_field(applied_field)
        with tempfile.TemporaryDirectory() as tmpdir:
            sim.write_input_file(Path(tmpdir) / "setup")
            param_path = Path(tmpdir) / "setup" / "include" / "picongpu" / "param" / "fieldBackground.param"
            return param_path.read_text()

    def _render_n_cfg(self, applied_field=None):
        import tempfile

        sim = _get_sim()
        if applied_field is not None:
            sim.add_applied_field(applied_field)
        with tempfile.TemporaryDirectory() as tmpdir:
            sim.write_input_file(Path(tmpdir) / "setup")
            cfg_path = Path(tmpdir) / "setup" / "etc" / "picongpu" / "N.cfg"
            return cfg_path.read_text()

    def test_default_rendering_equivalent_to_static_param(self):
        rendered = self._render_setup()
        static = STATIC_FIELDBACKGROUND_PARAM.read_text()
        assert _nonblank_lines(rendered) == _nonblank_lines(static)

    def test_default_rendering_n_cfg_has_no_field_background_option(self):
        # without a background the compatibility options are not rendered at all
        assert "fieldBackground.influences" not in self._render_n_cfg()

    def test_configured_rendering_enables_background(self):
        rendered = self._render_setup(picmi.ConstantAppliedField(Ey=1e6))
        assert "InfluenceParticlePusher = true" in rendered
        assert "1000000.0" in rendered
        # the J background stays off
        assert "FieldBackgroundJ" in rendered
        assert "activated = false" in rendered

    def test_configured_rendering_defaults_keep_plugins_and_dumps_on(self):
        rendered = self._render_setup(picmi.ConstantAppliedField(Ey=1e6))
        # the generated background is always applied around the particle push
        assert rendered.count("InfluenceParticlePusher = true") == 2
        cfg = self._render_n_cfg(picmi.ConstantAppliedField(Ey=1e6))
        assert "--fieldBackground.influencesPlugins true" in cfg
        assert "--fieldBackground.influencesDumps true" in cfg

    def test_configured_rendering_honours_visibility_knobs(self):
        applied_field = picmi.ConstantAppliedField(
            Ey=1e6,
            picongpu_influences_plugins=False,
            picongpu_influences_dumps=False,
        )
        rendered = self._render_setup(applied_field)
        # the generated background is always applied around the push
        assert rendered.count("InfluenceParticlePusher = true") == 2
        cfg = self._render_n_cfg(applied_field)
        assert "--fieldBackground.influencesPlugins false" in cfg
        assert "--fieldBackground.influencesDumps false" in cfg

    def test_configured_rendering_contains_analytic_expression(self):
        applied_field = picmi.AnalyticAppliedField(Ex_expression="1e5*sin(2*pi*y/wl)", wl=800e-9)
        rendered = self._render_setup(applied_field)
        assert "InfluenceParticlePusher = true" in rendered
        assert "pmacc::math::sin" in rendered
        # the parameter is rendered as a compile-time constant in the functors
        assert "constexpr float_64 wl =" in rendered

    def test_analytic_parameters_guarded_in_both_functors(self):
        # a parameter used only in the E expression must not trigger -Wunused-variable
        # in FieldBackgroundB (all parameter declarations carry [[maybe_unused]])
        applied_field = picmi.AnalyticAppliedField(Ex_expression="b0*cos(2*pi*y/wl)", b0=1e6, wl=800e-9)
        rendered = self._render_setup(applied_field)
        assert rendered.count("[[maybe_unused]] constexpr float_64 b0") == 2
        assert rendered.count("[[maybe_unused]] constexpr float_64 wl") == 2
