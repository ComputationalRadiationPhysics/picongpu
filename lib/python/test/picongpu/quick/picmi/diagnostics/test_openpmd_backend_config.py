"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
License: GPLv3+
"""

import json

import openpmd_api as io
import pytest
import tomli_w

from picongpu.picmi.diagnostics.binning import Binning, BinningAxis, BinningFunctor, BinSpec
from picongpu.picmi.diagnostics.timestepspec import TimeStepSpec
from picongpu.picmi.species import Species
from picongpu.pypicongpu.output.openpmd_backend import (
    Adios2Config,
    Adios2Engine,
    Adios2Operator,
    DatasetOverride,
    Hdf5Config,
    Hdf5Dataset,
    JsonTomlConfig,
    OpenPMDBackendConfig,
)
from picongpu.pypicongpu.output.openpmd_plugin import FieldDump, OpenPMDConfig, OpenPMDPlugin
from picongpu.pypicongpu.output.timestepspec import Spec, TimeStepSpec as PyTimeStepSpec


def _adios2_with_per_dataset_overrides() -> OpenPMDBackendConfig:
    return OpenPMDBackendConfig(
        backend="adios2",
        adios2=Adios2Config(
            engine=Adios2Engine(type="sst", parameters={"Profile": "On"}),
            dataset=[
                # default entry (no select): blosc compression by default
                {"cfg": {"operators": [Adios2Operator(type="blosc", parameters={"clevel": "1"})]}},
                # select entry: no compression for matching datasets
                {"select": [".*positionOffset.*", ".*particlePatches.*"], "cfg": {"operators": []}},
            ],
        ),
        hdf5=Hdf5Config(dataset=Hdf5Dataset(chunks="auto")),
    )


def _rendered_backend_config(tmp_path, backend_config) -> dict:
    (tmp_path / "etc").mkdir()
    plugin = OpenPMDPlugin(
        sources=[
            (
                PyTimeStepSpec(specs=[Spec(start=0, stop=-1, step=1)]),
                FieldDump(name="E", functor=None, filtername=None, species_name=None),
            )
        ],
        config=OpenPMDConfig(file="simData", backend_config=backend_config),
    )
    plugin.setup_dir = tmp_path
    return plugin._generate_config_file()["backend_config"]


def test_backend_config_renders_nested_toml_table(tmp_path):
    """A populated backend_config renders as a nested [backend_config] TOML table
    (what the C++ specialConversions() consumes) including the per-dataset
    [[backend_config.adios2.dataset]] override list."""
    (tmp_path / "etc").mkdir()
    plugin = OpenPMDPlugin(
        sources=[
            (
                PyTimeStepSpec(specs=[Spec(start=0, stop=-1, step=1)]),
                FieldDump(name="E", functor=None, filtername=None, species_name=None),
            )
        ],
        config=OpenPMDConfig(file="simData", backend_config=_adios2_with_per_dataset_overrides()),
    )
    plugin.setup_dir = tmp_path
    content = plugin._generate_config_file()
    toml_text = tomli_w.dumps(content)

    assert "[backend_config]" in toml_text
    assert 'backend = "adios2"' in toml_text
    assert "[backend_config.adios2.engine]" in toml_text
    assert 'type = "sst"' in toml_text
    assert 'Profile = "On"' in toml_text
    assert "[[backend_config.adios2.dataset]]" in toml_text
    assert 'type = "blosc"' in toml_text
    assert "select = [" in toml_text
    # the hdf5 global dataset default is also emitted
    assert "[backend_config.hdf5.dataset]" in toml_text
    assert 'chunks = "auto"' in toml_text

    # and the nested dict the C++ layer will stringify matches the model exactly
    assert content["backend_config"] == _adios2_with_per_dataset_overrides().model_dump(mode="json")


def test_unset_backend_config_is_absent():
    """No backend_config key is emitted when it is unset (openPMD defaults apply)."""
    config = OpenPMDConfig(file="simData")
    assert "backend_config" not in config.model_dump(mode="json", exclude_none=True)


def test_explicit_empty_backend_config_is_absent():
    """An explicit-but-empty backend_config is normalised to absent (no spurious empty key)."""
    config = OpenPMDConfig(file="simData", backend_config=OpenPMDBackendConfig())
    assert config.backend_config is None
    assert "backend_config" not in config.model_dump(mode="json", exclude_none=True)


def test_empty_nested_models_are_stripped():
    """Sub-models whose leaves are all unset are dropped, but meaningful empty lists stay."""
    model = OpenPMDBackendConfig(adios2=Adios2Config(engine=Adios2Engine(), dataset=[{"cfg": {"operators": []}}]))
    dumped = model.model_dump(mode="json")
    assert "engine" not in dumped["adios2"]
    # an explicit operators=[] (disabling compression) is preserved
    assert dumped["adios2"]["dataset"] == [{"cfg": {"operators": []}}]


# --------------------------------------------------------------------------- #
# The ``json`` backend key: aliased attribute avoids shadowing BaseModel.json
# --------------------------------------------------------------------------- #
def test_json_backend_key_round_trips():
    """The openPMD key ``json`` is modelled as ``json_config`` so it no longer shadows
    pydantic's deprecated ``BaseModel.json`` (which emitted a ``UserWarning`` and forced the
    former ``warnings.catch_warnings`` hack). It must accept the ``json`` key and serialise
    it back under the openPMD key, not the Python attribute name."""
    model = OpenPMDBackendConfig(backend="json", json=JsonTomlConfig(dataset={"mode": "template"}))
    assert "json" not in OpenPMDBackendConfig.model_fields
    assert model.json_config.dataset.mode == "template"
    assert model.model_dump(mode="json") == {
        "backend": "json",
        "json": {"dataset": {"mode": "template"}},
    }


def test_json_backend_key_accepts_python_name():
    """``populate_by_name`` keeps the Python attribute name usable too."""
    model = OpenPMDBackendConfig(json_config=JsonTomlConfig(attribute={"mode": "short"}))
    assert model.model_dump(mode="json") == {"json": {"attribute": {"mode": "short"}}}


def test_json_backend_key_renders_through_plugin(tmp_path):
    """EFFECT: the rendered config carries the openPMD ``json`` key, not ``json_config``."""
    rendered = _rendered_backend_config(
        tmp_path, OpenPMDBackendConfig(backend="json", json=JsonTomlConfig(dataset={"mode": "template"}))
    )
    assert rendered["json"] == {"dataset": {"mode": "template"}}


def _binning(backend_config):
    electron = Species(particle_type="electron")
    return Binning(
        name="electron_density",
        deposition_functor=BinningFunctor(name="weighting", functor=lambda p: p.get("weighting"), return_type="double"),
        axes=[
            BinningAxis(
                functor=BinningFunctor(name="position0", functor=lambda p: 0.0, return_type="double"),
                bin_spec=BinSpec(kind="linear", start=0, stop=1, nsteps=2),
            )
        ],
        species=electron,
        period=TimeStepSpec[:],
        openPMDBackendConfig=backend_config,
        openPMDExt="h5",
    )


def test_binning_backend_config_serialises_to_json_string():
    """Binning routes the shared model through its existing JSON-string transport
    (setOpenPMDBackendConfig), emitting the model's JSON for a populated config."""
    serialized = (
        _binning(_adios2_with_per_dataset_overrides())
        .get_as_pypicongpu(time_step_size=1.0, num_steps=1)
        .model_dump()["openPMDBackendConfig"]
    )
    assert json.loads(serialized) == _adios2_with_per_dataset_overrides().model_dump(mode="json")


def test_binning_none_backend_config_yields_no_transport():
    serialized = _binning(None).get_as_pypicongpu(time_step_size=1.0, num_steps=1).model_dump()["openPMDBackendConfig"]
    assert serialized is None


def test_binning_accepts_plain_dict():
    """The shared model also accepts a plain dict (coerced), preserving the prior call style."""
    b = _binning({"hdf5": {"dataset": {"chunks": "auto"}}})
    assert isinstance(b.openPMDBackendConfig, OpenPMDBackendConfig)
    serialized = b.get_as_pypicongpu(time_step_size=1.0, num_steps=1).model_dump()["openPMDBackendConfig"]
    assert json.loads(serialized) == {"hdf5": {"dataset": {"chunks": "auto"}}}


# --------------------------------------------------------------------------- #
# B1: rank_table is a string method description, not a bool
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["hostname", "mpi_processor_name", "posix_hostname"])
def test_rank_table_accepts_documented_method_strings(method):
    """openPMD's ``rank_table`` is a string method description; the documented values must be
    expressible and render as a string (a bool is coerced to ``"1"`` and rejected by openPMD)."""
    model = OpenPMDBackendConfig(backend="hdf5", rank_table=method)
    assert model.model_dump(mode="json")["rank_table"] == method


def test_rank_table_rejects_bool():
    """The former ``Optional[bool]`` type accepted ``True``, which openPMD aborts on with
    ``Wrong value for JSON option 'rank_table': '1'``; it must no longer validate."""
    with pytest.raises(Exception):
        OpenPMDBackendConfig(backend="hdf5", rank_table=True)


@pytest.mark.parametrize("method", ["hostname", "posix_hostname"])
def test_rank_table_rendered_value_is_accepted_by_openpmd(tmp_path, method):
    """EFFECT: the rendered value is accepted by the installed openPMD backend."""
    rendered = OpenPMDBackendConfig(backend="hdf5", rank_table=method).model_dump(mode="json")
    series = io.Series(str(tmp_path / "rank_table.h5"), io.Access.create, rendered)
    series.iterations[0].close()
    series.close()


# --------------------------------------------------------------------------- #
# B2: openPMD requires cfg; a valid empty cfg must survive rendering
# --------------------------------------------------------------------------- #
def test_dataset_override_requires_cfg():
    """``cfg`` is mandatory in openPMD's ``JSONMatcher::readPattern`` -- including for the
    default entry -- so the model must not allow its omission."""
    with pytest.raises(Exception):
        DatasetOverride(select=".*")
    with pytest.raises(Exception):
        Hdf5Config(dataset=[{"select": ".*"}])


def test_empty_default_cfg_is_preserved():
    """An explicitly-provided empty ``cfg`` (the docs' default form) must not be stripped,
    otherwise openPMD raises ``Mandatory key missing: 'cfg'!``."""
    model = OpenPMDBackendConfig(hdf5=Hdf5Config(dataset=[{"cfg": {}}]))
    assert model.model_dump(mode="json") == {"hdf5": {"dataset": [{"cfg": {}}]}}


def test_empty_select_cfg_is_preserved_through_rendering(tmp_path):
    """EFFECT: the rendered config keeps the empty ``cfg`` openPMD mandates."""
    rendered = _rendered_backend_config(
        tmp_path, OpenPMDBackendConfig(hdf5=Hdf5Config(dataset=[{"select": ".*E.*", "cfg": {}}]))
    )
    assert rendered == {"hdf5": {"dataset": [{"select": ".*E.*", "cfg": {}}]}}


def test_default_and_select_dataset_entries_are_accepted_by_openpmd(tmp_path):
    """EFFECT: a schema-legal default entry (``cfg = {}``) plus a pattern entry feeds openPMD
    successfully; without the preserved ``cfg`` it would raise ``ErrorBackendConfigSchema``."""
    rendered = OpenPMDBackendConfig(
        backend="hdf5",
        hdf5=Hdf5Config(dataset=[{"cfg": {}}, {"select": ".*E.*", "cfg": {"chunks": "auto"}}]),
    ).model_dump(mode="json")
    series = io.Series(str(tmp_path / "dataset_cfg.h5"), io.Access.create, rendered)
    series.iterations[0].close()
    series.close()


# --------------------------------------------------------------------------- #
# B3: resizable cannot be configured through the Series backend config
# --------------------------------------------------------------------------- #
def test_root_resizable_is_rejected():
    """A top-level ``resizable`` is silently ignored by openPMD (it only reads the key from
    per-``Dataset`` constructor options), so the model rejects it with a clear error rather
    than shipping a no-op knob."""
    with pytest.raises(Exception):
        OpenPMDBackendConfig(backend="hdf5", resizable=True)
    assert "resizable" not in OpenPMDBackendConfig.model_fields
    assert "resizable" not in Hdf5Dataset.model_fields


def test_root_resizable_rejection_message_is_actionable():
    with pytest.raises(Exception, match="per-Dataset"):
        OpenPMDBackendConfig(backend="hdf5", resizable=False)


# --------------------------------------------------------------------------- #
# N1: unknown/typo keys are rejected, not silently swallowed
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "build",
    [
        lambda: Hdf5Config(dataset={"chunkz": "auto"}),
        lambda: Adios2Engine(type="bp5", bogus=1),
        lambda: Adios2Config(dataset=[{"opertors": []}]),
        lambda: OpenPMDBackendConfig(backends="hdf5"),
    ],
)
def test_unknown_keys_are_rejected(build):
    with pytest.raises(Exception):
        build()


# --------------------------------------------------------------------------- #
# Previously-fixed QA items (regression)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "target", ["disk", "buffer", "new_step", "disk_override", "buffer_override", "new_step_override"]
)
def test_adios2_preferred_flush_target_accepts_override_variants(target):
    """openPMD's ``flushTargetFromString`` accepts the ``<value>_override`` variants, which take
    precedence over the non-suffixed values on a per-``flush()`` basis; the model must allow them."""
    assert Adios2Engine(preferred_flush_target=target).preferred_flush_target == target


def test_adios2_preferred_flush_target_rejects_unknown_value():
    with pytest.raises(Exception):
        Adios2Engine(preferred_flush_target="bogus")


@pytest.mark.parametrize(
    "set_backend",
    [
        lambda: Adios2Config(dont_warn_unused_keys=["a"]),
        lambda: Hdf5Config(dont_warn_unused_keys=["b"]),
        lambda: JsonTomlConfig(dont_warn_unused_keys=["c"]),
    ],
)
def test_dont_warn_unused_keys_is_acceptable_per_backend(set_backend):
    """openPMD honours ``dont_warn_unused_keys`` at any (backend) node, and the documented
    examples place it inside a backend table, so each per-backend config must accept it."""
    backend = set_backend()
    assert backend.model_dump(mode="json")["dont_warn_unused_keys"]
