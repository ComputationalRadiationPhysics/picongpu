"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from functools import reduce
from hashlib import sha256
from os import PathLike
from os.path import relpath
from pathlib import Path
from typing import Annotated, Any, Literal

import tomli_w
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    ValidationError,
    field_validator,
    model_serializer,
)

from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
from picongpu.pypicongpu.particle_functor.filtered_species import FilteredSpecies
from picongpu.pypicongpu.particle_functor.particle_functor import ParticleFunctor
from picongpu.pypicongpu.species.species import Species
from picongpu.pypicongpu.util import unique

NATIVE_FIELDS = ["E", "B", "J"]


class RangeSpecEntry(BaseModel):
    data: None | int | tuple[int, int] = None

    @model_serializer(mode="plain")
    def _serialize(self) -> str:
        if self.data is None:
            return ""
        if isinstance(self.data, int):
            return str(self.data)
        if isinstance(self.data, tuple):
            return ":".join(map(str, self.data))
        raise ValueError(f"Can't serialize RangeSpecEntry with {self.data=}.")


class RangeSpec(BaseModel):
    data: tuple[RangeSpecEntry, RangeSpecEntry, RangeSpecEntry] = (RangeSpecEntry(), RangeSpecEntry(), RangeSpecEntry())

    @model_serializer()
    def _serialize_data(self) -> str:
        return ",".join(map(BaseModel.model_dump, self.data))


class OpenPMDConfig(BaseModel):
    file: PathLike | str
    infix: str = "_%06T"
    ext: Annotated[str, AfterValidator(lambda s: s.strip("."))] = "bp5"
    backend_config: PathLike | None = None
    data_preparation_strategy: Literal["mappedMemory", "doubleBuffer"] = "mappedMemory"
    range: RangeSpec = RangeSpec()

    @field_validator("range", mode="before")
    @classmethod
    def _validate_range(cls, value):
        try:
            return RangeSpec(data=value)
        except ValidationError as error1:
            try:
                return RangeSpec(data=map(lambda x: RangeSpecEntry(data=x), value))
            except ValidationError as error2:
                raise error2 from error1
        return value

    def full_filename(self):
        return f"{self.file}{self.infix}.{self.ext}"

    def result_path(self, prefix_path: PathLike = Path()):
        filename = self.full_filename()
        if Path(filename).is_absolute():
            return filename
        return (Path(prefix_path) / filename).absolute()


def to_string(timestepspec: TimeStepSpec):
    return ",".join(
        map(
            lambda x: "{start}:{stop}:{step}".format(**x),
            timestepspec.get_rendering_context()["specs"],
        )
    )


class FieldDump(BaseModel):
    name: str
    functor: ParticleFunctor | None = None
    filtername: None | str
    species_name: str | None = None
    """Compile-time name of the source species this derived field is defined for.

    For native field dumps (E/B/J) this is ``None`` and no species-eligibility
    narrowing is generated. For derived fields it carries the species name so
    that the ``SpeciesEligibleForSolver`` trait can be specialised on the
    species' compile-time name, restricting the derived field to only the
    species(es) it is actually used for."""

    def get_rendering_context(self) -> dict:
        return self.model_dump(mode="json")


class OpenPMDPlugin(BaseModel):
    sources: list[tuple[TimeStepSpec, Species | FieldDump | FilteredSpecies]]
    config: OpenPMDConfig = OpenPMDConfig(file="simData")

    type_openPMD: Literal[True] = True

    def config_filename(self, content) -> str:
        # Content-hash-only: the plugin is stored in a per-run directory, so the
        # hash of its content uniquely identifies the config within a run.
        return f"openPMD_config_{sha256(tomli_w.dumps(content).encode()).hexdigest()}.toml"

    @property
    def _config_content(self):
        # There's some strange interaction with the custom hashing of TimeStepSpec
        # that's implemented on RenderedObject
        # hindering the storage of this data structure.
        # As a workaround, we're computing this on the fly.
        # Shouldn't be performance critical but it would be more elegant to normalise early on.
        sources = reduce(
            lambda dictionary, key_val: (
                dictionary.setdefault(to_string(key_val[0]), []).append(key_val[1].get_rendering_context()["name"])
                or dictionary
            ),
            self.sources,
            {},
        )
        return self.config.model_dump(mode="json", exclude_none=True) | {
            "sink": {"dummy_application_name": {"period": sources}}
        }

    def write_config_file(self, setup_dir):
        """
        Write the openPMD backend config into the render root's ``etc`` dir.

        ``setup_dir`` (the render root, i.e. ``run_dir/input``) is a layout
        concern that belongs to the rendering/generation layer, not to the
        model; passing it in explicitly keeps ``_get_serialized()`` a pure
        function of the model.
        """
        path = setup_dir / "etc" / self.config_filename(self._config_content)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            tomli_w.dump(self._config_content, file)
        return path

    @model_serializer(mode="plain")
    def _get_serialized(self) -> dict[str, Any] | None:
        filename = self.config_filename(self._config_content)
        return {
            "type_openPMD": True,
            # The batch job runs with its working directory set to the run's
            # ``simOutput`` dir (see the TBG batch templates), while the config is
            # written to ``input/etc`` (the render root). Deriving the path
            # relative to that CWD keeps it independent of where the run dir
            # lives on disk and of the active preset.
            "config_filename": relpath(Path("input") / "etc" / filename, "simOutput"),
            "derived_fields": unique(
                source[1].model_dump(mode="json")
                for source in self.sources
                if isinstance(source[1], FieldDump) and source[1].functor is not None
            ),
        }

    model_config = ConfigDict(arbitrary_types_allowed=True)
