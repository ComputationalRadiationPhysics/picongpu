# Handover: PICMI Native and Combined Derived-Field Output

## Purpose

This PIConGPU checkout contains an uncommitted Python implementation that lets a PICMI setup request PIConGPU's native particle-to-grid fields in openPMD output without defining an equivalent Python `ParticleFunctor`.

It adds two diagnostics:

- `NativeDerivedFieldDump`: use an existing PIConGPU `derivedAttributes` or supported parameter-free `combinedAttributes` implementation.
- `AverageDerivedFieldDump`: apply PIConGPU's native `combinedAttributes::AverageAttribute` to an eligible native derived attribute.

The existing `DerivedFieldDump` remains the interface for custom Python particle functors.

## Basic Setup-Repo Usage

Import the diagnostics from the public diagnostics package:

```python
from picongpu.picmi.diagnostics import (
    AverageDerivedFieldDump,
    DerivedFieldDump,
    NativeDerivedFieldDump,
    TimeStepSpec,
)
```

Add the requested diagnostics to `Simulation(picongpu_diagnostics=[...])`.

### Native scalar field

```python
electron_density = NativeDerivedFieldDump(
    species=electrons,
    field="Density",
    period=TimeStepSpec[::100],
)
```

### Native directional field

Directional fields require `direction="x"`, `"y"`, or `"z"`:

```python
momentum_x = NativeDerivedFieldDump(
    species=electrons,
    field="Momentum",
    direction="x",
    period=TimeStepSpec[::100],
)
```

### Raw weighted velocity

`WeightedVelocity` is exposed directly. It is the deposited weighted sum, not the mean velocity:

```python
weighted_velocity_x = NativeDerivedFieldDump(
    species=electrons,
    field="WeightedVelocity",
    direction="x",
    period=TimeStepSpec[::100],
)
```

### Average velocity

Use `AverageDerivedFieldDump` to divide weighted velocity by the native density field:

```python
average_velocity_x = AverageDerivedFieldDump(
    species=electrons,
    field="WeightedVelocity",
    direction="x",
    period=TimeStepSpec[::100],
)
```

This renders the native C++ type:

```cpp
deriveField::combinedAttributes::AverageAttribute<
    deriveField::derivedAttributes::WeightedVelocity<0>
>
```

### Other combined fields

Existing parameter-free combined fields are selected through `NativeDerivedFieldDump`:

```python
relativistic_density = NativeDerivedFieldDump(
    species=electrons,
    field="RelativisticDensity",
    period=TimeStepSpec[::100],
)

inverse_screening_length_squared = NativeDerivedFieldDump(
    species=electrons,
    field="ScreeningInvSquared",
    period=TimeStepSpec[::100],
)
```

### Filtered species

Native and averaged fields accept `FilteredSpecies` in the same way as custom derived fields:

```python
filtered_density = NativeDerivedFieldDump(
    species=FilteredSpecies(
        species=electrons,
        functor=energy_filter,
    ),
    field="Density",
    period=TimeStepSpec[::100],
)
```

The particle filter is part of the compile-time solver identity.

### Existing custom functor API

No API change is needed for custom fields:

```python
@ParticleFunctor
def kinetic_energy_density(particle):
    return particle.get("kinetic energy") / np.prod(CELL_SIZE)

custom_energy_density = DerivedFieldDump(
    species=electrons,
    functor=kinetic_energy_density,
    period=TimeStepSpec[::100],
)
```

## Supported Native Fields

Scalar fields:

- `Density`
- `BoundElectronDensity`
- `ChargeDensity`
- `Counter`
- `Energy`
- `EnergyDensity`
- `LarmorPower`
- `MacroCounter`

Directional fields:

- `MidCurrentDensityComponent`
- `Momentum`
- `MomentumDensity`
- `WeightedVelocity`

Parameter-free combined fields:

- `RelativisticDensity`
- `ScreeningInvSquared`

`EnergyDensityCutoff` is intentionally unsupported because it requires a user-provided C++ parameter class.

`AverageDerivedFieldDump` accepts fields supported by PIConGPU's `IsWeighted` trait. It rejects `MacroCounter` and already-combined fields.

## Expected openPMD Names

The record name format remains:

```text
<species>_<filter-or-all>_<native PIConGPU field name>
```

Examples:

```text
electrons_all_density
electrons_all_weightedVelocity/x
electrons_all_Average_weightedVelocity/x
electrons_all_relativisticDensity
electrons_all_invSquaredScreenLength
```

These are the names to use when reading the records through openPMD-api.

## Species-Specific Compilation Behavior

Derived-field solvers are generated only for the species selected in the diagnostic. They no longer use:

```cpp
CreateEligible_t<VectorAllSpecies, ...>
```

Instead, the generated `fileOutput.param` places each species-specific operation directly in the common solver sequence:

```cpp
using FieldTmpSolvers = MakeSeq_t<
    deriveField::CreateFieldTmpOperation_t<
        species_electrons,
        Native_Density_species_electrons_All_Attribute,
        Native_Density_species_electrons_All_Filter
    >
>;
```

Before instantiating it, the generated code explicitly checks the selected derived attribute and filter:

```cpp
static_assert(
    particles::traits::SpeciesEligibleForSolver<
        species_electrons,
        deriveField::FilteredDerivedAttribute<
            Native_Density_species_electrons_All_Attribute,
            Native_Density_species_electrons_All_Filter
        >
    >::type::value,
    "Derived field Native_Density ... is not supported for species_electrons"
);
```

An incompatible species/field/filter request should therefore fail during compilation instead of producing an empty solver list followed by a missing-source error at runtime.

## Deduplication and Metadata

The top-level PyPIConGPU simulation model contains:

- `derived_field_functors`: unique custom functor definitions;
- `field_tmp_solvers`: unique species-specific solver instances.

A solver is deduplicated by the complete combination of:

```text
species + derived attribute/custom functor + particle filter
```

Consequences:

- Repeating the same field for the same species and filter at several periods or in several openPMD configurations creates one C++ solver.
- The same field for two species creates two solvers.
- The same field and species with two filters creates two solvers.
- One custom particle functor can be defined once and reused by several species-specific solvers.

Each openPMD plugin retains its complete `sources` metadata, including period, record name, species, filter, and solver/functor description.

## Checking Generated Input

To generate input without running the simulation, use PICMI's write method:

```python
sim.write_input_file("path/to/generated-setup")
```

Inspect:

```text
path/to/generated-setup/include/picongpu/param/fileOutput.param
path/to/generated-setup/etc/openPMD_config_<hash>.toml
path/to/generated-setup/metadata/pypicongpu_rendering_context.json
```

Expected checks:

- Native fields do not generate custom C++ functor structs.
- Custom `DerivedFieldDump` functors are each defined once.
- `FieldTmpSolvers` directly contains the species-specific `CreateFieldTmpOperation_t` types, without one-element intermediate sequences.
- Derived-field aliases do not use `VectorAllSpecies` or `CreateEligible_t`.
- Every derived-field solver has an eligibility `static_assert`.
- The openPMD TOML lists the expected record names and periods.

## Example in This Checkout

`lib/python/examples/tutorial/03.2_particle_functors.py` contains examples of:

- a custom energy-density functor;
- native density;
- raw native weighted velocity;
- native average velocity.

For a write-only check, temporarily replace its final `sim.run(...)` call with:

```python
sim.write_input_file("/tmp/picongpu-032-check")
```

Do not commit that temporary invocation change unless the tutorial is intentionally being converted to write-only behavior.

## Current Implementation and Validation State

The implementation is currently present as uncommitted changes in the PIConGPU checkout. Before relying on it from another setup repository, ensure that setup resolves/imports this exact local `lib/python` package rather than an installed release lacking the changes.

Relevant implementation areas:

- `lib/python/picongpu/picmi/diagnostics/field_dump.py`
- `lib/python/picongpu/pypicongpu/output/openpmd_plugin.py`
- `lib/python/picongpu/pypicongpu/simulation.py`
- `lib/python/picongpu/templates/include/picongpu/param/fileOutput.param.mustache`

Validation already performed:

```text
198 passed, 3 xfailed, 3499 subtests passed
```

The tutorial generated successfully through `write_input_file()`, and its `fileOutput.param`, rendering metadata, and openPMD TOML were inspected.

The test command used Python 3.12 and disabled the repository pytest configuration because the installed upstream `picmistandard` emits an invalid docstring-escape warning that the repository promotes to an error:

```bash
cd lib/python
.venv/bin/python -m pytest -c /dev/null -q test/picongpu/quick
```

Focused tests are located at:

- `lib/python/test/picongpu/quick/picmi/diagnostics/test_derived_field_dump.py`
- `lib/python/test/picongpu/quick/pypicongpu/test_openpmd_derived_fields.py`
