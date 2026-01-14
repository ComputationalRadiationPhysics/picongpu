"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from pathlib import Path
import numpy as np
import openpmd_api as opmd
import pandas as pd
from picongpu.picmi.diagnostics import ParticleDump, Binning
from .arbitrary_parameters import CELL_SIZE
from openpmd_api.openpmd_api_cxx import ErrorWrongAPIUsage


def load_diagnostic_result(diagnostic, result_path):
    path = Path(str(diagnostic.result_path(result_path)).replace("%06T", 6 * "0"))
    if isinstance(diagnostic, ParticleDump):
        return read_particles(path).loc(axis=0)[*diagnostic.species.name.split("_", maxsplit=1)]
    if isinstance(diagnostic, Binning):
        return read_fields(path, ["Binning"])["Binning"]
    return read_fields(path, [diagnostic.fieldname])[diagnostic.fieldname]


def _normalize_range_spec_entry(data):
    if data.data is None:
        return (-np.inf, np.inf)
    if isinstance(data.data, int):
        return (data.data, data.data + 1)
    return data.data


def apply_range(particles, range):
    lower_bound, upper_bound = zip(*map(_normalize_range_spec_entry, range.data))
    cells = np.round(
        particles[["positionOffset_x", "positionOffset_y", "positionOffset_z"]].to_numpy() / CELL_SIZE
    ).astype(int)
    mask = np.all((cells >= lower_bound) * (cells < upper_bound), axis=1)
    return particles.loc(axis=0)[mask]


def read_fields(series_name, names=("E", "B")):
    series = opmd.Series(str(series_name), opmd.Access.read_only)
    tmp = {}
    for name in names:
        try:
            tmp[name] = [series.iterations[0].meshes[name][c].load_chunk() for c in "xyz"]
        except ErrorWrongAPIUsage:
            tmp[name] = series.iterations[0].meshes[name].load_chunk()
    series.flush()
    return {key: np.array(value) for key, value in tmp.items()}


def read_particles(series_name):
    series = opmd.Series(str(series_name), opmd.Access.read_only)
    names, particles = zip(*series.iterations[0].particles.items())

    data = pd.concat((particle.to_df() for particle in particles), keys=names)
    return data.assign(
        setup=data.index.to_frame()[0].apply(lambda x: x.split("_")[0]),
        impl=data.index.to_frame()[0].apply(lambda x: "_".join(x.split("_")[1:])),
    ).set_index(["setup", "impl"], drop=True)


def sort_particles(data):
    return data.groupby(["setup", "impl"]).apply(lambda df: df.sort_values(list(df.columns), axis=0))


def compare_particles_per_setup(data):
    result = True
    for setup_name, df in sort_particles(data).groupby("setup"):
        # We're doing (more than) twice the work here but that's fine:
        for lhs_name, lhs_df in df.groupby("impl"):
            for rhs_name, rhs_df in df.groupby("impl"):
                matched = np.allclose(
                    lhs_df.reset_index(drop=True),
                    rhs_df.reset_index(drop=True),
                    atol=0,
                    rtol=1.0e-4,
                )
                if not matched:
                    print(f"Mismatch found in {setup_name}: {lhs_name} != {rhs_name}")
                result *= matched
    return result


def compute_densities_per_setup_and_impl(data):
    return data.groupby(["setup", "impl", "positionOffset_x", "positionOffset_y", "positionOffset_z"])[
        "weighting"
    ].sum()


def compute_densities_from_particles(series_name):
    """
    This density is not normalised by volume yet.
    """
    return compute_densities_per_setup_and_impl(read_particles(series_name))


def _density_into_mesh(df, number_of_cells, cell_size):
    from_particles = np.zeros(number_of_cells)
    from_particles[*df.index.to_frame().to_numpy().T] = df["weighting"].to_numpy()
    return from_particles / np.prod(cell_size)


def read_densities_into_mesh(filename, number_of_cells, cell_size):
    df = (
        compute_densities_from_particles(filename)
        .reset_index(drop=False)
        .rename({"positionOffset_" + key: key for key in "xyz"}, axis=1)
    )

    for i, key in enumerate("xyz"):
        df[key] *= 1 / cell_size[i]
        df[key] = df[key].round()

    return (
        df.astype(dict(x=int, y=int, z=int))
        .set_index(["x", "y", "z"])
        .groupby(["setup", "impl"])
        .apply(
            lambda df: _density_into_mesh(df, number_of_cells, cell_size),
            include_groups=False,
        )
    )


def compare_particles(series_name):
    """
    Compare particles from the given series and return True if they all compare equal.
    """
    return compare_particles_per_setup(read_particles(series_name))


def read_binning(filename, cell_size):
    series = opmd.Series(
        str(filename),
        opmd.Access.read_only,
    )
    data = series.iterations[0].meshes["Binning"]
    mesh = data.load_chunk()
    series.flush()
    series.close()
    return mesh / np.prod(cell_size)


def read_position_check(filename):
    series = opmd.Series(
        str(filename),
        opmd.Access.read_only,
    )
    data = series.iterations[0].meshes["Binning"]
    mesh = data.load_chunk()
    series.flush()
    series.close()
    # There's an under-/overflow bin respectively.
    return mesh[1]
