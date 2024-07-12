#!/usr/bin/env python
"""
This file is part of PIConGPU.

Copyright 2022-2023 PIConGPU contributors
Authors: Pawel Ordyna
License: GPLv3+
"""

import itertools
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openpmd_api as api
import scipy.constants as cs
import seaborn as sns


def plot_with_std(ax, x, y, y_std, label=None):
    ax.plot(x, y, label=label)
    ax.fill_between(x, y - 2 * y_std, y + 2 * y_std, alpha=0.1)


def _get_debug_data(sim_path, collider_id):
    debug_file_path = Path(sim_path) / "simOutput" / f"debug_values_collider_{collider_id}_" "species_pair_0.dat"
    debug_data = None
    with open(debug_file_path, "r") as f:
        debug_data = np.loadtxt(
            f,
            dtype=[
                ("iteration", np.uint32),
                ("coulomb_log", np.float64),
                ("s_param", np.float64),
            ],
        )
    return debug_data


class BeamRelaxationVerifier:
    def __init__(self, sim_output_path):
        sim_path = Path(sim_output_path) / "simOutput/openPMD/simData_%T.bp"
        self.series = api.Series(str(sim_path.resolve()), api.Access_Type.read_only)

        def empty():
            return np.zeros(len(self.series.iterations), dtype=np.float64)

        self.quantities = ["e_vperp", "e_vx", "i_vx"]
        self.ratios = ["equal", "moreIons", "lessIons"]
        self.calculated_data = {}
        for key in ["mean", "std_mean", "std_dist"]:
            self.calculated_data[key] = {}
            for quantity in self.quantities:
                self.calculated_data[key][quantity] = {}
                for ratio in self.ratios:
                    self.calculated_data[key][quantity][ratio] = empty()

        self.sim_output_path = sim_output_path
        self.dt = self.series.iterations[0].dt * self.series.iterations[0].time_unit_SI

        collider_id_map = {"equal": 0, "lessIons": 1, "moreIons": 2}
        self.debug_values = {ratio: _get_debug_data(sim_output_path, collider_id_map[ratio]) for ratio in self.ratios}
        debug_file_path = Path(sim_output_path) / "simOutput" / "average_debye_length_for_collisions.dat"
        self.average_debye_present = True
        try:
            with open(debug_file_path, "r") as f:
                self.debug_values["all"] = np.loadtxt(f, dtype=[("iteration", np.uint32), ("debye_length", np.float64)])
        except FileNotFoundError:
            print("No average debye length output present")
            self.average_debye_present = False

    def calculate_values(self, n_cells=None):
        """Calculates mean temperatures for electrons and ions"""
        vperp_field_name = "all_Average_vPerp2"
        vx_field_name = "all_Average_particleVelocityX"
        for i, it in enumerate(self.series.iterations):
            iteration = self.series.iterations[it]
            local_array_dict = {
                "e_vperp": {"equal": None, "moreIons": None, "lessIons": None},
                "e_vx": {"equal": None, "moreIons": None, "lessIons": None},
                "i_vx": {"equal": None, "moreIons": None, "lessIons": None},
            }
            species_prefix = {
                "equal": {"e": "e_1000", "i": "i_1000"},
                "moreIons": {"e": "e_100", "i": "i_1000"},
                "lessIons": {"e": "e_1000", "i": "i_100"},
            }

            x_cell_range = {
                "equal": (0, 32),
                "lessIons": (32, 64),
                "moreIons": (64, 96),
            }
            for ratio in self.ratios:
                e_vperp_mrc = iteration.meshes[species_prefix[ratio]["e"] + "_" + vperp_field_name][
                    api.Mesh_Record_Component.SCALAR
                ]
                local_array_dict["e_vperp"][ratio] = e_vperp_mrc[:, x_cell_range[ratio][0] : x_cell_range[ratio][1]]
                e_vx_mrc = iteration.meshes[species_prefix[ratio]["e"] + "_" + vx_field_name][
                    api.Mesh_Record_Component.SCALAR
                ]
                local_array_dict["e_vx"][ratio] = e_vx_mrc[:, x_cell_range[ratio][0] : x_cell_range[ratio][1]]
                i_vx_mrc = iteration.meshes[species_prefix[ratio]["i"] + "_" + vx_field_name][
                    api.Mesh_Record_Component.SCALAR
                ]
                local_array_dict["i_vx"][ratio] = i_vx_mrc[:, x_cell_range[ratio][0] : x_cell_range[ratio][1]]

            self.series.flush()

            for ratio in self.ratios:
                local_array_dict["e_vperp"][ratio] *= iteration.meshes[
                    species_prefix[ratio]["e"] + "_" + vperp_field_name
                ][api.Mesh_Record_Component.SCALAR].unit_SI
                local_array_dict["e_vx"][ratio] *= iteration.meshes[species_prefix[ratio]["e"] + "_" + vx_field_name][
                    api.Mesh_Record_Component.SCALAR
                ].unit_SI
                local_array_dict["i_vx"][ratio] *= iteration.meshes[species_prefix[ratio]["i"] + "_" + vx_field_name][
                    api.Mesh_Record_Component.SCALAR
                ].unit_SI

            if n_cells is not None:
                for quantity, ratio in itertools.product(self.quantities, self.ratios):
                    local_array_dict[quantity][ratio] = np.ravel(local_array_dict[quantity][ratio])[0:n_cells]

            for ratio in self.ratios:
                local_array_dict["e_vperp"][ratio] = np.sqrt(local_array_dict["e_vperp"][ratio])

            for quantity, ratio in itertools.product(self.quantities, self.ratios):
                data = local_array_dict[quantity][ratio]
                self.calculated_data["mean"][quantity][ratio][i] = np.average(data)
                self.calculated_data["std_mean"][quantity][ratio][i] = np.std(data) / np.sqrt(data.size)
                self.calculated_data["std_dist"][quantity][ratio][i] = np.std(data)

    def plot(self, to_file=False, file_name=None, smilei_loader=None):
        times = np.array(self.series.iterations) * self.dt * 1e15
        f, ax_dict = plt.subplot_mosaic(
            [
                [
                    "equal_std_mean",
                    "equal_std_mean",
                    "moreIons_std_mean",
                    "moreIons_std_mean",
                    "lessIons_std_mean",
                    "lessIons_std_mean",
                ],
                [
                    "equal_std_dist",
                    "equal_std_dist",
                    "moreIons_std_dist",
                    "moreIons_std_dist",
                    "lessIons_std_dist",
                    "lessIons_std_dist",
                ],
                [
                    "all_pic",
                    "all_pic",
                    "all_pic",
                    "all_smilei",
                    "all_smilei",
                    "all_smilei",
                ],
            ],
            sharey=True,
            gridspec_kw={"height_ratios": [1, 1, 1]},
            figsize=(20, 22),
        )
        label_map = {
            "e_vperp": r"$v_{e, \perp}$",
            "e_vx": r"$v_{e,x}$",
            "i_vx": r"$v_{i,x}$",
        }
        for std_type in ["std_mean", "std_dist"]:
            for ratio in self.ratios:
                for quantity in self.quantities:
                    ax = ax_dict[ratio + "_" + std_type]
                    plot_with_std(
                        ax,
                        times,
                        self.calculated_data["mean"][quantity][ratio] / cs.c,
                        self.calculated_data[std_type][quantity][ratio] / cs.c,
                        label=label_map[quantity] + " PIConGPU",
                    )
                    if smilei_loader is not None:
                        plot_with_std(
                            ax,
                            smilei_loader.times_dict[ratio],
                            smilei_loader.calculated_data["mean"][quantity][ratio],
                            smilei_loader.calculated_data[std_type][quantity][ratio],
                            label=label_map[quantity] + " smilei",
                        )
        palette = itertools.cycle(sns.color_palette("colorblind"))
        line_styles = itertools.cycle(["-", "--", ":"])
        for quantity in self.quantities:
            color = next(palette)
            for ratio in self.ratios:
                line_style = next(line_styles)
                ax_dict["all_pic"].plot(
                    times,
                    self.calculated_data["mean"][quantity][ratio] / cs.c,
                    label=(label_map[quantity] + " " + ratio),
                    color=color,
                    ls=line_style,
                )
                if smilei_loader is not None:
                    ax_dict["all_smilei"].plot(
                        smilei_loader.times_dict[ratio],
                        smilei_loader.calculated_data["mean"][quantity][ratio],
                        label=(label_map[quantity] + " " + ratio),
                        color=color,
                        ls=line_style,
                    )

        for ax in ax_dict.values():
            ax.legend()

        ax_dict["equal_std_mean"].set_title(
            "(equal) 1000 ions per cell, 1000 electrons per cell \n with std " "of the mean"
        )
        ax_dict["moreIons_std_mean"].set_title(
            "(moreIons) 1000 ions per cell, 100 electrons per cell \n with " "std of the mean"
        )
        ax_dict["lessIons_std_mean"].set_title(
            "(lessIons) 100 ions per cell, 1000 electrons per cell \n with " "std of the mean"
        )
        ax_dict["equal_std_dist"].set_title(
            "(equal) 1000 ions per cell, 1000 electrons per cell \n with std " "of the distribution"
        )
        ax_dict["moreIons_std_dist"].set_title(
            "(moreIons) 1000 ions per cell, 100 electrons per cell \n with " "std of the distribution"
        )
        ax_dict["lessIons_std_dist"].set_title(
            "(lessIons) 100 ions per cell, 1000 electrons per cell \n with " "std of the distribution"
        )

        ax_dict["all_pic"].set_title("all PIConGPU")
        ax_dict["all_smilei"].set_title("all smilei")

        ax_dict["equal_std_dist"].set_ylabel("c")
        ax_dict["equal_std_mean"].set_ylabel("c")
        ax_dict["all_pic"].set_ylabel("c")

        for ax in ax_dict.values():
            ax.set_xlabel("t [fs]")

        with open(Path(self.sim_output_path) / "input/cmakeFlagsSetup", "r") as file:
            f.suptitle(file.readline() + "\n" + textwrap.fill(file.readline(), 80))
        f.tight_layout()
        f.subplots_adjust(top=0.90)
        if to_file:
            if file_name is None:
                file_name = "beam_relaxation_plot.png"
            f.savefig(file_name)

    def plot_debug_values(self, to_file=False, file_name=None, smilei_loader=None):
        f, axs = plt.subplots(1, 3, figsize=(15, 5))
        line_styles = itertools.cycle(["-", "--", ":"])
        times = self.debug_values["equal"]["iteration"] * self.dt * 1e15
        for ratio in self.ratios:
            line_style = next(line_styles)
            axs[0].plot(
                times,
                self.debug_values[ratio]["coulomb_log"],
                label=rf"$\ln \Lambda$ PIConGPU {ratio}",
                color="red",
                ls=line_style,
                alpha=0.75,
            )
            axs[1].plot(
                times,
                self.debug_values[ratio]["s_param"],
                label=rf"$s$ PIConGPU {ratio}",
                color="red",
                ls=line_style,
                alpha=0.75,
            )
            if smilei_loader is not None:
                times_smilei = smilei_loader.times_dict[ratio][:-1]
                axs[0].plot(
                    times_smilei,
                    smilei_loader.debug_values[ratio]["coulomb_log"],
                    label=rf"$\ln \Lambda$ smilei {ratio}",
                    color="black",
                    ls=line_style,
                    alpha=0.75,
                )
                axs[1].plot(
                    times_smilei,
                    smilei_loader.debug_values[ratio]["s_param"],
                    label=rf"$s$ smilei {ratio}",
                    color="black",
                    ls=line_style,
                    alpha=0.75,
                )
                axs[2].plot(
                    times_smilei,
                    smilei_loader.debug_values[ratio]["debye_length"],
                    label=rf"$\lambda_D$ smilei {ratio}",
                    color="black",
                    ls=line_style,
                    alpha=0.75,
                )
        if self.average_debye_present:
            axs[2].plot(
                times,
                self.debug_values["all"]["debye_length"],
                color="red",
                label=r"$\lambda_D$ PIConGPU",
                alpha=0.75,
            )
        for ax in axs:
            ax.legend()
            ax.set_xlabel(r"$t [fs]$")
        axs[0].set_ylabel(r"$\ln\Lambda$")
        axs[1].set_ylabel(r"$s$")
        axs[2].set_ylabel(r"$\lambda_D [m]$")

        with open(Path(self.sim_output_path) / "input/cmakeFlagsSetup", "r") as file:
            f.suptitle(file.readline() + "\n" + textwrap.fill(file.readline(), 80))
        plt.tight_layout()
        if to_file:
            if file_name is None:
                file_name = "debug_values_plot.png"
            f.savefig(file_name)
