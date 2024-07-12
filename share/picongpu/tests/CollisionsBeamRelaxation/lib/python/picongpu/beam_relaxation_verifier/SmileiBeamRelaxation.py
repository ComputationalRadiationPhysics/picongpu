"""
This file is part of PIConGPU.

Copyright 2022-2023 PIConGPU contributors
Authors: Pawel Ordyna
License: GPLv3+
"""

import numpy as np
import happi
import h5py
import scipy.constants as cs

from pathlib import Path


class SmileiBeamRelaxation:
    def __init__(self, path_equal, path_less_ions, path_more_ions):
        self.ratios = ["equal", "lessIons", "moreIons"]

        def get_iterator():
            return zip(self.ratios, [path_equal, path_less_ions, path_more_ions])

        self.sims = {key: happi.Open(path) for key, path in get_iterator()}
        self.unit_length = cs.c / self.sims[self.ratios[0]].namelist.Main.reference_angular_frequency_SI
        self.debug_values = {key: self._load_debug_output(path) for key, path in get_iterator()}

        def get_times(sim):
            return sim.ParticleBinning(diagNumber=0, units=["fs"]).getTimes()

        self.times_dict = {key: get_times(sim) for key, sim in self.sims.items()}

        self.quantities = ["e_vperp", "e_vx", "i_vx"]
        self.calculated_data = {}
        for key in ["mean", "std_mean", "std_dist"]:
            self.calculated_data[key] = {}
            for quantity in self.quantities:
                self.calculated_data[key][quantity] = {}
                for ratio in self.ratios:
                    self.calculated_data[key][quantity][ratio] = np.zeros_like(self.times_dict[ratio], dtype=np.float64)

    def _calculate_values(self, ratio):
        sim = self.sims[ratio]
        w_vx_e = sim.ParticleBinning(
            0,
        ).get()
        w_vperp2_e = sim.ParticleBinning(1).get()
        w_vx_i = sim.ParticleBinning(2).get()
        n_i = sim.ParticleBinning(3).get()
        n_e = sim.ParticleBinning(4).get()
        # cell_length = self.sims[ratio]

        for i, t in enumerate(self.times_dict[ratio]):
            local_array_dict = {
                "e_vx": w_vx_e["data"][i] / n_e["data"][i],
                "i_vx": w_vx_i["data"][i] / n_i["data"][i],
                "e_vperp": np.sqrt(w_vperp2_e["data"][i] / n_e["data"][i]),
            }

            for quantity in self.quantities:
                data = local_array_dict[quantity]
                self.calculated_data["mean"][quantity][ratio][i] = np.average(data)
                self.calculated_data["std_mean"][quantity][ratio][i] = np.std(data) / np.sqrt(data.size)
                self.calculated_data["std_dist"][quantity][ratio][i] = np.std(data)

    def calculate_values(self):
        for ratio in self.ratios:
            self._calculate_values(ratio)

    def _load_debug_output(self, path):
        path_to_debug_output = Path(path) / "BinaryProcesses0.h5"
        debug_output_file = h5py.File(path_to_debug_output)
        n_iterations = len(list(debug_output_file))
        coulomb_log = np.empty(n_iterations, dtype=np.float64)
        s_param = np.empty(n_iterations, dtype=np.float64)
        debye_length = np.empty(n_iterations, dtype=np.float64)
        for ii, iteration in enumerate(list(debug_output_file)):
            debye_length[ii] = np.average(debug_output_file[iteration]["debyelength"])
            coulomb_log[ii] = np.average(debug_output_file[iteration]["coulomb_log"])
            s_param[ii] = np.average(debug_output_file[iteration]["s"])
        output = np.empty(
            (n_iterations,),
            dtype=[
                ("coulomb_log", np.float64),
                ("s_param", np.float64),
                ("debye_length", np.float64),
            ],
        )
        output["coulomb_log"] = coulomb_log
        output["s_param"] = s_param
        output["debye_length"] = debye_length * self.unit_length
        return output
