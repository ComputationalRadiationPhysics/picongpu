"""
This file is part of the PIConGPU.

Copyright 2022-2023 PIConGPU contributors
Authors: Pawel Ordyna
License: GPLv3+
"""

import numpy as np
import happi
import scipy.constants as cs
from pathlib import Path
import h5py


class SmileiThermalization:
    def __init__(self, sim_path):
        self.sim = happi.Open(sim_path)
        self.times = np.double(self.sim.ParticleBinning(diagNumber=0, units=["fs"]).getTimes())
        self.e_T_mean = np.empty_like(self.times)
        self.i_T_mean = np.empty_like(self.times)
        self.e_T_std_mean = np.empty_like(self.e_T_mean)
        self.i_T_std_mean = np.empty_like(self.e_T_mean)
        self.e_T_std_dist = np.empty_like(self.e_T_mean)
        self.i_T_std_dist = np.empty_like(self.e_T_mean)

        self.unit_length = cs.c / self.sim.namelist.Main.reference_angular_frequency_SI
        self.pairs = ["ei", "ee", "ii"]
        collider_id_map = {"ei": 0, "ee": 1, "ii": 2}
        self.debug_values = {pair: self._load_debug_output(sim_path, collider_id_map[pair]) for pair in self.pairs}

    def calculate_temperatures(self):
        n_e_m = self.sim.ParticleBinning(9, units=["1/m^3"]).get()
        n_i_m = self.sim.ParticleBinning(8, units=["1/m^3"]).get()
        eps_e_m = self.sim.ParticleBinning(7, units=["eV/m^3"]).get()
        eps_i_m = self.sim.ParticleBinning(6, units=["eV/m^3"]).get()
        for i, t in enumerate(self.times):
            n_e = n_e_m["data"][i]
            n_i = n_i_m["data"][i]
            eps_e = eps_e_m["data"][i]
            eps_i = eps_i_m["data"][i]
            temp_e = (2 / 3) * eps_e / n_e
            temp_i = (2 / 3) * eps_i / n_i
            self.e_T_mean[i] = np.average(temp_e)
            self.e_T_std_dist[i] = np.std(temp_e)
            self.e_T_std_mean[i] = self.e_T_std_dist[i] / np.sqrt(temp_e.size)
            self.i_T_mean[i] = np.average(temp_i)
            self.i_T_std_dist[i] = np.std(temp_i)
            self.i_T_std_mean[i] = self.i_T_std_dist[i] / np.sqrt(temp_i.size)

    def _load_debug_output(self, path, collision_id):
        path_to_debug_output = Path(path) / f"BinaryProcesses{collision_id}.h5"
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
