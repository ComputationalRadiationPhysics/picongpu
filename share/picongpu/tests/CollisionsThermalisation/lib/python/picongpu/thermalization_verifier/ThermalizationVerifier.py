"""
This file is part of the PIConGPU.

Copyright 2022-2023 PIConGPU contributors
Authors: Pawel Ordyna
License: GPLv3+
"""

from os import path
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as cs
import openpmd_api as api
import textwrap
import seaborn as sns
import itertools


def _get_debug_data(sim_path, collider_id, pair_id):
    debug_file_path = (Path(sim_path) / 'simOutput'
                       / f'debug_values_collider_{collider_id}_'
                         f'species_pair_{pair_id}.dat')
    debug_data = None
    with open(debug_file_path, 'r') as f:
        debug_data = np.loadtxt(f,
                                dtype=[('iteration', np.uint32),
                                       ('coulomb_log', np.float64),
                                       ('s_param', np.float64)])
    return debug_data


class ThermalizationVerifier:
    """Verifies thermalization test output and generates reference output"""
    ELECTRON_MASS = cs.electron_mass
    ION_MASS = 10 * ELECTRON_MASS
    REFERENCE_FILE_NAME = "reference_output.npz"
    ION_DENSITY = 1.1e28
    ELECTRON_DENSITY = ION_DENSITY
    ION_CHARGE = 1
    INIT_TEMP_IONS = (1.8e-4 * cs.electron_mass * cs.speed_of_light ** 2
                      * 6.241509e18)  # eV
    INIT_TEMP_ELECTRONS = (2e-4 * cs.electron_mass * cs.speed_of_light ** 2
                           * 6.241509e18)  # eV

    def __init__(self, sim_output_path):
        self.series = api.Series(
            path.join(sim_output_path, 'simOutput/openPMD/simData_%T.bp'),
            api.Access_Type.read_only)
        self.unit_mass = self.series.iterations[0].get_attribute('unit_mass')
        self.e_T_mean = np.zeros(len(self.series.iterations), dtype=np.float64)
        self.i_T_mean = np.zeros(len(self.series.iterations), dtype=np.float64)
        self.e_T_std_mean = np.zeros(len(self.series.iterations),
                                     dtype=np.float64)
        self.i_T_std_mean = np.zeros(len(self.series.iterations),
                                     dtype=np.float64)
        self.e_T_std_dist = np.zeros(len(self.series.iterations),
                                     dtype=np.float64)
        self.i_T_std_dist = np.zeros(len(self.series.iterations),
                                     dtype=np.float64)
        self.dt = self.series.iterations[0].dt * self.series.iterations[
            0].time_unit_SI
        self.e_T_theory = None
        self.i_T_theory = None
        self.coulomb_log = None
        self.sim_output_path = sim_output_path

        self.pairs = ['ei', 'ee', 'ii']
        collider_id_map = {'ei': 1, 'ee': 0, 'ii': 0}
        pair_id_map = {'ei': 0, 'ee': 0, 'ii': 1}
        self.debug_values = {
            pair: _get_debug_data(sim_output_path, collider_id_map[pair],
                                  pair_id_map[pair]) for pair in self.pairs}
        debug_file_path = (Path(sim_output_path) / 'simOutput'
                           / 'average_debye_length_for_collisions.dat')
        self.average_debye_present = True
        try:
            with open(debug_file_path, 'r') as f:
                self.debug_values['all'] = np.loadtxt(f,
                                                      dtype=[('iteration',
                                                              np.uint32), (
                                                                'debye_length',
                                                                np.float64)])
        except FileNotFoundError:
            print('No average debye length output present')
            self.average_debye_present = False

    def calculate_temperatures(self, n_cells=None):
        """Calculates mean temperatures for electrons and ions"""
        iterations = self.series.iterations
        for i in iterations:
            iteration = self.series.iterations[i]
            average_e_energy_m = \
                iteration.meshes['e_all_Average_particleEnergy'][
                    api.Mesh_Record_Component.SCALAR]
            average_e_energy = average_e_energy_m[:]
            average_i_energy_m = \
                iteration.meshes['i_all_Average_particleEnergy'][
                    api.Mesh_Record_Component.SCALAR]
            average_i_energy = average_i_energy_m[:]
            self.series.flush()
            average_e_energy = average_e_energy.astype(np.float64)
            average_i_energy = average_i_energy.astype(np.float64)

            average_e_energy *= average_e_energy_m.unit_SI * 6.241509e18
            average_i_energy *= average_i_energy_m.unit_SI * 6.241509e18
            if n_cells is not None:
                average_e_energy = np.ravel(average_e_energy)[:n_cells]
                average_i_energy = np.ravel(average_i_energy)[:n_cells]
            self.e_T_mean[i] = 2 / 3 * np.average(average_e_energy)
            self.i_T_mean[i] = 2 / 3 * np.average(average_i_energy)
            self.e_T_std_dist[i] = np.std((2 / 3) * average_e_energy)
            self.i_T_std_dist[i] = np.std((2 / 3) * average_i_energy)
            self.i_T_std_mean[i] = self.i_T_std_dist[i] / np.sqrt(
                average_i_energy.size)
            self.e_T_std_mean[i] = self.e_T_std_dist[i] / np.sqrt(
                average_e_energy.size)

    def _calc_coulomb_log(self, temp_e, temp_i):
        n_e_cgs = self.ELECTRON_DENSITY / 100 ** 3
        return 24 - np.log(np.sqrt(n_e_cgs) / temp_e)

    def calculate_theretical_values(self, coulomb_log=None):
        self.e_T_theory = np.empty_like(self.e_T_mean)
        self.i_T_theory = np.empty_like(self.i_T_mean)
        self.coulomb_log = np.empty(self.e_T_mean.size - 1)

        self.e_T_theory[0] = self.INIT_TEMP_ELECTRONS
        self.i_T_theory[0] = self.INIT_TEMP_IONS
        calc_log = False
        if coulomb_log is None:
            calc_log = True
        for ii in range(self.e_T_theory.size - 1):
            temp_e = self.e_T_theory[ii]
            temp_i = self.i_T_theory[ii]
            temp_e_joul = temp_e / 6.241509e18
            temp_i_joul = temp_i / 6.241509e18
            if calc_log:
                coulomb_log = self._calc_coulomb_log(temp_e, temp_i)
            self.coulomb_log[ii] = coulomb_log
            rate = ((2 / 3) * np.sqrt(2 / np.pi) * cs.elementary_charge ** 4
                    * self.ION_CHARGE ** 2
                    * np.sqrt(self.ION_MASS * self.ELECTRON_MASS)
                    * self.ION_DENSITY * coulomb_log
                    / (4 * np.pi * cs.epsilon_0 ** 2
                       * (self.ELECTRON_MASS * temp_e_joul
                          + self.ION_MASS * temp_i_joul) ** (3 / 2)))
            delta_temp = rate * (temp_e - temp_i) * self.dt
            self.e_T_theory[ii + 1] = temp_e - delta_temp
            self.i_T_theory[ii + 1] = temp_i + delta_temp

    def plot(self, to_file=False, file_name=None, smilei_sim=None):
        f, ax_dict = plt.subplot_mosaic([['main', 'std_dist'],
                                         ['zoom', 'zoom', ]], gridspec_kw={
            'height_ratios': [1, 1]}, figsize=(20, 22))
        times = np.array(self.series.iterations) * self.series.iterations[
            0].dt * self.series.iterations[0].time_unit_SI * 1e15
        palette = itertools.cycle(sns.color_palette("colorblind"))

        if smilei_sim is not None:
            smilei_sim.calculate_temperatures()
            color_ions = next(palette)
            color_electrons = next(palette)
            color_mean = next(palette)
            for ax in [ax_dict['main'], ax_dict['zoom']]:
                ax.plot(smilei_sim.times, smilei_sim.e_T_mean,
                        label=r'$T_e$ smilei $\pm (2\sigma)/\sqrt{N}$',
                        color=color_electrons)
                ax.fill_between(smilei_sim.times,
                                smilei_sim.e_T_mean
                                - 2 * smilei_sim.e_T_std_mean,
                                smilei_sim.e_T_mean
                                + 2 * smilei_sim.e_T_std_mean,
                                alpha=0.1, color=color_electrons)
                ax.plot(smilei_sim.times, smilei_sim.i_T_mean,
                        label=r'$T_i$ smilei $\pm (2\sigma)/\sqrt{N}$',
                        color=color_ions)
                ax.fill_between(smilei_sim.times,
                                smilei_sim.i_T_mean
                                - 2 * smilei_sim.i_T_std_mean,
                                smilei_sim.i_T_mean
                                + 2 * smilei_sim.i_T_std_mean,
                                alpha=0.1, color=color_ions)
                ax.plot(smilei_sim.times,
                        (smilei_sim.e_T_mean + smilei_sim.i_T_mean) / 2,
                        label=r'$(T_e + T_i)/2$ smilei', linestyle='dotted',
                        color=color_mean)
            ax = ax_dict['std_dist']
            ax.plot(smilei_sim.times, smilei_sim.e_T_mean,
                    label=r'$T_e$ smilei $\pm 2\sigma$', color=color_electrons)
            ax.fill_between(smilei_sim.times,
                            smilei_sim.e_T_mean - 2 * smilei_sim.e_T_std_dist,
                            smilei_sim.e_T_mean + 2 * smilei_sim.e_T_std_dist,
                            alpha=0.1, color=color_electrons)
            ax.plot(smilei_sim.times, smilei_sim.i_T_mean,
                    label=r'$T_i$ smilei $\pm 2\sigma$', color=color_ions)
            ax.fill_between(smilei_sim.times,
                            smilei_sim.i_T_mean - 2 * smilei_sim.i_T_std_dist,
                            smilei_sim.i_T_mean + 2 * smilei_sim.i_T_std_dist,
                            alpha=0.1, color=color_ions)
            ax.plot(smilei_sim.times,
                    (smilei_sim.e_T_mean + smilei_sim.i_T_mean) / 2,
                    label=r'$(T_e + T_i)/2$ smilei', linestyle='dotted',
                    color=color_mean)

        color_ions = next(palette)
        color_electrons = next(palette)
        color_mean = next(palette)
        for ax in [ax_dict['main'], ax_dict['zoom']]:
            ax.plot(times, self.e_T_mean,
                    label=r'$T_e$ PIConGPU $\pm (2\sigma)/\sqrt{N}$ ',
                    color=color_electrons)
            ax.fill_between(times, self.e_T_mean - 2 * self.e_T_std_mean,
                            self.e_T_mean + 2 * self.e_T_std_mean, alpha=0.1,
                            color=color_electrons)
            ax.plot(times, self.i_T_mean,
                    label=r'$T_i$ PIConGPU $\pm (2\sigma)/\sqrt{N}$',
                    color=color_ions)
            ax.fill_between(times, self.i_T_mean - 2 * self.i_T_std_mean,
                            self.i_T_mean + 2 * self.i_T_std_mean, alpha=0.1,
                            color=color_ions)
            ax.plot(times, self.e_T_theory, label=r'$T_e$ theory',
                    color='black', linestyle='dashed')
            ax.plot(times, self.i_T_theory, label=r'$T_i$ theory',
                    color='black', linestyle='solid')
            ax.plot(times, (self.e_T_mean + self.i_T_mean) / 2,
                    label=r'$(T_e + T_i)/2$ PIConGPU', linestyle='dotted',
                    color=color_mean)
        ax = ax_dict['std_dist']
        ax.plot(times, self.e_T_mean, label=r'$T_e$ PIConGPU $\pm 2\sigma$',
                color=color_electrons)
        ax.fill_between(times, self.e_T_mean - 2 * self.e_T_std_dist,
                        self.e_T_mean + 2 * self.e_T_std_dist, alpha=0.1,
                        color=color_electrons)
        ax.plot(times, self.i_T_mean, label=r'$T_i$ PIConGPU $\pm 2\sigma$',
                color=color_ions)
        ax.fill_between(times, self.i_T_mean - 2 * self.i_T_std_dist,
                        self.i_T_mean + 2 * self.i_T_std_dist, alpha=0.1,
                        color=color_ions)
        ax.plot(times, self.e_T_theory, label=r'$T_e$ theory', color='black',
                linestyle='dashed')
        ax.plot(times, self.i_T_theory, label=r'$T_i$ theory', color='black',
                linestyle='solid')
        ax.plot(times, (self.e_T_mean + self.i_T_mean) / 2,
                label=r'$(T_e + T_i)/2$ PIConGPU', linestyle='dotted',
                color=color_mean)

        for ax in [ax_dict['main'], ax_dict['zoom'], ax_dict['std_dist']]:
            ax.set_ylabel('T [eV]')
        ax_dict['zoom'].set_ylim(96, 98)

        for ax in ax_dict.values():
            ax.legend()
            ax.set_xlabel(r'$t [fs]$')

        with open(Path(self.sim_output_path) / 'input/cmakeFlagsSetup',
                  'r') as file:
            f.suptitle(
                file.readline() + '\n' + textwrap.fill(file.readline(), 80))
        f.tight_layout()
        plt.subplots_adjust(top=0.90)
        if to_file:
            if file_name is None:
                file_name = 'thermalization_plot_main.png'
            f.savefig(file_name)

    def plot_debug_values(self,
                          to_file=False,
                          file_name=None,
                          smilei_loader=None):
        f, axs = plt.subplots(3, 3, figsize=(15, 15))
        line_style = '-'
        times = (self.debug_values['ii']['iteration']
                 * self.dt * 1e15)
        for ii, collision_pair in enumerate(self.pairs):
            axs[ii][0].plot(times,
                            self.debug_values[collision_pair]['coulomb_log'],
                            label=fr'$\ln\Lambda$ PIConGPU {collision_pair}',
                            color='red',
                            ls=line_style,
                            alpha=0.75)
            axs[ii][1].plot(times,
                            self.debug_values[collision_pair]['s_param'],
                            label=fr'$s$ PIConGPU {collision_pair}',
                            color='red',
                            ls=line_style,
                            alpha=0.75)
            if smilei_loader is not None:
                times_smilei = smilei_loader.times[:-1]
                axs[ii][0].plot(times_smilei,
                                (smilei_loader.debug_values[collision_pair]
                                 ['coulomb_log']),
                                label=fr'$\ln\Lambda$ smilei {collision_pair}',
                                color='black',
                                ls=line_style,
                                alpha=0.75)
                axs[ii][1].plot(times_smilei,
                                (smilei_loader.debug_values[collision_pair]
                                 ['s_param']),
                                label=fr'$s$ smilei {collision_pair}',
                                color='black',
                                ls=line_style,
                                alpha=0.75)

            axs[ii][0].set_ylabel(r'$\ln\Lambda$')
            axs[ii][1].set_ylabel(r'$s$')
            axs[ii][2].set_ylabel(r'$\lambda_D [m]$')

        axs[0][0].plot(times, self.coulomb_log,
                       label='coulomb log theory ei')

        if self.average_debye_present:
            axs[0][2].plot(times,
                           self.debug_values['all']['debye_length'],
                           color='red',
                           label=r'$\lambda_D$ PIConGPU',
                           alpha=0.75)
            axs[1][2].plot(times,
                           self.debug_values['all']['debye_length'],
                           color='red',
                           label=r'$\lambda_D$ PIConGPU',
                           alpha=0.75)
            if smilei_loader is not None:
                times_smilei = smilei_loader.times[:-1]
                axs[2][2].plot(times_smilei,
                               (smilei_loader.debug_values['ee']
                                ['debye_length']),
                               label=r'$\lambda_D$ smilei ee',
                               color='black',
                               ls=line_style,
                               alpha=0.75)
                axs[0][2].plot(times_smilei,
                               (smilei_loader.debug_values['ee']
                                ['debye_length']),
                               label=r'$\lambda_D$ smilei ee',
                               color='black',
                               ls=line_style,
                               alpha=0.75)

        for ax in np.ravel(axs):
            ax.legend()
            ax.set_xlabel(r'$t [fs]$')

        with open(Path(self.sim_output_path) / 'input/cmakeFlagsSetup',
                  'r') as file:
            f.suptitle(file.readline() + '\n' + textwrap.fill(file.readline(),
                                                              80))
        plt.tight_layout()
        if to_file:
            if file_name is None:
                file_name = 'debug_values_plot.png'
            f.savefig(file_name)
