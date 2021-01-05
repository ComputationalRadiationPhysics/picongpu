"""
This file is part of the PIConGPU.

Copyright 2017-2021 PIConGPU contributors
Authors: Axel Huebl, Finn-Ole Carstens
License: GPLv3+
"""
from .base_reader import DataReader

import numpy as np
import os


class TransitionRadiationData(DataReader):
    """
    Data Reader for Transition Radiation Plugin.
    """

    def __init__(self, run_directory):
        """
        Parameters
        ----------
        simulation_directory : string
            path to the run directory of PIConGPU
            (the path before ``simOutput/``)
        """
        super().__init__(run_directory)

        self.data_file_prefix = "_transRad_"
        self.data_file_suffix = ".dat"
        self.data_file_folder = "transRad/"
        self.data = None
        self.omegas = None
        self.thetas = None
        self.phis = None

    def get_data_path(self, species=None, iteration=None):
        """
        Return the path to the underlying data file.

        Parameters
        ----------
        species : string
            short name of the particle species, e.g. "e" for electrons
            (defined in ``speciesDefinition.param``)
        iteration: (unsigned) int or list of int
            The iteration at which to read the data.

        Returns
        -------
        A string with a path.
        """
        if species is None:
            raise ValueError("The species parameter can not be None!")

        sim_output_dir = os.path.join(self.run_directory, "simOutput")
        if not os.path.isdir(sim_output_dir):
            raise IOError("The simOutput/ directory does not exist inside "
                          "path:\n  {}\n"
                          "Did you set the proper path to the run directory?\n"
                          "Did the simulation already run?"
                          .format(self.run_directory))

        if iteration is None:
            return sim_output_dir
        else:
            data_file_path = os.path.join(
                sim_output_dir,
                self.data_file_folder + species + self.data_file_prefix +
                str(iteration) + self.data_file_suffix
            )
            if not os.path.isfile(data_file_path):
                raise IOError("The file {} does not exist.\n"
                              "Did the simulation already run?"
                              .format(data_file_path))
            return data_file_path

    def _get_for_iteration(self, iteration=None, **kwargs):
        """
        Returns data for transition radiation visualization as specified with
        "type" for transition_radiation_visualizer.py

        Parameters
        ----------
        iteration : (unsigned) int
            The iteration at which to read the data.

        Returns
        -------
        The data of the transition radiation plugin as a two dimensional array.
        """
        if kwargs["species"] is None:
            raise ValueError("The species parameter can not be None!")

        if iteration is None:
            raise ValueError("The iteration can't be None!")

        # Do loading once and store them in ram
        if self.data is None:
            data_file_path = self.get_data_path(species=kwargs["species"],
                                                iteration=iteration)

            # Actual loading of data
            self.data = np.loadtxt(data_file_path)

            # Read values to automatically create theta, phi and omega arrays
            f = open(data_file_path)
            parameters = f.readlines()[0].split("\t")
            f.close()

            # Create discretized arrays or angles and frequency as they are
            # discretized for the
            # calculation in PIConGPU. This is necessary for the labels for
            # the axes.
            indexOffset = 2
            if parameters[1] == "log":
                self.omegas = np.logspace(
                    np.log10(float(parameters[1 + indexOffset])),
                    np.log10(float(parameters[2 + indexOffset])),
                    int(parameters[0 + indexOffset]))
            elif parameters[1] == "lin":
                self.omegas = np.linspace(float(parameters[1 + indexOffset]),
                                          float(parameters[2 + indexOffset]),
                                          int(parameters[0 + indexOffset]))
            elif parameters[1] == "list":
                self.omegas = np.loadtxt(
                    self.get_data_path() + parameters[0 + indexOffset])
                indexOffset += 2
            self.phis = np.linspace(float(parameters[4 + indexOffset]),
                                    float(parameters[5 + indexOffset]),
                                    int(parameters[3 + indexOffset]))
            self.thetas = np.linspace(float(parameters[7 + indexOffset]),
                                      float(parameters[8 + indexOffset]),
                                      int(parameters[6 + indexOffset]))

        return self.get_data(iteration=iteration, **kwargs)

    def get_data(self, iteration=None, **kwargs):
        """
        Calculates data as specified with "type" for the plot from
        transition_radiation_visualizer.py.

        Parameters
        ----------
        iteration : (unsigned) int
            The iteration at which to read the data.
        kwargs: dictionary with further keyword arguments, valid are:
            species: string
                short name of the particle species, e.g. 'e' for electrons
                (defined in ``speciesDefinition.param``)
            iteration: int
                number of the iteration
            time: float
                simulation time.
                Only one of 'iteration' or 'time' should be passed!
            type: string
                name of figure type. valid figure types are:
                    'spectrum' - (default) plots transition radiation spectrum
                        at angles theta and phi over the frequency omega
                    'sliceovertheta' - shows angular distribution of
                    transition radiation
                        at a fixed angle phi and frequency omega
                    'sliceoverphi' - shows angular distribution of
                    transition radiation
                        at a fixed angle theta and frequency omega
                    'heatmap' - shows angular distribution as heatmap over
                    both observation angles
            phi: int
                index of polar angle for a fixed value
            theta: int
                index of azimuth angle for a fixed value
            omega: int
                index of frequency for a fixed value, pointless in a spectrum

        Returns
        -------
        A tuple of the x and y values for the plot as one dimensional arrays
        for normal figures
        (not heatmaps) and as colormeshes for heatmaps.
        """
        if iteration is None:
            raise ValueError("Can't return data for an unknown iteration!")
        if self.data is None:
            self.data = self._get_for_iteration(kwargs["species"], iteration,
                                                **kwargs)

        # Specify plot type
        type = kwargs["type"]

        # Load fixed values for observation angles from arguments, if given
        theta = kwargs["theta"]
        phi = kwargs["phi"]
        omega = kwargs["omega"]

        # Cast parameters to int and check for legitimacy
        if theta is not None:
            theta = int(theta)
            if theta < 0 or theta >= len(self.thetas):
                raise ValueError("Invalid index for Theta!")
        if phi is not None:
            phi = int(phi)
            if phi < 0 or phi >= len(self.phis):
                raise ValueError("Invalid index for Phi!")
        if omega is not None:
            omega = int(omega)
            if omega < 0 or omega >= len(self.omegas):
                raise ValueError("Invalid index for Omega!")

        if type == "spectrum":
            # find phi and theta with maximum intensity if they are not
            # given as parameters
            if theta is None and phi is None:
                maxIndex = np.argmax(self.data[:, :], 0)
                theta = int(np.floor(maxIndex[0] / len(self.phis)))
                phi = maxIndex[0] % len(self.phis)
            elif theta is None and phi is not None:
                theta = np.argmax(self.data[phi::len(self.phis), :], 0)[0]
            elif theta is not None and phi is None:
                phi = np.argmax(self.data[
                                theta * len(self.phis):(theta + 1) * len(
                                    self.phis):, :], 0)[0]

            print("Spectrum is plotted at phi={:.2e} and theta={:.2e}".format(
                self.phis[phi], self.thetas[theta]))
            return self.omegas, self.data[theta * len(self.phis) + phi, :]
        elif type == "sliceovertheta":
            # find phi and omega with maximum intensity if they are not
            # given as parameters
            if omega is None and phi is None:
                maxIndex = np.argmax(self.data[:, :], 0)
                phi = maxIndex[0] % len(self.phis)
                omega = 0
            if omega is not None and phi is None:
                maxIndex = np.argmax(self.data[:, omega])
                phi = maxIndex % len(self.phis)
            if omega is None and phi is not None:
                omega = 0

            print("Angular intensity distribution is sliced at phi={:.2e} "
                  "with omega={:.2e}.".format(self.phis[phi],
                                              self.omegas[omega]))
            return self.thetas, self.data[phi::len(self.phis), omega]
        elif type == "sliceoverphi":
            # find theta and omega with maximum intensity if they are not
            # given as parameters
            if theta is None and omega is None:
                maxIndex = np.argmax(self.data[:, :], 0)
                theta = int(np.floor(maxIndex[0] / len(self.phis)))
                omega = 0
            if theta is not None and omega is None:
                omega = 0
            if theta is None and omega is not None:
                maxIndex = np.argmax(self.data[:, omega])
                theta = int(np.floor(maxIndex / len(self.phis)))

            print("Angular intensity distribution is sliced at theta={:.2e} "
                  "with omega={:.2e}.".format(self.thetas[theta],
                                              self.omegas[omega]))
            return self.phis, self.data[
                              theta * len(self.phis):(theta + 1) * len(
                                  self.phis), omega]
        elif type == "heatmap":
            # find omega with maximum intensity if it is not given as parameter
            if omega is None:
                omega = 0
            # meshgrids for visualization
            theta_mesh, phi_mesh = np.meshgrid(self.thetas, self.phis)

            print("Heatmap is for omega={:.2e}.".format(self.omegas[omega]))
            return theta_mesh, phi_mesh, self.data[::, omega].reshape(
                (len(self.thetas), len(self.phis))).transpose()
        else:
            raise ValueError("Illegal type of figure!")

    def get_iterations(self, species):
        """
        Return an array of iterations with available data.

        Parameters
        ----------
        species : string
            short name of the particle species, e.g. "e" for electrons
            (defined in ``speciesDefinition.param``)

        Returns
        -------
        An array with unsigned integers.
        """
        ctr_path = self.get_data_path(species)

        ctr_files = [f for f in os.listdir(ctr_path) if f.endswith(".dat")]
        iters = [int(f.split("_")[2].split(".")[0]) for f in ctr_files]
        return np.array(sorted(iters))
