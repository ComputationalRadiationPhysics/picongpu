"""
This file is part of the PIConGPU.

Copyright 2017-2020 PIConGPU contributors
Authors: Axel Huebl, Finn-Ole Carstens, Juncheng E, Sergei Bastrakov
License: GPLv3+
"""
from .base_reader import DataReader

import numpy as np
import os


class XrayDiffractionData(DataReader):
    """
    Data Reader for X-ray diffraction Plugin.
    """

    def __init__(self, run_directory):
        """
        Parameters
        ----------
        run_directory : string
            path to the run directory of PIConGPU
            (the path before ``simOutput/``)
        """
        super().__init__(run_directory)

        self.data_file_prefix = "_intensity_"
        self.data_file_suffix = ".dat"
        self.data_file_folder = "xrayDiffraction/"
        self.data = None

        # TODO: update
        #self.filename = filename
        #self.nq = self.getNq(filename)
        #self.qx = np.zeros(self.nq)
        #self.qy = np.zeros(self.nq)
        #self.qz = np.zeros(self.nq)
        #self.intensity = np.zeros(self.nq)
        #with open(filename) as fin:
        #    # First 2 lines are headers, skip
        #    fin.readline()
        #    fin.readline()
        #    for i, line in enumerate(fin):
        #        sline = line.strip().split()
        #        self.qx[i] = float(sline[0])
        #        self.qy[i] = float(sline[1])
        #        self.qz[i] = float(sline[2])
        #        self.intensity[i] = float(sline[3])


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
        Returns data for X-ray diffraction visualization
        for xray_diffraction_visualizer.py

        Parameters
        ----------
        iteration : (unsigned) int
            The iteration at which to read the data.

        Returns
        -------
        The data of the X-ray diffraction plugin as a two dimensional array.
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

            # Read values to automatically create qx, qy, and qz arrays
            f = open(data_file_path)
            parameters = f.readlines()[0].split()
            f.close()

            # Create discretized arrays or angles and frequency as they are
            # discretized for the calculation in PIConGPU.
            # This is necessary for the labels for the axes.
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

        # Load fixed values for scattering vector components from arguments, if given
        qx = kwargs["qx"]
        qy = kwargs["qy"]
        qz = kwargs["qz"]

        # Cast parameters to int and check for legitimacy
        if qx is not None:
            qx = int(qx)
            if qx < 0 or qx >= len(self.qx_values):
                raise ValueError("Invalid index for qx!")
        if qy is not None:
            qy = int(qy)
            if qy < 0 or qy >= len(self.qy_values):
                raise ValueError("Invalid index for qy!")
        if qz is not None:
            qz = int(qz)
            if qz < 0 or qz >= len(self.qz_values):
                raise ValueError("Invalid index for qz!")


        elif type == "heatmap":
            # find omega with maximum intensity if it is not given as parameter
            if omega is None:
                omega = 0
            # meshgrids for visualization
            qx_mesh, qy_mesh = np.meshgrid(self.qx_values, self.qy_values)

            print("Heatmap is for omega={:.2e}.".format(self.omegas[omega]))
            return theta_mesh, phi_mesh, self.data[::, omega].reshape(
                (len(self.thetas), len(self.phis))).transpose()


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


    # get the number of q
    def getNq(self,filename):
        with open(filename) as fin:
             nq = int(fin.readline())
        fin.close()
        return nq
