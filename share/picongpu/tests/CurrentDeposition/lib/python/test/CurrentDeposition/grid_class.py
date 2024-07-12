"""
This file is part of PIConGPU.

Copyright 2023 PIConGPU contributors
Authors: Hannes Wolf
License: GPLv3+

Implementation of class grid.
It computes the current density J and current deposition vector W of a charged
particle with a defined assignment function during one timestep on a grid.
The positions of the particle before and after the movement are needed.

"""


import numpy as np
from assignment_and_W_func import NGP, CIC, TSC, PQS, W


class grid:
    """It computes the current density J and current deposition vector W of a
    charged particle with a defined assignment function during one timestep on
    a grid.
    The positions of the particle before and after the movement are needed.

    """

    def __init__(self, order):
        """__init__ method.

        Parameters:
        order(int): order of the assignment function

        """

        self.order = order
        if order == 0:
            self.func = NGP
        elif order == 1:
            self.func = CIC
        elif order == 2:
            self.func = TSC
        elif order == 3:
            self.func = PQS
        else:
            print(f"Order {order} not implemented (order <= 3)")
        # Number of cells, that assignment function fits in.
        # The needed size of the grid is given by the size of the support
        # of the assignment function, which is order + 1.
        # To compensate for movement of the particle we need + 2 cells
        # (one in every direction; positive and negative displacement).
        # An additional +1 has to be considered for the recursive current
        # calculation with the [i-1]th cell.
        self.num_cells = order + 1 + 2 + 1

    def create_grid(self):
        """Creates a grid appropriate to the order of the assignment function.

        Returns:
        grid_x, grid_y, grid_z(np.array): mesh grids for the calculation of
        the respective components of the current deposition vector

        """

        # axis of the grid
        axis = np.arange(0, self.num_cells, 1, dtype=float)
        # creates 3D a grid for each component of a vectorfield
        grid_x, grid_y, grid_z = np.meshgrid(axis, axis, axis)

        return grid_x, grid_y, grid_z

    def particle_step(self, pos1, pos2, pos_off1, pos_off2):
        """Performs a particle step for a particle with the given PIConGPU
        positions on a minimal grid.

        Parameters:
        pos1/2(array - float): relative position of the particle in cell [0,1]
        pos_off1/2(array - int): cell, where the particle is located

        Returns:
        start_coord(array - float): start_coordinate on minimal grid
        end_coord(array - float): end_coordinate on minimal grid

        """

        self._set_sim_add = np.minimum(np.asarray(pos_off1), np.asarray(pos_off2))
        # Choose starting cell at the center of the grid
        start_cell = int((self.num_cells - 1) / 2)
        # Coordinates in the middle cell + position in the cell
        start_coord = start_cell + np.asarray(pos1)
        # Calculate end point coordinates - starting from starting cell,
        # add position and any cell crossings pos_off1 - pos_off2
        end_coord = start_cell + np.asarray(pos2) - pos_off1 + pos_off2

        return start_coord, end_coord

    def current_deposition_field(self, start_coord, end_coord, grid_x, grid_y, grid_z):
        """Calculates the current deposition vector W component-wise
        for z, y, x.

        Parameters:
        start_coord(array - float): start_coordinate on minimal grid
        end_coord(array - float): end_coordinate on minimal grid

        Returns:
        grid_x, grid_y, grid_z(np.array): mesh grids of the respective
        components of the current deposition vector

        """
        func = self.func

        for z in range(self.num_cells):
            for y in range(self.num_cells):
                for x in range(self.num_cells):
                    # Calculate the si components for the W calculation

                    # Old
                    s1 = func(start_coord[0] - x)
                    s2 = func(start_coord[1] - y)
                    s3 = func(start_coord[2] - z)
                    # new
                    s4 = func(end_coord[0] - x)
                    s5 = func(end_coord[1] - y)
                    s6 = func(end_coord[2] - z)

                    # calculate W and assign it to the grid node
                    grid_x[z][y][x] = W(s1, s2, s3, s4, s5, s6)
                    grid_y[z][y][x] = W(s2, s1, s3, s5, s4, s6)
                    grid_z[z][y][x] = W(s3, s2, s1, s6, s5, s4)

        return grid_x, grid_y, grid_z

    def current_density_field(self, W_grid_x, W_grid_y, W_grid_z, start_coord, end_coord, Q, params, shape):
        """Calculates the current density in the x, y, and z directions on the
        grid using the current deposition vector(field) W_grid_x,y,z.

        Parameters:
        W_grid_x/y/z(np.array) : current deposition vector(field)
        start_coord(np.array): start_coordinate on minimal grid
        end_coord(np.array): end_coordinate on minimal grid
        Q(float): charge of the particle
        params(dict): simulation parameters(e.g. cell dimensions, unit_factors)

        Returns:
        j_grid_x/y/z(np.array): current density vector(field)

        """

        # create null grid to write the current density
        j_grid_x = np.zeros(shape)
        j_grid_y = np.zeros(shape)
        j_grid_z = np.zeros(shape)

        # determine how many grid points are needed to be included
        # in calculation; aka half-width of assignment function
        half_width = (self.order + 1) * 1 / 2

        # determine start x
        start = np.min([start_coord - half_width, end_coord - half_width], axis=0).astype(int)

        # determine end x
        end = np.min([start_coord + half_width + 1, end_coord + half_width + 1], axis=0).astype(int)

        # additive to transform the minimal grid index of the current density
        # to the equivalent index of the simulation grid.
        # -1 to ensure that calculation stops at last W
        r = self._set_sim_add - start - 1

        combined_factors = (-Q) / params["dt"]
        x_factor = combined_factors * params["cell_width"]

        # calculate the norm of the assignment function
        H = params["cell_width"] * params["cell_height"] * params["cell_depth"]
        norm = (1 / H) ** self.order

        for k in range(start[2], end[2], 1):
            for j in range(start[1], end[1], 1):
                for i in range(start[0], end[0], 1):
                    j_grid_x[k + r[2]][j + r[1]][i + r[0]] = (
                        x_factor * norm * W_grid_x[k][j][i] + j_grid_x[k + r[2]][j + r[1]][i + r[0] - 1]
                    )

        y_factor = combined_factors * params["cell_height"]
        for k in range(start[2], start[2], 1):
            for j in range(start[1], end[1], 1):
                for i in range(start[0], end[0], 1):
                    j_grid_y[k + r[2]][j + r[1]][i + r[0]] = (
                        y_factor * norm * W_grid_y[k][j][i] + j_grid_y[k + r[2]][j + r[1] - 1][i + r[0]]
                    )

        z_factor = combined_factors * params["cell_depth"]
        for k in range(start[2], start[2], 1):
            for j in range(start[1], end[1], 1):
                for i in range(start[0], end[0], 1):
                    j_grid_z[k + r[2]][j + r[1]][i + r[0]] = (
                        z_factor * norm * W_grid_z[k][j][i] + j_grid_y[k + r[2] - 1][j + r[1]][i + r[0]]
                    )

        return j_grid_x, j_grid_y, j_grid_z
