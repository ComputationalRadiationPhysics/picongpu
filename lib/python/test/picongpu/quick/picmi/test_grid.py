"""
# SPDX-FileCopyrightText: Richard Pausch, Brian Edward Marre
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

from picongpu import picmi

from unittest import TestCase
import typeguard
import pytest


class TestCartesian3DGrid(TestCase):
    COMMON_KWARGS = dict(
        lower_bound=[0, 0, 0],
        upper_bound=[3.40992e-5, 9.07264e-5, 2.1312e-6],
        lower_boundary_conditions=["open", "open", "periodic"],
        upper_boundary_conditions=["open", "open", "periodic"],
    )

    def setUp(self):
        """default setup"""
        self.grid = picmi.Cartesian3DGrid(
            number_of_cells=[192, 2048, 12],
            **self.COMMON_KWARGS,
        )

    def test_basic(self):
        """simple translation"""
        grid = self.grid
        g = grid.get_as_pypicongpu()
        assert [] != g.get_rendering_context(), "grid rendering context should not be empty"

    def test_typo_ngpus(self):
        """test common typo picongpu_ngpus instead of picongpu_n_gpus"""
        with pytest.raises(TypeError, match=".*Unexpected.*ngpus.*"):
            picmi.Cartesian3DGrid(
                number_of_cells=[192, 2048, 12],
                # common typo ngpus instead of picongpu_n_gpus
                picongpu_ngpus=None,
                **self.COMMON_KWARGS,
            )

    def test_n_gpus_type(self):
        """test wrong input type for picongpu_n_gpus"""
        for i, not_ngpus_type in enumerate([1, 1.0, 1.2, "abc", tuple([1])]):
            with pytest.raises(
                typeguard.TypeCheckError,
                match='.*argument "picongpu_n_gpus"(.*) did not match any element.*',
            ):
                picmi.Cartesian3DGrid(
                    number_of_cells=[192, 2048, 12],
                    picongpu_n_gpus=not_ngpus_type,
                    **self.COMMON_KWARGS,
                )

    def test_n_gpus_asserts(self):
        """test too many GPUs for grid"""
        for not_ngpus_dist in [[1, 1, 2], [5, 1, 1], [1, 512, 1]]:
            grid = picmi.Cartesian3DGrid(
                number_of_cells=[192, 2048, 12],
                picongpu_n_gpus=not_ngpus_dist,
                **self.COMMON_KWARGS,
            )
            with pytest.raises(Exception, match=".*GPU- and/or super-cell-distribution.*"):
                grid.get_as_pypicongpu()

    def test_n_gpus_wrong_numbers(self):
        """test negativ numbers or zero as number of gpus"""
        for not_ngpus_dist in [[0], [1, 1, 0], [-1], [-1, 1, 1], [-7]]:
            with pytest.raises(Exception, match=".*Number of gpus must be positive integer.*"):
                picmi.Cartesian3DGrid(
                    number_of_cells=[192, 2048, 12],
                    picongpu_n_gpus=not_ngpus_dist,
                    **self.COMMON_KWARGS,
                )

    def test_supercell(self):
        """test explicitly setting the super cell size default value"""
        grid = picmi.Cartesian3DGrid(
            number_of_cells=[192, 2048, 12],
            picongpu_n_gpus=[1, 1, 1],
            picongpu_super_cell_size=(8, 8, 4),
            **self.COMMON_KWARGS,
        )
        g = grid.get_as_pypicongpu()
        assert g.super_cell_size == (8, 8, 4), "supercell should be [8,8,4]"

    def test_super_cell_mismatch_no_dist(self):
        grid = picmi.Cartesian3DGrid(
            number_of_cells=[192, 2048, 12],
            picongpu_super_cell_size=(7, 8, 4),
            **self.COMMON_KWARGS,
        )
        with pytest.raises(Exception, match=".*GPU- and/or super-cell-distribution.*"):
            grid.get_as_pypicongpu()

    def test_super_cell_mismatch_with_dist(self):
        grid = picmi.Cartesian3DGrid(
            number_of_cells=[192, 2048, 12],
            picongpu_n_gpus=[2, 1, 1],
            picongpu_super_cell_size=(7, 8, 4),
            picongpu_grid_dist=([12, 180], [2048], [12]),
            **self.COMMON_KWARGS,
        )
        with pytest.raises(Exception, match=".*grid distribution in x dimension must be multiple.*"):
            grid.get_as_pypicongpu()

    def test_super_cell_size_zero(self):
        grid = picmi.Cartesian3DGrid(
            number_of_cells=[192, 2048, 12],
            picongpu_super_cell_size=(0, 8, 4),
            **self.COMMON_KWARGS,
        )
        with pytest.raises(Exception, match=".*super cell size must be an integer greater than 1.*"):
            grid.get_as_pypicongpu()

    def test_super_cell_size_negative(self):
        grid = picmi.Cartesian3DGrid(
            number_of_cells=[192, 2048, 12],
            picongpu_super_cell_size=(8, -8, 4),
            **self.COMMON_KWARGS,
        )
        with pytest.raises(Exception, match=".*super cell size must be an integer greater than 1.*"):
            grid.get_as_pypicongpu()

    def test_grid_dist_values_lt_one(self):
        grid = picmi.Cartesian3DGrid(
            number_of_cells=[192, 2048, 12],
            picongpu_n_gpus=[1, 1, 1],
            picongpu_grid_dist=([192], [2048], [0]),
            **self.COMMON_KWARGS,
        )
        with pytest.raises(Exception, match=".*All values in grid distribution must be greater than 0.*"):
            grid.get_as_pypicongpu()

    def test_grid_dist_sum_mismatch(self):
        grid = picmi.Cartesian3DGrid(
            number_of_cells=[192, 2048, 12],
            picongpu_n_gpus=[2, 1, 1],
            picongpu_grid_dist=([100, 64], [2048], [12]),
            **self.COMMON_KWARGS,
        )
        with pytest.raises(Exception, match=".*sum of grid distribution.*must match number of cells.*"):
            grid.get_as_pypicongpu()

    def test_grid_dist_length_mismatch(self):
        grid = picmi.Cartesian3DGrid(
            number_of_cells=[192, 2048, 12],
            picongpu_n_gpus=[1, 1, 1],
            # length 2 in x but n_gpus=1
            picongpu_grid_dist=([96, 96], [2048], [12]),  # length 2 in x but n_gpus=1
            **self.COMMON_KWARGS,
        )
        with pytest.raises(Exception, match=".*number of grid distributions.*must match number of gpus.*"):
            grid.get_as_pypicongpu()

    def test_grid_dist_correct(self):
        grid = picmi.Cartesian3DGrid(
            number_of_cells=[192, 2048, 12],
            picongpu_n_gpus=[2, 1, 1],
            picongpu_super_cell_size=(8, 8, 4),
            picongpu_grid_dist=([96, 96], [2048], [12]),
            **self.COMMON_KWARGS,
        )
        g = grid.get_as_pypicongpu()
        assert g.grid_dist == ([96, 96], [2048], [12]), "grid_dist should be [96,96], [2048], [12]"
