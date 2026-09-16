"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Richard Pausch, Brian Edward Marre
License: GPLv3+
"""

from picongpu import picmi

from unittest import TestCase
import pytest
import pydantic


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
            with pytest.raises(Exception, match=".*picongpu_n_gpus.*|.*Number of gpus must be positive integer.*"):
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
        with pytest.raises(Exception, match=".*super cell size must be a positive integer.*"):
            grid.get_as_pypicongpu()

    def test_super_cell_size_negative(self):
        grid = picmi.Cartesian3DGrid(
            number_of_cells=[192, 2048, 12],
            picongpu_super_cell_size=(8, -8, 4),
            **self.COMMON_KWARGS,
        )
        with pytest.raises(Exception, match=".*super cell size must be a positive integer.*"):
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

    def test_zero_number_of_cells_rejected(self):
        """a degenerate box (zero cells in any dimension) is rejected at construction"""
        for bad_cells in [[0, 2048, 12], [192, 0, 12], [192, 2048, 0]]:
            with pytest.raises(pydantic.ValidationError, match=".*number_of_cells.*must be a positive integer.*"):
                picmi.Cartesian3DGrid(
                    number_of_cells=bad_cells,
                    **self.COMMON_KWARGS,
                )

    def test_negative_number_of_cells_rejected(self):
        """a negative number of cells is rejected at construction"""
        with pytest.raises(pydantic.ValidationError, match=".*number_of_cells.*must be a positive integer.*"):
            picmi.Cartesian3DGrid(
                number_of_cells=[192, -2048, 12],
                **self.COMMON_KWARGS,
            )

    def test_upper_bound_le_lower_bound_rejected(self):
        """an empty (upper == lower) or inverted (upper < lower) extent is rejected at construction"""
        cases = [
            # empty extent in x: upper == lower (0.0)
            dict(lower_bound=[0, 0, 0], upper_bound=[0.0, 9.07264e-5, 2.1312e-6]),
            # inverted extent in x: upper < lower
            dict(lower_bound=[1.0, 0.0, 0.0], upper_bound=[0.5, 9.07264e-5, 2.1312e-6]),
        ]
        for kwargs in cases:
            with pytest.raises(pydantic.ValidationError, match=".*upper_bound.*must be greater than lower_bound.*"):
                picmi.Cartesian3DGrid(
                    number_of_cells=[192, 2048, 12],
                    lower_boundary_conditions=["open", "open", "periodic"],
                    upper_boundary_conditions=["open", "open", "periodic"],
                    **kwargs,
                )

    def test_valid_grid_constructs(self):
        """a valid grid with positive cells and upper>lower still constructs and renders"""
        grid = picmi.Cartesian3DGrid(
            number_of_cells=[192, 2048, 12],
            **self.COMMON_KWARGS,
        )
        assert grid.picongpu_cell_size[0] == self.COMMON_KWARGS["upper_bound"][0] / 192
        assert grid.get_as_pypicongpu().get_rendering_context() != []

    def test_super_cell_message_positive_integer(self):
        """the super-cell error message reads 'positive integer' (matches the < 1 check)"""
        grid = picmi.Cartesian3DGrid(
            number_of_cells=[192, 2048, 12],
            picongpu_super_cell_size=(0, 8, 4),
            **self.COMMON_KWARGS,
        )
        with pytest.raises(Exception, match=".*super cell size must be a positive integer.*"):
            grid.get_as_pypicongpu()
