"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Richard Pausch, Brian Edward Marre
License: GPLv3+
"""

from picongpu import picmi

from unittest import TestCase
import json
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
        """bare int as well as list forms of zero or negative numbers are rejected"""
        for not_ngpus_dist in [0, -1, -7, [0], [1, 1, 0], [-1], [-1, 1, 1], [-7]]:
            with pytest.raises(ValueError, match=".*Number of gpus must be positive integer.*"):
                picmi.Cartesian3DGrid(
                    number_of_cells=[192, 2048, 12],
                    picongpu_n_gpus=not_ngpus_dist,
                    **self.COMMON_KWARGS,
                )

    def test_n_gpus_wrong_length(self):
        """empty, 2-element and 4-element sequences are rejected"""
        for invalid in [[], (), [2, 3], (2, 3), [1, 2, 3, 4], (1, 2, 3, 4)]:
            with pytest.raises(ValueError, match=".*could not be mapped to a 3-component list of integers.*"):
                picmi.Cartesian3DGrid(
                    number_of_cells=[192, 2048, 12],
                    picongpu_n_gpus=invalid,
                    **self.COMMON_KWARGS,
                )

    def test_n_gpus_rejects_bool(self):
        """True/False are rejected instead of silently meaning 1/0 GPUs"""
        for invalid in [True, False]:
            with pytest.raises(ValueError, match=".*not a bool.*"):
                picmi.Cartesian3DGrid(
                    number_of_cells=[192, 2048, 12],
                    picongpu_n_gpus=invalid,
                    **self.COMMON_KWARGS,
                )

    def test_n_gpus_integral_float(self):
        """integral floats are accepted via pydantic's lax coercion and documented behaviour"""
        for value, expected in [(4.0, (1, 4, 1)), ((1, 2.0, 3), (1, 2, 3)), ([1.0, 1.0, 1.0], (1, 1, 1))]:
            grid = picmi.Cartesian3DGrid(
                number_of_cells=[192, 2048, 12],
                picongpu_n_gpus=value,
                **self.COMMON_KWARGS,
            )
            assert grid.picongpu_n_gpus == expected, f"{value} should normalize to {expected}"

    def test_n_gpus_invalid(self):
        """non-int elements (incl. fractional floats) are rejected by the type check"""
        for invalid in ["abc", [1, "x", 2], [[4]], 2.5]:
            with pytest.raises(ValueError, match=".*Input should be a valid integer.*"):
                picmi.Cartesian3DGrid(
                    number_of_cells=[192, 2048, 12],
                    picongpu_n_gpus=invalid,
                    **self.COMMON_KWARGS,
                )

    def test_n_gpus_bare_int(self):
        """a bare int is normalized to (1, N, 1), i.e. parallelized in y"""
        grid = picmi.Cartesian3DGrid(
            number_of_cells=[192, 2048, 12],
            picongpu_n_gpus=4,
            **self.COMMON_KWARGS,
        )
        assert grid.picongpu_n_gpus == (1, 4, 1), "bare int should normalize to (1, N, 1)"

    def test_n_gpus_tuple(self):
        """a three-tuple is normalized to (Nx, Ny, Nz)"""
        grid = picmi.Cartesian3DGrid(
            number_of_cells=[192, 2048, 12],
            picongpu_n_gpus=(2, 3, 1),
            **self.COMMON_KWARGS,
        )
        assert grid.picongpu_n_gpus == (2, 3, 1), "three-tuple should be kept as (Nx, Ny, Nz)"

    def test_n_gpus_list_unchanged(self):
        """existing list behaviour is unchanged"""
        for n_gpus, expected in [([4], (1, 4, 1)), ([2, 3, 1], (2, 3, 1))]:
            grid = picmi.Cartesian3DGrid(
                number_of_cells=[192, 2048, 12],
                picongpu_n_gpus=n_gpus,
                **self.COMMON_KWARGS,
            )
            assert grid.picongpu_n_gpus == expected, f"{n_gpus} should normalize to {expected}"

    def test_n_gpus_none(self):
        """None means the simulation runs on a single GPU"""
        grid = picmi.Cartesian3DGrid(
            number_of_cells=[192, 2048, 12],
            picongpu_n_gpus=None,
            **self.COMMON_KWARGS,
        )
        assert grid.picongpu_n_gpus == (1, 1, 1), "None should normalize to (1, 1, 1)"

    def test_n_gpus_render_list_identical(self):
        """grids built from a list render byte-identically to the normalized forms"""

        def rendered(n_gpus, number_of_cells):
            grid = picmi.Cartesian3DGrid(
                number_of_cells=number_of_cells,
                picongpu_n_gpus=n_gpus,
                picongpu_super_cell_size=(8, 8, 4),
                **self.COMMON_KWARGS,
            )
            context = grid.get_as_pypicongpu().get_rendering_context()
            return json.dumps(context, sort_keys=True).encode()

        assert rendered([4], [192, 32, 12]) == rendered(4, [192, 32, 12]) == rendered((1, 4, 1), [192, 32, 12])
        assert rendered([2, 3, 1], [192, 960, 12]) == rendered((2, 3, 1), [192, 960, 12])
        assert rendered([1, 1, 1], [192, 2048, 12]) == rendered(None, [192, 2048, 12])

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
