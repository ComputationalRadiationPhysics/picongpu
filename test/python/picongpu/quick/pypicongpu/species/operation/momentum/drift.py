"""
This file is part of PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.species.operation.momentum import Drift

import unittest

import typeguard
import itertools
import math


@typeguard.typechecked
class TestDrift(unittest.TestCase):
    def test_passthru(self):
        """values are present in output"""
        drift = Drift()
        drift.gamma = 17893
        drift.direction_normalized = (
            0.0413868011324242,
            0.7989163623763952,
            0.6000164819563655,
        )

        context = drift.get_rendering_context()
        self.assertEqual(
            context,
            {
                "gamma": drift.gamma,
                "direction_normalized": {
                    "x": drift.direction_normalized[0],
                    "y": drift.direction_normalized[1],
                    "z": drift.direction_normalized[2],
                },
            },
        )

    def test_types(self):
        """typechecks are applied"""
        drift = Drift()

        for invalid in [[], None, 0, tuple([0]), [1, 2, 3], (0, 1)]:
            with self.assertRaises(typeguard.TypeCheckError):
                drift.direction_normalized = invalid

        for invalid in [[], None, "1", (1, 2, 3), tuple([0]), (0, 1)]:
            with self.assertRaises(typeguard.TypeCheckError):
                drift.gamma = invalid

    def test_invalid_gamma(self):
        """invalid values for gamma are rejected"""
        for invalid in [-1, -123.3, 0, 0.9999999999, math.inf, math.nan]:
            drift = Drift()
            drift.direction_normalized = (1, 0, 0)
            drift.gamma = invalid
            with self.assertRaises(ValueError):
                drift.check()

    def test_normalized_checked(self):
        """non-normalized direction is rejected"""
        non_normalized_directions = [
            (1, 2, 3),
            (0, 0, 0),
            (1, 1, 0),
            (-1, 0, 1),
        ]
        normalized_directions = [
            (1, 0, 0),
            (-1, 0, 0),
            (0.7071067811865475, 0.7071067811865475, 0.0),
            (0.5773502691896258, 0.5773502691896258, 0.5773502691896258),
            (0.0413868011324242, 0.7989163623763952, 0.6000164819563655),
        ]

        for components in non_normalized_directions:
            for permutation in itertools.permutations(components):
                drift = Drift()
                drift.gamma = 1
                drift.direction_normalized = tuple(permutation)

                with self.assertRaisesRegex(Exception, ".*[Nn]ormalize.*"):
                    drift.check()

        for components in normalized_directions:
            for permutation in itertools.permutations(components):
                drift = Drift()
                drift.gamma = 1
                drift.direction_normalized = tuple(permutation)

                # no error
                drift.check()

    def test_fill_from_invalid(self):
        """fill_... methods reject invalid inputs"""
        invalid_inputs = [
            (0, 0, 0),
            (math.nan, 1, 0),
            (math.inf, 1, 0),
            (-math.inf, 1, 0),
        ]

        for invalid in invalid_inputs:
            for permutation in itertools.permutations(invalid):
                vector3 = tuple(permutation)
                self.assertEqual(3, len(vector3))

                with self.assertRaises(ValueError):
                    Drift().fill_from_velocity(vector3)
                with self.assertRaises(ValueError):
                    Drift().fill_from_gamma_velocity(vector3)

    def test_fill_from_velocity(self):
        """computation based on velocity vector"""
        d = Drift()
        d.fill_from_velocity((1742, 1925, 1984))
        self.assertAlmostEqual(d.gamma, 1.00000000006)
        self.assertAlmostEqual(d.direction_normalized[0], 0.533132081381511)
        self.assertAlmostEqual(d.direction_normalized[1], 0.5891384940639545)
        self.assertAlmostEqual(d.direction_normalized[2], 0.607195206349551)

        d.fill_from_velocity((41782731.0, 61723581.0, 212931235.0))
        self.assertAlmostEqual(d.gamma, 1.5184434266)
        self.assertAlmostEqual(d.direction_normalized[0], 0.18520723575308878)
        self.assertAlmostEqual(d.direction_normalized[1], 0.2735975735475949)
        self.assertAlmostEqual(d.direction_normalized[2], 0.9438446098662472)

        d.fill_from_velocity((1, 0, 0))
        self.assertAlmostEqual(d.direction_normalized[0], 1)
        self.assertAlmostEqual(d.direction_normalized[1], 0)
        self.assertAlmostEqual(d.direction_normalized[2], 0)

    def test_faster_than_light(self):
        """filling from faster than light velocity fails"""
        d = Drift()
        faster_than_light_list = [
            (3e8, 0, 0),
            (3e8, -3e8, 0),
            (2e8, 2e8, 2e8),
            (299792458, 0, 0),
        ]
        for components in faster_than_light_list:
            for ftl in itertools.permutations(components):
                # on that note: the game with the same name as the iteration
                # var is awesome
                with self.assertRaisesRegex(ValueError, ".*[Ll]ight.*"):
                    d.fill_from_velocity(ftl)

        # works:
        d.fill_from_velocity((299792457.9, 0, 0))

    def test_fill_from_gamma_velocity(self):
        """computation based on velocity vector multiplied with gamma"""
        d = Drift()
        d.fill_from_gamma_velocity((29379221.65264335, 141390308.68517736, 265336.4756518417))
        self.assertAlmostEqual(d.gamma, 1.10997153564)
        self.assertAlmostEqual(d.direction_normalized[0], 0.20344224506631242)
        self.assertAlmostEqual(d.direction_normalized[1], 0.9790852245720864)
        self.assertAlmostEqual(d.direction_normalized[2], 0.001837375031333983)

        d.fill_from_gamma_velocity((359876252.1771755, 42747107.177341804, 43958755.5088825))
        self.assertAlmostEqual(d.gamma, 1.57570157478)
        self.assertAlmostEqual(d.direction_normalized[0], 0.9857936257001714)
        self.assertAlmostEqual(d.direction_normalized[1], 0.11709532239932068)
        self.assertAlmostEqual(d.direction_normalized[2], 0.1204143388517734)

        d.fill_from_gamma_velocity((1, 0, 0))
        self.assertAlmostEqual(d.gamma, 1)
        self.assertAlmostEqual(d.direction_normalized[0], 1)
        self.assertAlmostEqual(d.direction_normalized[1], 0)
        self.assertAlmostEqual(d.direction_normalized[2], 0)
