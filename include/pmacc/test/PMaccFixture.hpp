/*
 * SPDX-FileCopyrightText: Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <pmacc/Environment.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/types.hpp>

namespace pmacc
{
    namespace test
    {
        /** Fixture that initializes PMacc for a given dimensionality */
        template<unsigned T_dim>
        struct PMaccFixture
        {
            PMaccFixture()
            {
                pmacc::DataSpace<T_dim> const devices = pmacc::DataSpace<T_dim>::create(1);
                pmacc::DataSpace<T_dim> const periodic = pmacc::DataSpace<T_dim>::create(1);
                pmacc::Environment<T_dim>::get().initDevices(devices, periodic);
            }

            ~PMaccFixture()
            {
                /* finalize the PMacc context */
                pmacc::Environment<>::get().finalize();
            }

            void initGrids(
                DataSpace<T_dim> gridSizeGlobal,
                DataSpace<T_dim> gridSizeLocal,
                DataSpace<T_dim> gridOffset)
            {
                Environment<T_dim>::get().initGrids(gridSizeGlobal, gridSizeLocal, gridOffset);
            }
        };

        using PMaccFixture2D = PMaccFixture<2>;
        using PMaccFixture3D = PMaccFixture<3>;

    } // namespace test
} // namespace pmacc
