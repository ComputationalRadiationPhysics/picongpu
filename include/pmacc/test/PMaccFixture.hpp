/* Copyright 2016-2021 Alexander Grund
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <pmacc/types.hpp>
#include <pmacc/Environment.hpp>
#include <pmacc/dimensions/DataSpace.hpp>


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
                const pmacc::DataSpace<T_dim> devices = pmacc::DataSpace<T_dim>::create(1);
                const pmacc::DataSpace<T_dim> periodic = pmacc::DataSpace<T_dim>::create(1);
                pmacc::Environment<T_dim>::get().initDevices(devices, periodic);
            }

            ~PMaccFixture()
            {
                /* finalize the PMacc context */
                pmacc::Environment<>::get().finalize();
            }
        };

        using PMaccFixture2D = PMaccFixture<2>;
        using PMaccFixture3D = PMaccFixture<3>;

    } // namespace test
} // namespace pmacc
