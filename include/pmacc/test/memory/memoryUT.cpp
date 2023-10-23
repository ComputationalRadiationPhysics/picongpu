/* Copyright 2015-2023 Erik Zenker, Alexander Grund
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

#include <pmacc/boost_workaround.hpp>

#include <pmacc/test/PMaccFixture.hpp>

// STL
#include <cstdint> /* uint8_t */
#include <iostream> /* cout, endl */
#include <string>

#include <catch2/catch_test_macros.hpp>

// BOOST
#include "pmacc/meta/Mp11.hpp"

// MPI
#include <mpi.h> /* MPI_Init, MPI_Finalize */

// PMacc
#include "pmacc/types.hpp" /* DIM1,DIM2,DIM3 */

#include <pmacc/Environment.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/memory/buffers/DeviceBuffer.hpp>
#include <pmacc/memory/buffers/HostBuffer.hpp>


namespace pmacc
{
    namespace test
    {
        namespace memory
        {
            /*******************************************************************************
             * Configuration
             ******************************************************************************/

            /**
             * Defines for which numbers of elements a
             * test should be verfied e.g. the size
             * of a host or device buffer.
             */
            template<typename T_Dim>
            std::vector<size_t> getElementsPerDim()
            {
                std::vector<size_t> nElements;
                std::vector<size_t> nElementsPerDim;

                // Elements total
                nElements.push_back(1);
                nElements.push_back(1 * 1000);
                nElements.push_back(1 * 1000 * 1000);
                nElements.push_back(1 * 1000 * 1000 * 10);

                // Elements per dimension
                for(size_t i = 0; i < nElements.size(); ++i)
                {
                    nElementsPerDim.push_back(
                        std::pow(nElements[i], static_cast<double>(1) / static_cast<double>(T_Dim::value)));
                }
                return nElementsPerDim;
            }

        } // namespace memory
    } // namespace test
} // namespace pmacc

/**
 * Definition of a list of dimension types. This
 * List is used to test memory operations in
 * each dimension setup automatically. For this
 * purpose pmacc::mp_for_each is used.
 */
using Dims = ::pmacc::mp_list<pmacc::mp_int<DIM1>, pmacc::mp_int<DIM2>, pmacc::mp_int<DIM3>>;

/*******************************************************************************
 * Test Suites
 ******************************************************************************/
using MyPMaccFixture = pmacc::test::PMaccFixture<TEST_DIM>;

static MyPMaccFixture fixture;

#include "HostBuffer/copyFrom.hpp"
#include "HostBuffer/reset.hpp"
#include "HostBuffer/setValue.hpp"
