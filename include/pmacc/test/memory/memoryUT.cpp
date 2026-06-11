/*
 * SPDX-FileCopyrightText: Erik Zenker, Alexander Grund
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
