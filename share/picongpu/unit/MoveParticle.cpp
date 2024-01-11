/* Copyright 2023 Julian Lenz
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <pmacc/boost_workaround.hpp>

#include <picongpu/simulation_defines.hpp>

#include <catch2/catch_test_macros.hpp>

using ::picongpu::floatD_X;
using ::picongpu::lcellId_t;
using position = ::picongpu::position<::picongpu::position_pic>;

/** A tiny stub of a particle implementing the interface expected by moveParticle()
 */
struct ParticleStub
{
    floatD_X pos;
    lcellId_t localCellIdx;

    lcellId_t& operator[](lcellId_t const& index)
    {
        return localCellIdx;
    }

    floatD_X& operator[](position const& index)
    {
        return pos;
    }
};

TEST_CASE("unit::moveParticle", "[moveParticle test]")
{
}
