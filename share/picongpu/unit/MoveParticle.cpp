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
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <picongpu/particles/Particles.hpp>

using ::picongpu::floatD_X;
using ::pmacc::localCellIdx;
using ::pmacc::multiMask;
using position = ::picongpu::position<>;
using ::picongpu::simDim;
using ::picongpu::particles::moveParticle;

/** A tiny stub of a particle implementing the interface expected by moveParticle()
 */
struct ParticleStub
{
    floatD_X pos = floatD_X::create(0.);
    int localCellIdxValue;
    int multiMaskValue;

    int& operator[](localCellIdx const& index)
    {
        return localCellIdxValue;
    }

    floatD_X& operator[](position const& index)
    {
        return pos;
    }

    int& operator[](multiMask const& index)
    {
        return multiMaskValue;
    }

    bool operator==(ParticleStub const& other)
    {
        return pos == other.pos and localCellIdxValue == other.localCellIdxValue
            and multiMaskValue == other.multiMaskValue;
    }
};

TEST_CASE("unit::moveParticle", "[moveParticle test]")
{
    ParticleStub particle;
    auto expectedParticle = particle;
    floatD_X newPos = particle.pos;
    std::array<int, 3> neighbouringLocalCellIdx{1,8,64};

    SECTION("does nothing for unchanged position")
    {
        REQUIRE(newPos == particle.pos);

        moveParticle(particle, newPos);

        CHECK(particle == expectedParticle);
    }

    auto i = GENERATE(range(0u, simDim));

    SECTION("moves trivially inside cell")
    {
        newPos[i] = .5;
        expectedParticle.pos = newPos;

        REQUIRE(newPos != particle.pos);

        moveParticle(particle, newPos);

        CHECK(particle == expectedParticle);
    }

    SECTION("moves outside of cell in positive direction")
    {
        newPos[i] = 1.5;
        expectedParticle.pos[i] = .5;
        expectedParticle.localCellIdxValue = neighbouringLocalCellIdx[i];

        moveParticle(particle, newPos);

        CHECK(particle == expectedParticle);
    }
}
