/*
 * SPDX-FileCopyrightText: PIConGPU contributors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/* Copyright 2025 Tapish Narwal
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <pmacc/boost_workaround.hpp>

#include <pmacc/test/PMaccFixture.hpp>

#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <picongpu/defines.hpp>
#include <picongpu/simulation/control/MovingWindow.hpp>

using namespace picongpu;

//! Helper to setup the PMacc environment
static pmacc::test::PMaccFixture<simDim> pmaccFixture;

struct MovingWindowTestFixture
{
    MovingWindow& window;

    MovingWindowTestFixture() : window(MovingWindow::getInstance())
    {
        window.resetForTesting();
    }

    ~MovingWindowTestFixture()
    {
        window.resetForTesting();
    }
};

TEST_CASE("unit::MovingWindow_origin", "[movingWindow test]")
{
    MovingWindowTestFixture fixture;
    auto& movingWindow = fixture.window;

    pmaccFixture.initGrids(
        pmacc::DataSpace<3>{500, 1000, 700}.shrink<simDim>(),
        pmacc::DataSpace<3>{50, 100, 70}.shrink<simDim>(),
        pmacc::DataSpace<3>{0, 0, 0}.shrink<simDim>());

    auto const globalWindowSizeInMoveDirection = 900;
    auto const gpuNumCellsInMoveDirection = 100;

    auto const c = static_cast<float_64>(sim.pic.getSpeedOfLight());
    auto const dt = static_cast<float_64>(sim.pic.getDt());
    // should both be 1 for correctness of the tests
    REQUIRE(c == 1.);
    REQUIRE(dt == 1.);
    // We now assume in tests that c and dt are 1

    auto const lightDistancePerStep = c * dt;

    auto const cellSize = sim.pic.getCellSize();
    constexpr auto moveDirection = MovingWindow::moveDirection;
    auto const cellSizeInMoveDirection = static_cast<float_64>(cellSize[moveDirection]);
    auto const cellsPerStep = lightDistancePerStep / cellSizeInMoveDirection;

    SECTION("Window disabled")
    {
        movingWindow.setEndSlideOnStep(0);
        REQUIRE_FALSE(movingWindow.isEnabled());
        uint32_t const testStep = 100;
        auto origin = movingWindow.getMovingWindowOriginPositionCells(testStep);
        for(unsigned i = 0; i < simDim; ++i)
        {
            CHECK(origin[i] == 0.0);
        }
    }

    SECTION("Window enabled")
    {
        float_64 const movePoint = GENERATE(0.4, 0.8);
        uint32_t const endStep = 10000;
        movingWindow.setMovePoint(movePoint);
        movingWindow.setEndSlideOnStep(endStep);
        REQUIRE(movingWindow.isEnabled());

        // Manually calculate when the window should start moving to verify pre-movement phase.
        auto const virtualParticleInitialStartCell = static_cast<uint32_t>(
            std::ceil(static_cast<float_64>(globalWindowSizeInMoveDirection) * (1.0 - movePoint)));

        // Calculate firstMoveStep again for the given movePoint.
        auto const wayToFirstMove
            = static_cast<float_64>(globalWindowSizeInMoveDirection - virtualParticleInitialStartCell)
              * cellSizeInMoveDirection;

        auto const firstMoveStep = static_cast<int32_t>(std::ceil(wayToFirstMove / lightDistancePerStep)) - 1;


        SECTION("Window enabled, but not yet moving")
        {
            // Move point is > 0 so we dont immediately start moving.
            REQUIRE(firstMoveStep > 0);

            // At the step just before movement starts, origin must be zero.
            auto origin = movingWindow.getMovingWindowOriginPositionCells(firstMoveStep - 1);
            for(unsigned i = 0; i < simDim; ++i)
            {
                CHECK(origin[i] == 0.0);
            }
        }

        SECTION("Window enabled, and moving")
        {
            uint32_t const testStep = static_cast<uint32_t>(firstMoveStep) + 100u;
            auto const origin = movingWindow.getMovingWindowOriginPositionCells(testStep);

            // Adding one here because the comoving particle starts from the position after the first move at
            // firstMoveStep
            auto const virtualParticleCellsPassed = cellsPerStep * static_cast<float_64>(testStep + 1);

            float_64 const expectedOriginY
                = virtualParticleCellsPassed + virtualParticleInitialStartCell - globalWindowSizeInMoveDirection;

            for(unsigned i = 0; i < simDim; ++i)
            {
                if(i == moveDirection)
                {
                    CHECK(origin[i] == Catch::Approx(expectedOriginY));
                }
                else
                {
                    CHECK(origin[i] == 0.0);
                }
            }
        }
    }

    SECTION("Window instantly moving")
    {
        float_64 const movePoint = 0.0;
        uint32_t const endStep = 10000;
        movingWindow.setMovePoint(movePoint);
        movingWindow.setEndSlideOnStep(endStep);
        REQUIRE(movingWindow.isEnabled());

        auto const origin0 = movingWindow.getMovingWindowOriginPositionCells(0);
        // The comoving particle origin is defined to start at from the location equal to the distance travelled from
        // the origin at c in one timestep at first move step3
        CHECK(origin0[moveDirection] == Catch::Approx(cellsPerStep));
    }

    SECTION("Alternative virtual particle definition from negative domain")
    {
        float_64 const movePoint = 0.341;
        uint32_t const endStep = 10000;
        movingWindow.setMovePoint(movePoint);
        movingWindow.setEndSlideOnStep(endStep);
        REQUIRE(movingWindow.isEnabled());

        auto const classicVirtualParticleInitialStartCell = static_cast<uint32_t>(
            std::ceil(static_cast<float_64>(globalWindowSizeInMoveDirection) * (1.0 - movePoint)));
        auto const classicWayToFirstMove
            = static_cast<float_64>(globalWindowSizeInMoveDirection - classicVirtualParticleInitialStartCell)
              * cellSizeInMoveDirection;
        auto const classicFirstMoveStep
            = static_cast<int32_t>(std::ceil(classicWayToFirstMove / lightDistancePerStep)) - 1;


        // This virtual particle starts from the negative domain and moves immidiately with c
        auto const virtualParticleInitialStartCell
            = -1 * math::floor(static_cast<float_64>(globalWindowSizeInMoveDirection) * (movePoint));
        auto const wayToFirstMove = static_cast<float_64>(-virtualParticleInitialStartCell) * cellSizeInMoveDirection;
        auto const firstMoveStep = static_cast<int32_t>(std::ceil(wayToFirstMove / lightDistancePerStep)) - 1;

        REQUIRE(firstMoveStep == classicFirstMoveStep);
    }

    SECTION("Window position evolution")
    {
        float_64 const movePoint = GENERATE(0, 0.341, 0.6);
        uint32_t const endStep = 10000;
        movingWindow.setMovePoint(movePoint);
        movingWindow.setEndSlideOnStep(endStep);
        REQUIRE(movingWindow.isEnabled());

        // This virtual particle starts from the negative domain and moves immidiately with c
        auto const virtualParticleInitialStartCell
            = -1 * math::floor(static_cast<float_64>(globalWindowSizeInMoveDirection) * (movePoint));
        auto const wayToFirstMove = static_cast<float_64>(-virtualParticleInitialStartCell) * cellSizeInMoveDirection;
        auto const firstMoveStep = static_cast<int32_t>(std::ceil(wayToFirstMove / lightDistancePerStep)) - 1;

        for(int step = 1; step < endStep; ++step)
        {
            auto const origin = movingWindow.getMovingWindowOriginPositionCells(step);

            auto const virtualParticlePosition
                = virtualParticleInitialStartCell + cellsPerStep * static_cast<float_64>(step + 1);

            CHECK(origin[moveDirection] == Catch::Approx(std::max(virtualParticlePosition, 0.0)));

            auto window = movingWindow.getWindow(step);
            auto windowRelativeParticlePos = origin[moveDirection]
                                             - (window.globalDimensions.offset[moveDirection]
                                                + movingWindow.getSlideCounter(step) * gpuNumCellsInMoveDirection);

            // If step == firstMoveStep, this check can and should fail for some move points. We check strict less than
            if(step < firstMoveStep)
            {
                CHECK(windowRelativeParticlePos == 0);
            }
            else
            {
                CHECK((windowRelativeParticlePos >= 0 && windowRelativeParticlePos < 1));
            }
        }
    }


    SECTION("Origin is clamped after endSlidingOnStep")
    {
        uint32_t const endStep = 500;
        movingWindow.setMovePoint(0.0);
        movingWindow.setEndSlideOnStep(endStep);
        REQUIRE(movingWindow.isEnabled());

        auto const originAtEnd = movingWindow.getMovingWindowOriginPositionCells(endStep);
        auto const originAfterEnd = movingWindow.getMovingWindowOriginPositionCells(endStep + 100);

        for(unsigned i = 0; i < simDim; ++i)
        {
            CHECK(originAfterEnd[i] == Catch::Approx(originAtEnd[i]));
        }
    }
}
