/* Copyright 2026 Alexander Debus
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

#include <catch2/catch_test_macros.hpp>
#include <picongpu/fields/incidentField/profiles/TWTSPulse.def>
#include <picongpu/param/precision.param>

using namespace picongpu;

//! Helper to setup the PMacc environment
static pmacc::test::PMaccFixture<simDim> pmaccFixture;

TEST_CASE("unit::TWTSPulseWindow", "[TWTSPulse window test]")
{
    using window = fields::incidentField::profiles::window;

    SECTION("zero length disables the window")
    {
        float_X const value = window::switchAt(0.0, 0.0, 0.0, 0.0);
        CHECK(std::isfinite(static_cast<double>(value)));
        CHECK(value == float_X(1.0));
    }

    SECTION("zero length disables the window for a non-empty range")
    {
        float_X const value = window::switchAt(1.0, 0.0, 2.0, 0.0);
        CHECK(std::isfinite(static_cast<double>(value)));
        CHECK(value == float_X(1.0));
    }

    SECTION("positive length keeps the Blackman-Nuttall window in range")
    {
        for(float_X const step : {float_X(0.0), float_X(25.0), float_X(50.0)})
        {
            float_X const value = window::switchAt(step, 0.0, 100.0, 50.0);
            CHECK(std::isfinite(static_cast<double>(value)));
            CHECK(value >= float_X(0.0));
            CHECK(value <= float_X(1.0));
        }
    }
}
