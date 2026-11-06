/* Copyright 2026-2026 Tapish Narwal
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

#include <pmacc/Environment.hpp>
#include <pmacc/lockstep.hpp>
#include <pmacc/memory/buffers/DeviceBuffer.hpp>
#include <pmacc/memory/buffers/HostBuffer.hpp>
#include <pmacc/test/PMaccFixture.hpp>

#include <limits>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <picongpu/defines.hpp>

using namespace picongpu;
using namespace pmacc;

//! Helper to setup the PMacc environment
static pmacc::test::PMaccFixture<simDim> pmaccFixture;

namespace
{
    //! number of checked quantities, one device thread per quantity
    constexpr uint32_t numValues = 4u;

    /** Units requested from the device
     *
     * The accessors are called with the same syntax as in the PIConGPU kernels, i.e. without
     * any `constexpr` locals. This only compiles if the accessors do not odr-use the host-only
     * `Sim` objects (`sim.pic`, `sim.si`, `sim.unit`, `sim.atomicUnit`) in device code.
     */
    HDINLINE double getSimUnit(uint32_t const idx)
    {
        if(idx == 0u)
            return static_cast<double>(sim.pic.getCellSize()[0]);
        if(idx == 1u)
            return static_cast<double>(sim.unit.energy());
        if(idx == 2u)
            return static_cast<double>(sim.pic.getEps0());
        return static_cast<double>(sim.unit.typicalNumParticlesPerMacroParticle());
    }
} // namespace

/** Check that the unit accessors are usable in device code and return the host values
 *
 * This is a compile test for the unit accessors in device code, therefore the device kernel must
 * use them exactly like the PIConGPU kernels do (no `constexpr` locals, no copy of the objects).
 */
TEST_CASE("unit::simUnits", "[sim unit accessors in device code]")
{
    ::pmacc::HostBuffer<double, 1u> resultHost(numValues);
    ::pmacc::DeviceBuffer<double, 1u> resultDevice(numValues);
    resultDevice.setValue(0.0);

    auto kernel = [] DEVICEONLY(auto const& worker, auto result)
    {
        auto forEach = lockstep::makeForEach<numValues>(worker);
        forEach([&](uint32_t const idx) { result[idx] = getSimUnit(idx); });
    };

    PMACC_LOCKSTEP_KERNEL(kernel).template config<numValues>(1)(resultDevice.getDataBox());
    resultHost.copyFrom(resultDevice);

    auto result = resultHost.getDataBox();
    for(uint32_t i = 0u; i < numValues; ++i)
    {
        double const expected = getSimUnit(i);
        INFO("index: " << i);
        REQUIRE(result[i] == Catch::Approx(expected).margin(std::numeric_limits<double>::epsilon()));
    }
}
