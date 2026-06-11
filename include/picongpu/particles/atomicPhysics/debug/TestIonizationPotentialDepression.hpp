/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/** @file unit tests for the Ionization Potential Depression(IPD) calculation
 *
 * test are activated by the global debug switch debug::ionizationPotentialDepression::RUN_UNIT_TESTS
 *  in atomicPhysics_Debug.param
 *
 * for updating the tests see the python [rate calculator tool](
 *  https://github.com/BrianMarre/picongpuAtomicPhysicsTools/tree/dev/RateCalculationReference)
 */

#pragma once

#include "picongpu/defines.hpp"
// need unit.param

#include "picongpu/particles/atomicPhysics/ConvertEnum.hpp"
#include "picongpu/particles/atomicPhysics/debug/TestRelativeError.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/RelativisticTemperatureFunctor.hpp"
#include "picongpu/particles/atomicPhysics/ionizationPotentialDepression/StewartPyattIPD.hpp"

namespace picongpu::particles::atomicPhysics::debug
{
    template<bool T_consoleOutput = true>
    struct TestIonizationPotentialDepression
    {
        //! @return true =^= test passed
        bool testStewartPyattIPD() const
        {
            // 1/m^3
            float_64 const electronDensity = 4.e28;

            // eV
            float_64 const temperatureTimesk_Boltzman = 1.e3;

            uint8_t const chargeState = 2u;
            float_64 const zStar = static_cast<float_64>(chargeState);
            float_64 const unit_length = static_cast<float_64>(sim.unit.length());

            // sqrt(UNIT_CHARGE^2 * UNIT_TIME^2 * / UNIT_LENGTH^3 / UNIT_MASS * UNIT_ENERGY
            //  * 1/( UNIT_CHARGE^2 * 1/m^3 * (m/UNIT_LENGTH)^3))
            //  = sqrt(1/UNIT_ENERGY * 1/UNIT_LENGTH * UNIT_ENERGY * UNIT_LENGTH^3) = UNIT_LENGTH
            float_64 const debyeLength = std::sqrt(
                sim.pic.getEps0<float_64>() * sim.pic.conv().eV2Joule<float_64>(temperatureTimesk_Boltzman)
                / (sim.pic.getElectronCharge<float_64>() * sim.pic.getElectronCharge<float_64>() * electronDensity
                   * (unit_length * unit_length * unit_length) * static_cast<float_64>(chargeState + 1)));

            // eV
            float_64 const correctIPDValue = 6.306390823271927;

            using StewartPyattIPD = particles::atomicPhysics::ionizationPotentialDepression::template StewartPyattIPD<
                particles::atomicPhysics::ionizationPotentialDepression::RelativisticTemperatureFunctor,
                false>;

            auto const superCellConstantInput = StewartPyattIPD::SuperCellConstantInput{
                static_cast<float_X>(temperatureTimesk_Boltzman),
                static_cast<float_X>(debyeLength),
                static_cast<float_X>(zStar),
                static_cast<float_X>(electronDensity * (sim.unit.length() * sim.unit.length() * sim.unit.length()))};

            // eV
            float_64 const ipd = static_cast<float_X>(StewartPyattIPD::ipd(superCellConstantInput, chargeState));

            return testRelativeError<T_consoleOutput>(
                correctIPDValue,
                ipd,
                "Stewart-Pyatt ionization potential depression",
                1e-5);
        }

        bool testAll()
        {
            bool passTotal = testStewartPyattIPD();

            if constexpr(T_consoleOutput)
            {
                std::cout << "Result:";
                if(passTotal)
                    std::cout << " * Success" << std::endl;
                else
                    std::cout << " x Fail" << std::endl;
            }
            std::cout << std::endl;
            return passTotal;
        }
    };
} // namespace picongpu::particles::atomicPhysics::debug
