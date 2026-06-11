/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/ConvertEnum.hpp"
#include "picongpu/particles/atomicPhysics/debug/param.hpp"
#include "picongpu/particles/atomicPhysics/rateCalculation/BinomialCoefficient.hpp"

#include <pmacc/algorithms/math.hpp>

#include <cstdint>

namespace picongpu::particles::atomicPhysics::rateCalculation
{
    /** number of physical different atomic configurations for a given configNumber
     *
     * @param configNumber configNumber of atomic state, unitless
     *
     * @return degeneracy, number of physical configurations, unitless
     */
    template<typename T_ConfigNumber, typename T_ConfigNumberStorage>
    HDINLINE float_64 multiplicityConfigNumber(T_ConfigNumberStorage const configNumber)
    {
        using LevelVector = pmacc::math::Vector<uint8_t, T_ConfigNumber::numberLevels>; // unitless

        // check for overflows
        PMACC_CASSERT_MSG(Too_high_n_max_in_Multiplicity, T_ConfigNumber::numberLevels < 12u);

        LevelVector const levelVector = T_ConfigNumber::getLevelVector(configNumber); // unitless

        uint64_t result = 1.;
        for(uint8_t i = 0u; i < T_ConfigNumber::numberLevels; i++)
        {
            //  number configurations over number electrons
            result *= picongpu::particles::atomicPhysics::rateCalculation::binomialCoefficient(
                // 2*n^2, number of atomic configurations in (i+1)-th shell
                u8(2u) * pmacc::math::cPow(i + u8(1u), u8(2u)),
                // k, number electrons in the (i+1)-th shell
                levelVector[i]);
        }

        return result; // unitless
    }

    /** combinatorial multiplicity of a bound-free transition
     *
     * @tparam T_LevelVector vector of shell occupation numbers, ascending in principal quantum number
     *
     * @attention must fulfil lowerState[i] >= removedElectrons[i], not checked outside debug compile
     */
    template<typename T_LevelVector>
    HDINLINE float_64 multiplicityBoundFreeTransition(T_LevelVector lowerState, T_LevelVector removedElectrons)
    {
        float_64 combinatorialFactor = 1.;
        for(uint8_t i = 0u; i < T_LevelVector::dim; i++)
        {
            if constexpr(picongpu::atomicPhysics::debug::rateCalculation::DEBUG_CHECKS)
                if(lowerState[i] < removedElectrons[i])
                {
                    printf(
                        "atomicPhysics ERROR: in multiplicityBoundFreeTransition lowerState[i] < "
                        "removedElectrons[i]\n");
                    return 1.;
                }

            combinatorialFactor *= picongpu::particles::atomicPhysics::rateCalculation::binomialCoefficient(
                lowerState[i],
                removedElectrons[i]);
        }

        return combinatorialFactor;
    }

} // namespace picongpu::particles::atomicPhysics::rateCalculation
