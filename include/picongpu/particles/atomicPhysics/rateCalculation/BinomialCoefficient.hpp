/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/ConvertEnum.hpp"
#include "picongpu/particles/atomicPhysics/param.hpp"

namespace picongpu::particles::atomicPhysics::rateCalculation
{
    /** binomial coefficient calculated using multiplicative formula
     *
     * @attention binomials get very big very fast and may exceed value ranges,
     *    should be no problem for FLYonPIC flyCHK-input data since largest value ~10^10,
     *    might become a problem if all possible states are used.
     *
     * see https://en.wikipedia.org/wiki/Binomial_coefficient#Computing_the_value_of_binomial_coefficients
     *  for more information
     */
    HDINLINE float_64 binomialCoefficient(uint8_t n, uint8_t k)
    {
        // check for limits, no check for < 0 necessary, since uint
        if constexpr(picongpu::atomicPhysics::debug::rateCalculation::DEBUG_CHECKS)
            if(n < k)
            {
                printf("invalid call binomial(n,k), n < k\n");
                return static_cast<float_64>(0.);
            }

        /// @todo beneficial?, Brian Marre, 2022
        // reduce necessary steps using symmetry in k
        if(k > (n / u8(2u)))
        {
            k = n - k;
        }

        float_64 result = 1.;

        for(uint8_t i = 1u; i <= k; i++)
        {
            result *= static_cast<float_64>(n - i + 1) / static_cast<float_64>(i);
        }
        return result;
    }

} // namespace picongpu::particles::atomicPhysics::rateCalculation
