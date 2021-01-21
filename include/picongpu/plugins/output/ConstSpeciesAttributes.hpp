/* Copyright 2014-2021 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Franz Poeschel, Richard Pausch
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

#pragma once

#include "picongpu/simulation_defines.hpp"
#include "picongpu/traits/frame/GetMass.hpp"
#include "picongpu/traits/frame/GetCharge.hpp"

#include <pmacc/traits/HasFlag.hpp>

#include <type_traits>

namespace picongpu
{
    namespace plugins
    {
        namespace output
        {
            template<typename T_FrameType>
            struct GetChargeOrZero
            {
                static constexpr bool hasChargeRatio = pmacc::traits::HasFlag<T_FrameType, chargeRatio<>>::type::value;

                template<typename T_Defer = float_X>
                typename std::enable_if<hasChargeRatio, T_Defer>::type operator()() const
                {
                    return frame::getCharge<T_FrameType>();
                }

                template<typename T_Defer = float_X>
                typename std::enable_if<!hasChargeRatio, T_Defer>::type operator()() const
                {
                    return float_X(0.);
                }

                std::vector<float_64> dimension() const
                {
                    // L, M, T, I, theta, N, J
                    std::vector<float_64> unitDimension(NUnitDimension, 0.0);
                    unitDimension.at(SIBaseUnits::electricCurrent) = 1.0;
                    unitDimension.at(SIBaseUnits::time) = 1.0;

                    return unitDimension;
                }
            };

            template<typename T_FrameType>
            struct GetMassOrZero
            {
                static constexpr bool hasMassRatio = pmacc::traits::HasFlag<T_FrameType, massRatio<>>::type::value;

                template<typename T_Defer = float_X>
                typename std::enable_if<hasMassRatio, T_Defer>::type operator()() const
                {
                    return frame::getMass<T_FrameType>();
                }

                template<typename T_Defer = float_X>
                typename std::enable_if<!hasMassRatio, T_Defer>::type operator()() const
                {
                    return float_X(0.);
                }

                std::vector<float_64> dimension() const
                {
                    // L, M, T, I, theta, N, J
                    std::vector<float_64> unitDimension(NUnitDimension, 0.0);
                    unitDimension.at(SIBaseUnits::mass) = 1.0;

                    return unitDimension;
                }
            };
        } // namespace output
    } // namespace plugins
} // namespace picongpu
