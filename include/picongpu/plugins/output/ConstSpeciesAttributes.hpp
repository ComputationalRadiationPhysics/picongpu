/*
 * SPDX-FileCopyrightText: Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera, Franz Poeschel, Richard Pausch
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/traits/SIBaseUnits.hpp"
#include "picongpu/traits/frame/GetCharge.hpp"
#include "picongpu/traits/frame/GetMass.hpp"

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
                    return traits::frame::getCharge<T_FrameType>();
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
                    return picongpu::traits::frame::getMass<T_FrameType>();
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
