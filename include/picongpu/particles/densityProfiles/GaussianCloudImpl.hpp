/*
 * SPDX-FileCopyrightText: Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/simulation/control/MovingWindow.hpp"

namespace picongpu
{
    namespace densityProfiles
    {
        template<typename T_ParamClass>
        struct GaussianCloudImpl : public T_ParamClass
        {
            using ParamClass = T_ParamClass;

            template<typename T_SpeciesType>
            struct apply
            {
                using type = GaussianCloudImpl<ParamClass>;
            };

            HINLINE GaussianCloudImpl(uint32_t currentStep)
            {
            }

            /** Calculate the normalized density
             *
             * @param totalCellOffset total offset including all slides [in cells]
             */
            HDINLINE float_X operator()(DataSpace<simDim> const& totalCellOffset)
            {
                float_64 const unit_length = sim.unit.length();
                float_X const vacuum_y = float_X(ParamClass::vacuumCellsY) * sim.pic.getCellSize().y();
                constexpr auto centerSI = ParamClass::center_SI;
                floatD_X const center = precisionCast<float_X>(centerSI / unit_length);
                constexpr auto sigmaSI = ParamClass::sigma_SI;
                floatD_X const sigma = precisionCast<float_X>(sigmaSI / unit_length);

                floatD_X const globalCellPos(
                    precisionCast<float_X>(totalCellOffset) * sim.pic.getCellSize().shrink<simDim>());

                if(globalCellPos.y() < vacuum_y)
                    return float_X(0.0);

                /* for x, y, z calculate: x-x0 / sigma_x */
                floatD_X const r0overSigma = (globalCellPos - center) / sigma;
                /* get lenghts of r0 over sigma */
                float_X const exponent = pmacc::math::l2norm(r0overSigma);

                /* calculate exp(factor * exponent**power) */
                float_X const power = ParamClass::gasPower;
                float_X const factor = ParamClass::gasFactor;
                float_X const density = math::exp(factor * math::pow(exponent, power));

                return density;
            }
        };
    } // namespace densityProfiles
} // namespace picongpu
