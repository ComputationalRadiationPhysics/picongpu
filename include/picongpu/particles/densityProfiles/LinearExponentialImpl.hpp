/*
 * SPDX-FileCopyrightText: Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

namespace picongpu
{
    namespace densityProfiles
    {
        template<typename T_ParamClass>
        struct LinearExponentialImpl : public T_ParamClass
        {
            using ParamClass = T_ParamClass;

            template<typename T_SpeciesType>
            struct apply
            {
                using type = LinearExponentialImpl<ParamClass>;
            };

            HINLINE LinearExponentialImpl(uint32_t currentStep)
            {
            }

            /* Calculate the normalized density
             *
             * @param totalCellOffset total offset including all slides [in cells]
             */
            HDINLINE float_X operator()(DataSpace<simDim> const& totalCellOffset)
            {
                float_X const vacuum_y = float_X(ParamClass::vacuumCellsY) * sim.pic.getCellSize().y();
                float_X const gas_a = ParamClass::gasA_SI * sim.unit.length();
                float_X const gas_d = ParamClass::gasD_SI * sim.unit.length();
                float_X const gas_y_max = ParamClass::gasYMax_SI / sim.unit.length();

                floatD_X const globalCellPos(
                    precisionCast<float_X>(totalCellOffset) * sim.pic.getCellSize().shrink<simDim>());
                auto density = float_X(0.0);

                if(globalCellPos.y() < vacuum_y)
                    return density;

                if(globalCellPos.y() <= gas_y_max) // linear slope
                    density = gas_a * globalCellPos.y() + ParamClass::gasB;
                else // exponential slope
                    density = math::exp((globalCellPos.y() - gas_y_max) * gas_d);

                // avoid < 0 densities for the linear slope
                if(density < float_X(0.0))
                    density = float_X(0.0);

                return density;
            }
        };
    } // namespace densityProfiles
} // namespace picongpu
