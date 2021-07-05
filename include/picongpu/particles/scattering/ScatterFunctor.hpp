/* Copyright 2021 Pawel Ordyna
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
namespace picongpu
{
    namespace particles
    {
        namespace scattering
        {
            namespace acc
            {
                template<typename T_AccConditionFunctor, typename T_AccDirectionFunctor>
                struct ScatterFunctor
                {
                    HDINLINE ScatterFunctor(
                        T_AccConditionFunctor const& accConditionFunctor,
                        T_AccDirectionFunctor const& accDirectionFunctor)
                        : accConditionFunctor_m(accConditionFunctor)
                        , accDirectionFunctor_m(accDirectionFunctor)
                    {
                    }
                    template<typename T_Acc, typename T_Particle>
                    DINLINE void operator()(T_Acc const& acc, T_Particle& particle, float_X const& density)
                    {
                        const bool condition = accConditionFunctor_m(acc, particle, density);
                        if(condition)
                        {
                            accDirectionFunctor_m(acc, particle, density);
                        }
                    }

                private:
                    PMACC_ALIGN(accConditionFunctor_m, T_AccConditionFunctor);
                    PMACC_ALIGN(accDirectionFunctor_m, T_AccDirectionFunctor);
                };
            } // namespace acc

            template<typename T_DensityFields, typename T_ConditionFunctor, typename T_DirectionFunctor>
            struct ScatterFunctor
                : private T_ConditionFunctor
                , private T_DirectionFunctor
            {
                using RequiredDerivedFields = T_DensityFields;
                HINLINE ScatterFunctor(uint32_t const& currentStep)
                    : T_ConditionFunctor(currentStep)
                    , T_DirectionFunctor(currentStep)
                {
                }

                template<typename T_Acc, typename T_WorkerCfg>
                HDINLINE auto operator()(
                    T_Acc const& acc,
                    DataSpace<simDim> const& localSupercellOffset,
                    T_WorkerCfg const& workerCfg) const
                {
                    return acc::ScatterFunctor<
                        typename T_ConditionFunctor::AccFunctorType,
                        typename T_DirectionFunctor::AccFunctorType>(
                        T_ConditionFunctor::operator()(acc, localSupercellOffset, workerCfg),
                        T_DirectionFunctor::operator()(acc, localSupercellOffset, workerCfg));
                }
            };
        } // namespace scattering
    } // namespace particles
} // namespace picongpu
