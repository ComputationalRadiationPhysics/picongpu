/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera
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

#include <pmacc/mappings/simulation/SubGrid.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>


namespace picongpu
{
    namespace fields
    {
        namespace laserProfiles
        {
            namespace none
            {
                template<typename T_Params>
                struct Unitless : public T_Params
                {
                    using Params = T_Params;

                    static constexpr float_X WAVE_LENGTH
                        = float_X(Params::WAVE_LENGTH_SI / UNIT_LENGTH); // unit: meter
                    static constexpr float_X PULSE_LENGTH
                        = float_X(Params::PULSE_LENGTH_SI / UNIT_TIME); // unit: seconds (1 sigma)
                    static constexpr float_X AMPLITUDE
                        = float_X(Params::AMPLITUDE_SI / UNIT_EFIELD); // unit: Volt /meter
                    static constexpr float_X INIT_TIME = 0.0_X; // unit: seconds (no initialization time)
                };
            } // namespace none
            namespace acc
            {
                template<typename T_Unitless>
                struct None : public T_Unitless
                {
                    using Unitless = T_Unitless;

                    /** Device-Side Constructor
                     */
                    HDINLINE None()
                    {
                    }

                    /** device side manipulation for init plane (transversal)
                     *
                     * @tparam T_Args type of the arguments passed to the user manipulator functor
                     */
                    template<typename T_Acc>
                    HDINLINE void operator()(T_Acc const&, DataSpace<simDim> const&)
                    {
                    }
                };
            } // namespace acc

            template<typename T_Params>
            struct None : public none::Unitless<T_Params>
            {
                using Unitless = none::Unitless<T_Params>;

                /** constructor
                 */
                HINLINE None(uint32_t)
                {
                }

                /** create device manipulator functor
                 *
                 * @tparam T_WorkerCfg pmacc::mappings::threads::WorkerCfg, configuration of the worker
                 * @tparam T_Acc alpaka accelerator type
                 */
                template<typename T_WorkerCfg, typename T_Acc>
                HDINLINE acc::None<Unitless> operator()(T_Acc const&, DataSpace<simDim> const&, T_WorkerCfg const&)
                    const
                {
                    return acc::None<Unitless>();
                }

                //! get the name of the laser profile
                static HINLINE std::string getName()
                {
                    return "None";
                }
            };

        } // namespace laserProfiles
    } // namespace fields
} // namespace picongpu
