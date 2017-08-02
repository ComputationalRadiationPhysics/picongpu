/* Copyright 2017 Rene Widera
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
namespace filter
{

namespace acc
{

    //! check the particle handle
    struct IsHandleValid
    {

        /** check particle handle
         *
         * @tparam T_Particle pmacc::Particles, type of the particle
         * @tparam alpaka accelerator type
         *
         * @param alpaka accelerator
         * @param particle  particle which is checked
         * @return true if particle handle is valid, else false
         */
        template<
            typename T_Particle,
            typename T_Acc
        >
        DINLINE bool operator()(
            T_Acc const &,
            T_Particle const & particle
        )
        {
            return  particle.isHandleValid( );
        }
    };

} // namespace acc

    struct IsHandleValid
    {
        template< typename T_SpeciesType >
        struct apply
        {
            using type = IsHandleValid;
        };

        /** create filter for the accelerator
         *
         * @tparam T_WorkerCfg pmacc::mappings::threads::WorkerCfg, configuration of the worker
         * @param offset (in superCells, without any guards) relative
         *                        to the origin of the local domain
         * @param configuration of the worker
         */
        template<
            typename T_WorkerCfg,
            typename T_Acc
        >
        DINLINE acc::IsHandleValid
        operator( )(
            T_Acc const & acc,
            DataSpace< simDim > const &,
            T_WorkerCfg const &
        )
        {
            return acc::IsHandleValid{ };

        }
    };

} //namespace filter
} //namespace particles
} //namespace picongpu
