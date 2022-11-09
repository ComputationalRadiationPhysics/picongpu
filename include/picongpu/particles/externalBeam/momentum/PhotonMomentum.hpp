/* Copyright 2021-2023 Pawel Ordyna
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

#include <type_traits>

namespace picongpu::particles::externalBeam::momentum
{
    namespace acc
    {
        //! Device side implementation fo the PhotonMomentum functor
        template<typename T_ParamClass>
        struct PhotonMomentum
        {
        private:
            using ParamClass = T_ParamClass;
            // Defines from which side the beam enters the simulation box.
            using Side = typename T_ParamClass::Side;
            static constexpr float_64 photonMomentum{
                ParamClass::photonEnergySI / UNIT_ENERGY / static_cast<float_64>(SPEED_OF_LIGHT)};
            // Photons propagate along the beam z direction
            // Check if the beam z axis is a reversed pic axis. If so reverse z before axis swap.
            static constexpr bool reverse = Side::reverse[2];
            using DirectionBeam = typename std::conditional<reverse, mCT::Int<0, 0, -1>, mCT::Int<0, 0, 1>>::type;
            // Convert into the pic simulation system
            using DirectionSim = typename Side::template TwistBeamToSim_t<DirectionBeam>;

        public:
            /** Set particle momentum
             *
             * @tparam T_Context start attributes context
             * @tparam T_Particle pmacc::Particle, particle type
             * @tparam T_Args pmacc::Particle, arbitrary number of particles types
             *
             * @param context start attributes context
             * @param particle particle to be manipulated
             * @param ... unused particles
             */
            template<typename T_Context, typename T_Particle, typename... T_Args>
            DINLINE void operator()(T_Context const& context, T_Particle& particle, T_Args&&...) const
            {
                float_X const macroWeighting{particle[weighting_]};
                float3_X const direction{precisionCast<float_X>(DirectionSim::toRT())};
                // Don't need to normalize direction here since the norm is 1.
                float3_X const mom{direction * (static_cast<float_X>(photonMomentum) * macroWeighting)};
                particle[momentum_] = mom;
            }
        };
    } // namespace acc

    template<typename T_ParamClass>
    struct PhotonMomentum
    {
        template<typename T_Species>
        struct apply
        {
            using type = PhotonMomentum<T_ParamClass>;
        };

        HINLINE PhotonMomentum(uint32_t const& currentStep)
        {
        }

        /** create functor for the accelerator
         *
         * @tparam T_Worker lockstep worker type
         *
         * @param worker lockstep worker
         * @param localSupercellOffset offset (in superCells, without any guards) relative
         *                        to the origin of the local domain
         */
        template<typename T_Worker>
        DINLINE auto operator()(T_Worker const& worker, DataSpace<simDim> const& localSupercellOffset) const
        {
            return acc::PhotonMomentum<T_ParamClass>();
        }
    };
} // namespace picongpu::particles::externalBeam::momentum
