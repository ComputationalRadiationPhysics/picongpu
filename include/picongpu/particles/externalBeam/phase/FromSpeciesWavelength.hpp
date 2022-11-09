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

#include "picongpu/particles/PhotonFunctors.hpp"

namespace picongpu::particles::externalBeam::phase
{
    namespace acc
    {
        template<typename T_ParamClass>
        struct FromSpeciesWavelength
        {
            using Side = typename T_ParamClass::Side;

            DINLINE FromSpeciesWavelength(float_64 const& curPhase) : curPhase_m(curPhase)
            {
            }
            /* Set phase
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
                static constexpr float_X cellSizeCT[3] = {CELL_WIDTH, CELL_HEIGHT, CELL_DEPTH};
                static constexpr float_X cellDepth{cellSizeCT[Side::template BeamToSimIdx_t<2u>::value]};

                /* this functor will be called only in the first cell (counting from the boundary where
                the photon beam enters the simulation. So global position along the propagation axis
                is just the in cell position. */
                const floatD_X position = particle[position_];
                // distance from the z_beam=0 plane (the simulation boundary) where the position contribution
                // to the plane wave phase is 0.
                // note:  in theory we could just do operation on the z component only
                const float_X distance{
                    Side::transformOffsetSimToBeam(position, floatD_X::create(1.0_X)).z() * cellDepth};
                const float_X waveNumber{GetAngFrequency<T_Particle>()() / SPEED_OF_LIGHT};
                const float_X spatialContribution = waveNumber * distance;
                const float_X phase = spatialContribution + curPhase_m;

                particle[startPhase_] = phase;
            }

        private:
            PMACC_ALIGN(curPhase_m, float_64);
        };
    } // namespace acc

    //! Implementation of the host side factory for FromSpeciesWavelength
    template<typename T_ParamClass, typename T_Species>
    struct FromSpeciesWavelengthImpl
    {
        using Side = typename T_ParamClass::Side;
        static constexpr float_64 phi0 = T_ParamClass::phi0;

        HINLINE FromSpeciesWavelengthImpl(uint32_t const& currentStep)
            : curPhase(precisionCast<float_X>(GetPhaseByTimestep<T_Species>()(currentStep, phi0)))
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
            return acc::FromSpeciesWavelength<T_ParamClass>(curPhase);
        }
        PMACC_ALIGN(curPhase, float_64);
    };

    /*
     * We need this type wrapper since using a placeholder for T_Species somehow collides with all the applies
     * and it is not working correctly. It break the apply in StartAttributes, so that the StartAttributes is used
     *  instead of the host side factory implementation.
     */
    template<typename T_ParamClass>
    struct FromSpeciesWavelength
    {
        template<typename T_SpeciesType>
        struct apply
        {
            using type = FromSpeciesWavelengthImpl<T_ParamClass, T_SpeciesType>;
        };
    };
} // namespace picongpu::particles::externalBeam::phase
