/* Copyright 2013-2023 Axel Huebl, Heiko Burau, Rene Widera, Pawel Ordyna
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

#include "picongpu/particles/startPosition/detail/WeightMacroParticles.hpp"
#include "picongpu/particles/startPosition/generic/FreeRng.def"

namespace picongpu
{
    namespace particles
    {
        namespace externalBeam
        {
            namespace startPosition
            {
                namespace acc
                {
                    //! Device side implementation for RandomProbingBeam
                    template<typename T_ParamClass>
                    struct RandomProbingBeam
                    {
                        using ParamClass = T_ParamClass;
                        // Defines from which side the beam enters the simulation box.
                        using Side = typename T_ParamClass::Side;

                    private:
                        /* compile-time calculation of the in-cell position range. Along the beam propagation direction
                         * ( z in the beam system) particles are created only up to the distance that a particle
                         * travels in one time-step. Here we assume the particles are photons and travel with the speed
                         * of light.
                         */
                        static constexpr float_X cellSizeCT[3] = {CELL_WIDTH, CELL_HEIGHT, CELL_DEPTH};
                        static constexpr float_X cellDepth{cellSizeCT[Side::template BeamToSimIdx_t<2u>::value]};
                        static constexpr float_X posLimBeam[3] = {1.0_X, 1.0_X, DELTA_T* SPEED_OF_LIGHT / cellDepth};
                        static constexpr float_X posLimPic_x = posLimBeam[Side::template SimToBeamIdx_t<0u>::value];
                        static constexpr float_X posLimPic_y = posLimBeam[Side::template SimToBeamIdx_t<1u>::value];
                        static constexpr float_X posLimPic_z = posLimBeam[Side::template SimToBeamIdx_t<2u>::value];

                        // This is true when the beam travels **against** one of the simulation coordinate system unit
                        // vectors.
                        static constexpr bool reverse = Side::reverse[2];


                    public:
                        /** Set in-cell position and weighting
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
                            // get the random number generator from context
                            // Get a random float value from 0,1
                            auto const& worker = *context.m_worker;
                            auto& rngHandle = *context.m_hRng;
                            auto rng
                                = rngHandle
                                      .template applyDistribution<pmacc::random::distributions::Uniform<float_X>>();
                            floatD_X tmpPos;

                            // array is not available in device code. even constexpr  array.
                            const float3_X posLimPicVec{posLimPic_x, posLimPic_y, posLimPic_z};

                            // generate a random in-cell position for each coordinate. In the beam propagation
                            // direction the position is limited by the distance a particle can travel in one
                            // time-step.
                            for(uint32_t d = 0; d < simDim; ++d)
                                tmpPos[d] = rng(worker) * posLimPicVec[d];


                            // Shift the coordinate along the beam propagation direction towards the external boundary
                            // if the boundary is on the end of the cell. ( The beam enters the simulation from the
                            // bottom, rear or the right side).
                            if constexpr(reverse)
                            {
                                tmpPos[Side::template BeamToSimIdx_t<2u>::value]
                                    = 1.0_X - tmpPos[Side::template BeamToSimIdx_t<2u>::value];
                            }

                            particle[position_] = tmpPos;
                            particle[weighting_] = m_weighting;
                        }

                        /* Get the number of particles needed to be created in the simulation cell
                         *
                         * @tparam T_Context start attributes context
                         * @tparam T_Particle type of the particles that should be created
                         *
                         * @param context start attributes context
                         * @param realParticlesPerCell number of new real particles in the cell in which this instance
                         * creates particles
                         *
                         * @return number of macro particles that need to be created in this cell (The operator() will
                         * be called that many times)
                         */
                        template<typename T_Particle, typename T_Context>
                        HDINLINE uint32_t
                        numberOfMacroParticles(T_Context const& context, float_X const realParticlesPerCell)
                        {
                            // Only create particles if weighting is not below the minimal value.
                            // Notice this is different as in the usual startPosition functor where the number of macro
                            // particles would be reduced to satisfy this condition.
                            m_weighting = realParticlesPerCell / T_ParamClass::numParticlesPerCell;
                            if(m_weighting < T_ParamClass::minWeighting)
                                return 0u;
                            return T_ParamClass::numParticlesPerCell;
                        }

                        float_X m_weighting;
                    };
                } // namespace acc
                template<typename T_ParamClass>
                struct RandomProbingBeam
                {
                    template<typename T_Species>
                    struct apply
                    {
                        using type = RandomProbingBeam<T_ParamClass>;
                    };

                    HINLINE RandomProbingBeam(uint32_t const& currentStep)
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
                    HDINLINE auto operator()(T_Worker const& worker, DataSpace<simDim> const& localSupercellOffset)
                        const
                    {
                        return acc::RandomProbingBeam<T_ParamClass>();
                    }
                };
            } // namespace startPosition
        } // namespace externalBeam
    } // namespace particles
} // namespace picongpu
