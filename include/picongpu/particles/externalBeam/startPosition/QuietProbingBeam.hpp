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

#include "picongpu/particles/externalBeam/beam/Side.hpp"

#include <boost/mpl/integral_c.hpp>


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
                    //! Device side implementation for QuietProbingBeam
                    template<typename T_ParamClass>
                    struct QuietProbingBeam
                    {
                        using ParamClass = T_ParamClass;
                        // Defines from which side the beam enters the simulation box.
                        using Side = typename T_ParamClass::Side;

                    private:
                        // number of particles in each direction
                        using numParPerDimension =
                            typename Side::template TwistBeamToSim_t<typename T_ParamClass::numParticlesPerDimension>;

                        /* compile-time calculation of the in-cell particle spacing. Along the beam propagation
                         * direction ( z in the beam system) particles are created only up to the distance that a
                         * particle travels in one time-step. Here we assume the particles are photons and travel with
                         * the speed of light.
                         */
                        static constexpr float_X cellSizeCT[3] = {CELL_WIDTH, CELL_HEIGHT, CELL_DEPTH};
                        static constexpr float_X cellDepth{cellSizeCT[Side::template BeamToSimIdx_t<2u>::value]};
                        static constexpr float_X spacingBeam[3] = {1.0_X, 1.0_X, DELTA_T* SPEED_OF_LIGHT / cellDepth};
                        static constexpr float_X spacing3D_x{
                            spacingBeam[Side::template SimToBeamIdx_t<0u>::value]
                            / static_cast<float_X>(numParPerDimension::template at<0>::type::value)};
                        static constexpr float_X spacing3D_y{
                            spacingBeam[Side::template SimToBeamIdx_t<1u>::value]
                            / static_cast<float_X>(numParPerDimension::template at<1>::type::value)};
                        static constexpr float_X spacing3D_z{
                            spacingBeam[Side::template SimToBeamIdx_t<2u>::value]
                            / static_cast<float_X>(numParPerDimension::template at<2>::type::value)};
                        // discard the extra dimension in case of a 2D simulation
                        using numParShrinked = typename mCT::shrinkTo<numParPerDimension, simDim>::type;

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
                        DINLINE void operator()(T_Context const& context, T_Particle& particle, T_Args&&...)
                        {
                            uint32_t maxNumMacroParticles = pmacc::math::CT::volume<
                                typename T_ParamClass::numParticlesPerDimension>::type::value;
                            /* reset the particle position if the operator is called more times than allowed
                             * (m_currentMacroParticles underflow protection.
                             */
                            if(maxNumMacroParticles <= m_currentMacroParticles)
                                m_currentMacroParticles = maxNumMacroParticles - 1u;

                            // particle spacing as a run-time float vector
                            const float3_X spacing3D{spacing3D_x, spacing3D_y, spacing3D_z};
                            const floatD_X spacing = spacing3D.shrink<simDim>();
                            /* coordinate in the local in-cell lattice
                             *   x = [0, numParsPerCell_X-1]
                             *   y = [0, numParsPerCell_Y-1]
                             *   z = [0, numParsPerCell_Z-1]
                             */
                            DataSpace<simDim> inCellCoordinate
                                = DataSpaceOperations<simDim>::map(numParShrinked::toRT(), m_currentMacroParticles);

                            floatD_X inCellPosition
                                = precisionCast<float_X>(inCellCoordinate) * spacing + spacing * float_X(0.5);

                            // Shift the coordinate along the beam propagation direction towards the external boundary
                            // if the boundary is on the end of the cell. ( The beam enters the simulation from the
                            // bottom, rear or the right side).
                            if constexpr(reverse)
                            {
                                inCellPosition[Side::template BeamToSimIdx_t<2u>::value]
                                    = 1.0_X - inCellPosition[Side::template BeamToSimIdx_t<2u>::value];
                            }

                            particle[position_] = inCellPosition;
                            particle[weighting_] = m_weighting;

                            // decrease the number of unprocessed particles by 1
                            --m_currentMacroParticles;
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
                            m_weighting = float_X(0.0);
                            uint32_t numMacroParticles = pmacc::math::CT::volume<numParShrinked>::type::value;

                            // particle weighting if all macro particles are created
                            if(numMacroParticles > 0u)
                                m_weighting = realParticlesPerCell / float_X(numMacroParticles);

                            // Only create particles if weighting is not below the minimal value.
                            // Notice this is different as in the usual startPosition functor where the number of macro
                            // particles would be reduced to satisfy this condition.

                            if(m_weighting < T_ParamClass::minWeighting)
                                return 0u;
                            else
                            {
                                m_currentMacroParticles = numMacroParticles - 1u;
                                return numMacroParticles;
                            }
                        }

                    private:
                        PMACC_ALIGN(m_weighting, float_X);
                        PMACC_ALIGN(m_currentMacroParticles, uint32_t);
                    };

                } // namespace acc

                template<typename T_ParamClass>
                struct QuietProbingBeam
                {
                    template<typename T_Species>
                    struct apply
                    {
                        using type = QuietProbingBeam<T_ParamClass>;
                    };

                    HINLINE QuietProbingBeam(uint32_t const& currentStep)
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
                        return acc::QuietProbingBeam<T_ParamClass>();
                    }
                };
            } // namespace startPosition
        } // namespace externalBeam
    } // namespace particles
} // namespace picongpu
