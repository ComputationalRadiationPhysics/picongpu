/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera, Pawel Ordyna
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

#include "picongpu/plugins/externalBeam/AxisSwap.hpp"
#include "picongpu/plugins/externalBeam/Side.hpp"

#include <boost/mpl/integral_c.hpp>


namespace picongpu
{
    namespace particles
    {
        namespace externalBeam
        {
            namespace startPosition
            {
                template<typename T_ParamClass>
                struct QuietProbingBeam
                {
                    using ParamClass = T_ParamClass;
                    using SideCfg = typename T_ParamClass::ProbingBeam::SideCfg;

                private:
                    template<uint32_t idx>
                    using BeamToPicIdx_t = typename SideCfg::AxisSwapCT::template BeamToPicIdx<idx>::type;
                    template<uint32_t idx>
                    using PicToBeamIdx_t = typename SideCfg::AxisSwapCT::template PicToBeamIdx<idx>::type;

                    // spacing between particles in each direction in the cell in thepic system
                    using numParPerDimension = typename SideCfg::AxisSwapCT::template ReverseSwap<
                        typename T_ParamClass::numParticlesPerDimension>::type;
                    static constexpr float_X cellSizeCT[3] = {CELL_WIDTH, CELL_HEIGHT, CELL_DEPTH};
                    static constexpr float_X cellDepth{cellSizeCT[BeamToPicIdx_t<2u>::value]};
                    static constexpr float_X spacingBeam[3] = {1.0_X, 1.0_X, DELTA_T* SPEED_OF_LIGHT / cellDepth};
                    static constexpr float_X spacing3D_x{
                        spacingBeam[PicToBeamIdx_t<0u>::value]
                        / static_cast<float_X>(numParPerDimension::template at<0>::type::value)};
                    static constexpr float_X spacing3D_y{
                        spacingBeam[PicToBeamIdx_t<1u>::value]
                        / static_cast<float_X>(numParPerDimension::template at<1>::type::value)};
                    static constexpr float_X spacing3D_z{
                        spacingBeam[PicToBeamIdx_t<2u>::value]
                        / static_cast<float_X>(numParPerDimension::template at<2>::type::value)};
                    using numParShrinked = typename mCT::shrinkTo<numParPerDimension, simDim>::type;
                    static constexpr bool reverse = SideCfg::Side::reverse[2];

                public:
                    HINLINE QuietProbingBeam(uint32_t currentStep): axisSwap()
                    {
                    }

                    /** set in-cell position and weighting
                     *
                     * @warning It is not allowed to call this functor as many times as
                     *          the resulting value of numberOfMacroParticles.
                     *
                     * @tparam T_Particle pmacc::Particle, particle type
                     * @tparam T_Args pmacc::Particle, arbitrary number of particles types
                     *
                     * @param particle particle to be manipulated
                     * @param ... unused particles
                     */
                    template<typename T_Acc, typename T_MetaData, typename T_Particle, typename... T_Args>
                    HDINLINE void operator()(T_Acc const& acc, T_MetaData& meta, T_Particle& particle, T_Args&&...)
                    {
                        uint32_t maxNumMacroParticles
                            = pmacc::math::CT::volume<typename T_ParamClass::numParticlesPerDimension>::type::value;
                        /* reset the particle position if the operator is called more
                        times
                         * than allowed (m_currentMacroParticles underflow protection
                         for)
                         */
                        if(maxNumMacroParticles <= m_currentMacroParticles)
                            m_currentMacroParticles = maxNumMacroParticles - 1u;

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
                        if(reverse)
                        {
                            inCellPosition[BeamToPicIdx_t<2u>::value]
                                = 1.0_X - inCellPosition[BeamToPicIdx_t<2u>::value];
                        }

                        particle[position_] = inCellPosition;
                        particle[weighting_] = m_weighting;

                        --m_currentMacroParticles;
                    }

                    template<typename T_Particle, typename T_MetaData>
                    HDINLINE uint32_t
                    numberOfMacroParticles(T_MetaData const& meta, float_X const realParticlesPerCell)

                    {
                        // Check if the beam is coming from this side.
                        DataSpace<simDim> globalDomainOffset = meta.domInfo.global.offset;
                        DataSpace<DIM3> globalDomainOffsetBeamSystem = axisSwap.transformCellIdx(globalDomainOffset);
                        if(globalDomainOffsetBeamSystem.z() != 0)
                            return 0u;

                        using numParPerDimension = typename SideCfg::AxisSwapCT::template ReverseSwap<
                            typename T_ParamClass::numParticlesPerDimension>::type;
                        using numParShrinked = typename mCT::shrinkTo<numParPerDimension, simDim>::type;
                        auto numParInCell = numParShrinked::toRT();

                        m_weighting = float_X(0.0);
                        uint32_t numMacroParticles = pmacc::math::CT::volume<numParShrinked>::type::value;

                        if(numMacroParticles > 0u)
                            m_weighting = realParticlesPerCell / float_X(numMacroParticles);

                        while(m_weighting < MIN_WEIGHTING && numMacroParticles > 0u)
                        {
                            /* decrement component with greatest value*/
                            uint32_t max_component = 0u;
                            for(uint32_t i = 1; i < simDim; ++i)
                            {
                                if(numParInCell[i] > numParInCell[max_component])
                                    max_component = i;
                            }
                            numParInCell[max_component] -= 1u;

                            numMacroParticles = numParInCell.productOfComponents();

                            if(numMacroParticles > 0u)
                                m_weighting = realParticlesPerCell / float_X(numMacroParticles);
                            else
                                m_weighting = float_X(0.0);
                        }
                        m_currentMacroParticles = numMacroParticles - 1u;
                        return numMacroParticles;
                    }

                private:
                    PMACC_ALIGN(m_weighting, float_X);
                    PMACC_ALIGN(m_currentMacroParticles, uint32_t);
                    PMACC_ALIGN(axisSwap, typename SideCfg::AxisSwapRT);
                };

            } // namespace startPosition
        } // namespace externalBeam
    } // namespace particles
} // namespace picongpu
