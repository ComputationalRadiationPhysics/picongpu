/* Copyright 2017-2021 Heiko Burau
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

#include "picongpu/algorithms/KinEnergy.hpp"
#include <pmacc/types.hpp>

namespace picongpu
{
    namespace plugins
    {
        namespace particleMerging
        {
            /** Status of a Voronoi cell */
            enum struct VoronoiStatus : uint8_t
            {
                /* !< a Voronoi cell is collecting particles (first state) */
                collecting,
                /* !< the Voronoi cell is splitting thus all its particles have
                 * to move to one of two sub-Voronoi cells */
                splitting,
                /* !< the cell needs to be destroyed. Before this can happen
                 * all its particles need to clear their voronoiCellId attribute. */
                abort,
                /* !< the Voronoi cell is ready for merging. After merging it is destroyed. */
                readyForMerging,
            };


            /** Stage of a Voronoi cell
             *
             * The spliiting process is two-fold: at first, the splitting is done regarding
             * only the spread in position and then by looking at the spread of momentum.
             */
            enum struct VoronoiSplittingStage : bool
            {
                /* !< the spatial distribution is splitted */
                position,
                /* !< the momentum distribution is splitted */
                momentum
            };


            /** Represents a Voronoi cell */
            struct VoronoiCell
            {
                VoronoiStatus status;
                VoronoiSplittingStage splittingStage;
                uint32_t numMacroParticles;
                float_X numRealParticles;

                float3_X meanValue;
                float3_X meanSquaredValue;

                uint8_t splittingComponent;
                int32_t lowerCellId;
                int32_t higherCellId;
                int firstParticleFlag;

                HDINLINE
                VoronoiCell(VoronoiSplittingStage splittingStage = VoronoiSplittingStage::position)
                    : status(VoronoiStatus::collecting)
                    , splittingStage(splittingStage)
                    , numMacroParticles(0)
                    , numRealParticles(float_X(0.0))
                    , meanValue(float3_X::create(0.0))
                    , meanSquaredValue(float3_X::create(0.0))
                    , firstParticleFlag(0)
                {
                }

                /** status setter */
                HDINLINE
                void setToAbort()
                {
                    this->status = VoronoiStatus::abort;
                }


                /** status setter */
                HDINLINE
                void setToSplitting(
                    const uint8_t splittingComponent,
                    const int32_t lowerCellId,
                    const int32_t higherCellId)
                {
                    this->status = VoronoiStatus::splitting;
                    this->splittingComponent = splittingComponent;
                    this->lowerCellId = lowerCellId;
                    this->higherCellId = higherCellId;
                }


                /** status setter */
                HDINLINE
                void setToReadyForMerging()
                {
                    this->status = VoronoiStatus::readyForMerging;
                }

                /** check if the current thread is associated to the first particle */
                template<typename T_Acc>
                DINLINE bool isFirstParticle(T_Acc const& acc)
                {
                    return cupla::atomicExch(acc, &this->firstParticleFlag, 1) == 0;
                }


                /** add a particle to this Voronoi cell */
                template<typename T_Acc>
                DINLINE void addParticle(
                    T_Acc const& acc,
                    const floatD_X position,
                    const float3_X momentum,
                    const float_X weighting)
                {
                    cupla::atomicAdd(
                        acc,
                        &this->numMacroParticles,
                        static_cast<uint32_t>(1),
                        ::alpaka::hierarchy::Threads{});
                    cupla::atomicAdd(acc, &this->numRealParticles, weighting, ::alpaka::hierarchy::Threads{});

                    if(this->splittingStage == VoronoiSplittingStage::position)
                    {
                        const floatD_X position2 = position * position;

                        for(int i = 0; i < simDim; i++)
                        {
                            cupla::atomicAdd(
                                acc,
                                &this->meanValue[i],
                                weighting * position[i],
                                ::alpaka::hierarchy::Threads{});
                            cupla::atomicAdd(
                                acc,
                                &this->meanSquaredValue[i],
                                weighting * position2[i],
                                ::alpaka::hierarchy::Threads{});
                        }
                    }
                    else
                    {
                        const float3_X momentum2 = momentum * momentum;

                        for(int i = 0; i < DIM3; i++)
                        {
                            cupla::atomicAdd(
                                acc,
                                &this->meanValue[i],
                                weighting * momentum[i],
                                ::alpaka::hierarchy::Threads{});
                            cupla::atomicAdd(
                                acc,
                                &this->meanSquaredValue[i],
                                weighting * momentum2[i],
                                ::alpaka::hierarchy::Threads{});
                        }
                    }
                }


                /** finalize mean value calculation */
                HDINLINE
                void finalizeMeanValues()
                {
                    this->meanValue /= this->numRealParticles;
                    this->meanSquaredValue /= this->numRealParticles;
                }

                /** get the mean energy of this Voronoi cell if called in momentum stage */
                HDINLINE
                float_X getMeanEnergy(const float_X mass) const
                {
                    return KinEnergy<>()(this->meanValue, mass);
                }

                /** get the mean momentum squared of this Voronoi cell if called in momentum stage */
                HDINLINE
                float_X getMeanMomentum2() const
                {
                    return pmacc::math::abs2(this->meanValue);
                }


                /** determine in which of the two sub-Voronoi cells a particle falls */
                HDINLINE
                int32_t getSubVoronoiCell(const floatD_X position, const float3_X momentum) const
                {
                    const float_X valParticle = this->splittingStage == VoronoiSplittingStage::position
                        ? position[this->splittingComponent]
                        : momentum[this->splittingComponent];

                    const float_X meanVoronoi = this->meanValue[this->splittingComponent];

                    return valParticle < meanVoronoi ? this->lowerCellId : this->higherCellId;
                }


                /** auxillary function for getting the mean squared deviation in position or momentum */
                HDINLINE
                float_X getMaxValueSpread2(uint8_t& component, const uint8_t dimension) const
                {
                    const float3_X meanValue2 = this->meanValue * this->meanValue;
                    const float3_X valueSpread2 = this->meanSquaredValue - meanValue2;

                    /* find component of most spread in position */
                    component = 0;
                    float_X maxValueSpread2 = valueSpread2[0];
                    for(uint8_t i = 1; i < dimension; i++)
                    {
                        if(valueSpread2[i] > maxValueSpread2)
                        {
                            maxValueSpread2 = valueSpread2[i];
                            component = i;
                        }
                    }

                    return maxValueSpread2;
                }


                /** calculate the maxmimum squared spread in position
                 *
                 * @param component index of position component of maxmimum spread
                 * @return maxmimum squared spread in position
                 */
                HDINLINE
                float_X getMaxPositionSpread2(uint8_t& component) const
                {
                    return this->getMaxValueSpread2(component, simDim);
                }


                /** calculate the maxmimum squared spread in momentum
                 *
                 * @param component index of momentum component of maxmimum spread
                 * @return maxmimum squared spread in momentum
                 */
                HDINLINE
                float_X getMaxMomentumSpread2(uint8_t& component) const
                {
                    return this->getMaxValueSpread2(component, DIM3);
                }
            };

        } // namespace particleMerging
    } // namespace plugins
} // namespace picongpu
