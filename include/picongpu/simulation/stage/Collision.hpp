/* Copyright 2019-2023 Rene Widera, Pawel Ordyna
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

#include "picongpu/particles/collision/collision.hpp"
#include "picongpu/particles/collision/fieldSlots.hpp"
#include "picongpu/particles/particleToGrid/ComputeFieldValue.hpp"
#include "picongpu/particles/particleToGrid/FoldDeriveFields.hpp"
#include "picongpu/particles/particleToGrid/combinedAttributes/CombinedAttributes.def"

#include <pmacc/Environment.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/meta/ForEach.hpp>

#include <cstdint>
#include <iostream>
#include <utility>


namespace picongpu
{
    namespace simulation
    {
        namespace stage
        {
            namespace collision
            {
                //! For each implementation for calling collisions with a loop index
                struct CallColliders
                {
                    template<std::size_t... I>
                    HINLINE void operator()(
                        std::index_sequence<I...>,
                        std::shared_ptr<DeviceHeap> const& deviceHeap,
                        uint32_t const& currentStep)
                    {
                        (particles::collision::CallCollider<
                             pmacc::mp_at_c<particles::collision::CollisionPipeline, I>,
                             I>{}(deviceHeap, currentStep),
                         ...);
                    }
                };

                template<typename T_Type>
                struct Sqrt1DimVector
                {
                    using result = T_Type;

                    HDINLINE result operator()(T_Type vector) const
                    {
                        using ScalarType = typename T_Type::type;
                        return T_Type(math::sqrt(static_cast<ScalarType>(vector)));
                    }
                };

                //! Calculates the Debye length squared from the sum of the species contributions
                struct GetSquaredScreeningLength
                {
                    template<typename T_Acc>
                    DINLINE void operator()(T_Acc const& acc, float1_X& lengthInvSquared, const float1_X& dens) const
                    {
                        /* avoid dividing by zero, lengthInvSquared is zero if there were no charged particles.
                         * In that case the Debye length should also be set to zero.
                         */
                        if(lengthInvSquared[0] <= std::numeric_limits<float_X>::min())
                        {
                            // if no particles set debye length to zero
                            lengthInvSquared[0] = 0.0_X;
                            return;
                        }
                        // the default case
                        const float_X squaredLength = 1.0_X / lengthInvSquared[0];

                        // enforce a lower cutoff for the Debye length equal to the mean inter atomic distance
                        // of the species with the highest density
                        const float_X maxDens
                            = dens[0] * static_cast<float_X>(particles::TYPICAL_NUM_PARTICLES_PER_MACROPARTICLE);
                        const float_X val = 2.0_X * pmacc::math::Pi<float_X>::doubleValue / 3.0_X * maxDens;
                        const float_X rMin = 1.0_X / math::cbrt(val);
                        const float_X rMin2 = rMin * rMin;
                        lengthInvSquared[0] = math::max(squaredLength, rMin2);
                    }
                };
            } // namespace collision

            //! Functor for the stage of the PIC loop performing particle collision
            class Collision
            {
            public:
                Collision(std::shared_ptr<DeviceHeap>& heap) : m_heap(heap)
                {
                }

                static void debug(uint32_t const currentStep, const std::shared_ptr<FieldTmp>& screeningLengthSquared)
                {
                    // write Debye length averaged over the whole simulation volume to a text file for debugging
                    if constexpr(particles::collision::debugScreeningLength)
                    {
                        algorithms::GlobalReduce globalReduce(1024);
                        using SqrtBox = DataBoxUnaryTransform<
                            typename GridBuffer<float1_X, simDim>::DataBoxType,
                            collision::Sqrt1DimVector>;
                        DataBoxDim1Access<SqrtBox> d1access(
                            SqrtBox(screeningLengthSquared->getDeviceDataBox().shift(
                                screeningLengthSquared->getGridLayout().getGuard())),
                            screeningLengthSquared->getGridLayout().getDataSpaceWithoutGuarding());
                        const auto elements = static_cast<uint32_t>(screeningLengthSquared->getGridLayout()
                                                                        .getDataSpaceWithoutGuarding()
                                                                        .productOfComponents());
                        auto reducedValue = globalReduce(pmacc::math::operation::Add(), d1access, elements);

                        mpi::MPIReduce reduce{};
                        auto localCells = static_cast<uint64_t>(elements);
                        uint64_t reducedCellAmount;
                        reduce(
                            pmacc::math::operation::Add(),
                            &reducedCellAmount,
                            &localCells,
                            1,
                            mpi::reduceMethods::Reduce());

                        if(reduce.hasResult(mpi::reduceMethods::Reduce()))
                        {
                            std::ofstream outFile{};
                            std::string fileName = "average_debye_length_for_collisions.dat";
                            outFile.open(fileName.c_str(), std::ofstream::out | std::ostream::app);
                            outFile << currentStep << " " << std::scientific
                                    << static_cast<float_64>(reducedValue[0])
                                    / static_cast<float_64>(reducedCellAmount) * UNIT_LENGTH
                                    << std::endl;
                            outFile.flush();
                            outFile.close();
                        }
                    }
                }
                /** Perform particle particle collision
                 *
                 * @param step index of time iteration
                 */
                void operator()(uint32_t const currentStep) const
                {
                    // Calculate squared Debye length using the formula form [Perez 2012].
                    // A species temperature is assumed to be (2/3)<E_kin>
                    constexpr auto numScreeningSpecies
                        = pmacc::mp_size<picongpu::particles::collision::CollisionScreeningSpecies>::value;
                    if constexpr(numScreeningSpecies > 0u)
                    {
                        constexpr size_t requiredSlots = numScreeningSpecies == 1u ? 2 : 3;

                        PMACC_CASSERT_MSG(
                            _please_allocate_enough_FieldTmp_in_memory_param,
                            fieldTmpNumSlots >= requiredSlots || numScreeningSpecies == 0u);
                        DataConnector& dc = Environment<>::get().DataConnector();
                        using SpeciesSeq = picongpu::particles::collision::CollisionScreeningSpecies;
                        constexpr uint32_t slot = picongpu::particles::collision::screeningLengthSlot;
                        // get a FieldTmp for storring the result
                        auto screeningLengthSquared = dc.get<FieldTmp>(FieldTmp::getUniqueId(slot));
                        // The inverse of the Debye length squared is the sum over ScreeningInvSquared of each
                        // species. Calculate it using the fold algorithm:
                        particles::particleToGrid::FoldDeriveFields<
                            CORE + BORDER,
                            SpeciesSeq,
                            pmacc::math::operation::Add,
                            particles::particleToGrid::combinedAttributes::ScreeningInvSquared>{}(
                            *screeningLengthSquared,
                            currentStep,
                            slot + 1u);

                        // The Debye length is bound to be greater than the mean inter-atomic distance of the most
                        // dense species ( this definition of the lower cut-off comes from [Perez 2012]).
                        // To find this cut-off we need to find the highest density:
                        auto maxDensity = dc.get<FieldTmp>(FieldTmp::getUniqueId(slot + 1u));
                        particles::particleToGrid::FoldDeriveFields<
                            CORE + BORDER,
                            SpeciesSeq,
                            pmacc::math::operation::Max,
                            particles::particleToGrid::derivedAttributes::Density>{}(
                            *maxDensity,
                            currentStep,
                            slot + 2u);

                        // Invert the calculated value and apply the cut-off to get the squared debye length
                        screeningLengthSquared->modifyByField<CORE + BORDER, collision::GetSquaredScreeningLength>(
                            *maxDensity);
                        // the FieldTmp slot used for storing  the max density is no longer needed beyond this point
                        maxDensity.reset();
                        debug(currentStep, screeningLengthSquared);
                    }
                    // Call all colliders
                    constexpr size_t numColliders = pmacc::mp_size<particles::collision::CollisionPipeline>::value;
                    std::make_index_sequence<numColliders> indexColliders{};
                    collision::CallColliders{}(indexColliders, m_heap, currentStep);
                }

            private:
                std::shared_ptr<DeviceHeap> m_heap;
            };

        } // namespace stage
    } // namespace simulation
} // namespace picongpu
