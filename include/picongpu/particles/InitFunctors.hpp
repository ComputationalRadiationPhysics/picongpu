/* Copyright 2014-2022 Rene Widera, Sergei Bastrakov
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

#include "picongpu/fields/Fields.def"
#include "picongpu/particles/Manipulate.hpp"
#include "picongpu/particles/densityProfiles/IProfile.def"
#include "picongpu/particles/filter/filter.def"
#include "picongpu/particles/manipulators/manipulators.def"

#include <pmacc/Environment.hpp>
#include <pmacc/meta/conversion/TypeToPointerPair.hpp>
#include <pmacc/particles/meta/FindByNameOrType.hpp>
#include <pmacc/traits/GetFlagType.hpp>
#include <pmacc/traits/HasFlag.hpp>
#include <pmacc/traits/Resolve.hpp>

#include <boost/mpl/apply.hpp>

namespace picongpu
{
    namespace particles
    {
        /** Sample macroparticles according to the given spatial density profile
         *
         * Create macroparticles inside a species.
         *
         * This function only concerns the number of macroparticles, positions, and weighting.
         * So it basically performs sampling in the coordinate space, while not initializing other attributes.
         * When needed, those should be set (for then-existing macroparticles) by subsequently calling Manipulate.
         *
         * User input to this functor is two-fold.
         * T_DensityFunctor represents spatial density of real particles, normalized according to our requirements.
         * It describes the physical setup being simulated and only deals with real, not macro-, particles.
         * T_PositionFunctor is more of a PIC simulation parameter.
         * It defines how real particles in each cell will be represented with macroparticles.
         * This concerns the count, weighting, and in-cell positions of the created macroparticles.
         *
         * The sampling process operates independently for each cell, as follows:
         *    - Evaluate the amount of real particles in the cell, Nr, using T_DensityFunctor.
         *    - If Nr > 0, decide how to represent it with macroparticles using T_PositionFunctor:
         *        - (For simplicity we describe how all currently used functors but RandomPositionAndWeightingImpl
         *           and RandomBinomialImpl operate, see below for customization)
         *        - Try to have exactly T_PositionFunctor::numParticlesPerCell macroparticles
         *          with same weighting w = Nr / T_PositionFunctor::numParticlesPerCell.
         *        - If such w < MIN_WEIGHTING, instead use fewer macroparticles and higher weighting.
         *        - In any case the combined weighting of all new macroparticles will match Nr.
         *    - Create the selected number of macroparticles with selected weighting.
         *    - Set in-cell positions according to T_PositionFunctor.
         *
         * In principle, one could override the logic inside the (If Nr > 0) block by implementing a custom functor.
         * Then one could have an arbitrary number of macroparticles and weight distribution between them.
         * The only requirement is that together it matches Nr.
         * For an example of non-uniform weight distribution @see startPosition::RandomPositionAndWeightingImpl.
         * Note that in this scheme almost all non-vacuum cells will start with the same number of macroparticles.
         * Having a higher density in a cell would mean larger weighting, but not more macroparticles.
         *
         * @note FillAllGaps is automatically called after creation.
         *
         * @tparam T_DensityFunctor unary lambda functor with profile description,
         *                          see density.param,
         *                          example: picongpu::particles::densityProfiles::Homogenous
         * @tparam T_PositionFunctor unary lambda functor with position description and number of macroparticles per
         * cell, see particle.param, examples: picongpu::particles::startPosition::Quiet,
         *                                     picongpu::particles::startPosition::Random
         * @tparam T_SpeciesType type or name as PMACC_CSTRING of the used species,
         *                       see speciesDefinition.param
         */
        template<typename T_DensityFunctor, typename T_PositionFunctor, typename T_SpeciesType = boost::mpl::_1>
        struct CreateDensity
        {
            using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SpeciesType>;
            using FrameType = typename SpeciesType::FrameType;


            using UserDensityFunctor = typename boost::mpl::apply1<T_DensityFunctor, SpeciesType>::type;
            /* add interface for compile time interface validation*/
            using DensityFunctor = densityProfiles::IProfile<UserDensityFunctor>;

            using UserPositionFunctor = typename boost::mpl::apply1<T_PositionFunctor, SpeciesType>::type;
            /* add interface for compile time interface validation*/
            using PositionFunctor = manipulators::IUnary<UserPositionFunctor>;

            HINLINE void operator()(const uint32_t currentStep)
            {
                DataConnector& dc = Environment<>::get().DataConnector();
                auto speciesPtr = dc.get<SpeciesType>(FrameType::getName());

                DensityFunctor densityFunctor(currentStep);
                PositionFunctor positionFunctor(currentStep);
                speciesPtr->initDensityProfile(densityFunctor, positionFunctor, currentStep);
            }
        };


        /** Generate particles in a species by deriving and manipulating from another species' particles
         *
         * Create particles in `T_DestSpeciesType` by deriving (copying) all particles
         * and their matching attributes (except `particleId`) from `T_SrcSpeciesType`.
         * During the derivation, the particle attributes in can be manipulated with
         * `T_ManipulateFunctor`.
         *
         * @note FillAllGaps is called on on T_DestSpeciesType after the derivation is
         *       finished.
         *       If the derivation also manipulates the T_SrcSpeciesType, e.g. in order
         *       to deactivate some particles for a move, FillAllGaps needs to be
         *       called for the T_SrcSpeciesType manually in the next step!
         *
         * @tparam T_Manipulator a pseudo-binary functor accepting two particle species:
         *                       destination and source,
         *                       @see picongpu::particles::manipulators
         * @tparam T_SrcSpeciesType type or name as PMACC_CSTRING of the source species
         * @tparam T_DestSpeciesType type or name as PMACC_CSTRING of the destination species
         * @tparam T_SrcFilter picongpu::particles::filter, particle filter type to
         *                     select particles in T_SrcSpeciesType to derive into
         *                     T_DestSpeciesType
         */
        template<
            typename T_Manipulator,
            typename T_SrcSpeciesType,
            typename T_DestSpeciesType = boost::mpl::_1,
            typename T_SrcFilter = filter::All>
        struct ManipulateDerive
        {
            using DestSpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_DestSpeciesType>;
            using DestFrameType = typename DestSpeciesType::FrameType;
            using SrcSpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SrcSpeciesType>;
            using SrcFrameType = typename SrcSpeciesType::FrameType;

            using DestFunctor = typename boost::mpl::apply1<T_Manipulator, DestSpeciesType>::type;

            using SrcFilter = typename boost::mpl::apply1<T_SrcFilter, SrcSpeciesType>::type;

            /* note: this is a FilteredManipulator with filter::All for
             * destination species, users can filter the destination directly via if's
             * in the T_Manipulator.
             */
            using FilteredManipulator = manipulators::IBinary<DestFunctor>;
            using SrcFilterInterfaced = filter::IUnary<SrcFilter>;

            HINLINE void operator()(const uint32_t currentStep)
            {
                DataConnector& dc = Environment<>::get().DataConnector();
                auto speciesPtr = dc.get<DestSpeciesType>(DestFrameType::getName());
                auto srcSpeciesPtr = dc.get<SrcSpeciesType>(SrcFrameType::getName());

                FilteredManipulator filteredManipulator(currentStep);
                SrcFilterInterfaced srcFilter(currentStep);

                speciesPtr->deviceDeriveFrom(*srcSpeciesPtr, filteredManipulator, srcFilter);
            }
        };


        /** Generate particles in a species by deriving from another species' particles
         *
         * Create particles in `T_DestSpeciesType` by deriving (copying) all particles
         * and their matching attributes (except `particleId`) from `T_SrcSpeciesType`.
         *
         * @note FillAllGaps is called on on `T_DestSpeciesType` after the derivation is
         *       finished.
         *
         * @tparam T_SrcSpeciesType type or name as PMACC_CSTRING of the source species
         * @tparam T_DestSpeciesType type or name as PMACC_CSTRING of the destination species
         * @tparam T_Filter picongpu::particles::filter,
         *                  particle filter type to select source particles to derive
         */
        template<
            typename T_SrcSpeciesType,
            typename T_DestSpeciesType = boost::mpl::_1,
            typename T_Filter = filter::All>
        struct Derive : ManipulateDerive<manipulators::generic::None, T_SrcSpeciesType, T_DestSpeciesType, T_Filter>
        {
        };


        /** Generate a valid, contiguous list of particle frames
         *
         * Some operations, such as deactivating or adding particles to a particle
         * species can generate "gaps" in our internal particle storage, a list
         * of frames.
         *
         * This operation copies all particles from the end of the frame list to
         * "gaps" in the beginning of the frame list.
         * After execution, the requirement that all particle frames must be filled
         * contiguously with valid particles and that all frames but the last are full
         * is fulfilled.
         *
         * @tparam T_SpeciesType type or name as PMACC_CSTRING of the particle species
         *                       to fill gaps in memory
         */
        template<typename T_SpeciesType = boost::mpl::_1>
        struct FillAllGaps
        {
            using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SpeciesType>;
            using FrameType = typename SpeciesType::FrameType;

            HINLINE void operator()(const uint32_t currentStep)
            {
                DataConnector& dc = Environment<>::get().DataConnector();
                auto speciesPtr = dc.get<SpeciesType>(FrameType::getName());
                speciesPtr->fillAllGaps();
            }
        };

    } // namespace particles
} // namespace picongpu
