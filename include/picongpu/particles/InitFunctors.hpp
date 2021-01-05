/* Copyright 2014-2021 Rene Widera
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
#include <pmacc/meta/conversion/TypeToPointerPair.hpp>
#include "picongpu/particles/manipulators/manipulators.def"
#include "picongpu/particles/densityProfiles/IProfile.def"
#include "picongpu/particles/Manipulate.hpp"
#include "picongpu/particles/filter/filter.def"
#include "picongpu/particles/manipulators/manipulators.def"

#include <pmacc/Environment.hpp>
#include <pmacc/traits/Resolve.hpp>
#include <pmacc/traits/HasFlag.hpp>
#include <pmacc/traits/GetFlagType.hpp>
#include <pmacc/math/MapTuple.hpp>
#include <pmacc/particles/meta/FindByNameOrType.hpp>

#include <boost/mpl/if.hpp>
#include <boost/mpl/plus.hpp>
#include <boost/mpl/accumulate.hpp>
#include <boost/mpl/apply.hpp>
#include <boost/mpl/apply_wrap.hpp>


namespace picongpu
{
    namespace particles
    {
        /** call a functor
         *
         * @tparam T_Functor unary lambda functor
         *                   operator() must take two params
         *                      - first: storage tuple
         *                      - second: current time step
         */
        template<typename T_Functor = bmpl::_1>
        struct CallFunctor
        {
            using Functor = T_Functor;

            HINLINE void operator()(const uint32_t currentStep)
            {
                Functor()(currentStep);
            }
        };

        /** Create particle distribution from a normalized density profile
         *
         * Create particles inside a species. The created particles are macroscopically
         * distributed according to a given normalized density profile
         * (`T_DensityFunctor`). Their microscopic position inside individual cells is
         * determined by the `T_PositionFunctor`.
         *
         * @note FillAllGaps is automatically called after creation.
         *
         * @tparam T_DensityFunctor unary lambda functor with profile description,
         *                          see density.param,
         *                          example: picongpu::particles::densityProfiles::Homogenous
         * @tparam T_PositionFunctor unary lambda functor with position description,
         *                           see particle.param,
         *                           examples: picongpu::particles::startPosition::Quiet,
         *                                     picongpu::particles::startPosition::Random
         * @tparam T_SpeciesType type or name as boost::mpl::string of the used species,
         *                       see speciesDefinition.param
         */
        template<typename T_DensityFunctor, typename T_PositionFunctor, typename T_SpeciesType = bmpl::_1>
        struct CreateDensity
        {
            using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SpeciesType>;
            using FrameType = typename SpeciesType::FrameType;


            using UserDensityFunctor = typename bmpl::apply1<T_DensityFunctor, SpeciesType>::type;
            /* add interface for compile time interface validation*/
            using DensityFunctor = densityProfiles::IProfile<UserDensityFunctor>;

            using UserPositionFunctor = typename bmpl::apply1<T_PositionFunctor, SpeciesType>::type;
            /* add interface for compile time interface validation*/
            using PositionFunctor = manipulators::IUnary<UserPositionFunctor>;

            HINLINE void operator()(const uint32_t currentStep)
            {
                DataConnector& dc = Environment<>::get().DataConnector();
                auto speciesPtr = dc.get<SpeciesType>(FrameType::getName(), true);

                DensityFunctor densityFunctor(currentStep);
                PositionFunctor positionFunctor(currentStep);
                speciesPtr->initDensityProfile(densityFunctor, positionFunctor, currentStep);

                dc.releaseData(FrameType::getName());
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
         * @tparam T_SrcSpeciesType type or name as boost::mpl::string of the source species
         * @tparam T_DestSpeciesType type or name as boost::mpl::string of the destination species
         * @tparam T_SrcFilter picongpu::particles::filter, particle filter type to
         *                     select particles in T_SrcSpeciesType to derive into
         *                     T_DestSpeciesType
         */
        template<
            typename T_Manipulator,
            typename T_SrcSpeciesType,
            typename T_DestSpeciesType = bmpl::_1,
            typename T_SrcFilter = filter::All>
        struct ManipulateDerive
        {
            using DestSpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_DestSpeciesType>;
            using DestFrameType = typename DestSpeciesType::FrameType;
            using SrcSpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SrcSpeciesType>;
            using SrcFrameType = typename SrcSpeciesType::FrameType;

            using DestFunctor = typename bmpl::apply1<T_Manipulator, DestSpeciesType>::type;

            using SrcFilter = typename bmpl::apply1<T_SrcFilter, SrcSpeciesType>::type;

            /* note: this is a FilteredManipulator with filter::All for
             * destination species, users can filter the destination directly via if's
             * in the T_Manipulator.
             */
            using FilteredManipulator = manipulators::IBinary<DestFunctor>;
            using SrcFilterInterfaced = filter::IUnary<SrcFilter>;

            HINLINE void operator()(const uint32_t currentStep)
            {
                DataConnector& dc = Environment<>::get().DataConnector();
                auto speciesPtr = dc.get<DestSpeciesType>(DestFrameType::getName(), true);
                auto srcSpeciesPtr = dc.get<SrcSpeciesType>(SrcFrameType::getName(), true);

                FilteredManipulator filteredManipulator(currentStep);
                SrcFilterInterfaced srcFilter(currentStep);

                speciesPtr->deviceDeriveFrom(*srcSpeciesPtr, filteredManipulator, srcFilter);

                dc.releaseData(DestFrameType::getName());
                dc.releaseData(SrcFrameType::getName());
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
         * @tparam T_SrcSpeciesType type or name as boost::mpl::string of the source species
         * @tparam T_DestSpeciesType type or name as boost::mpl::string of the destination species
         * @tparam T_Filter picongpu::particles::filter,
         *                  particle filter type to select source particles to derive
         */
        template<typename T_SrcSpeciesType, typename T_DestSpeciesType = bmpl::_1, typename T_Filter = filter::All>
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
         * @tparam T_SpeciesType type or name as boost::mpl::string of the particle species
         *                       to fill gaps in memory
         */
        template<typename T_SpeciesType = bmpl::_1>
        struct FillAllGaps
        {
            using SpeciesType = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_SpeciesType>;
            using FrameType = typename SpeciesType::FrameType;

            HINLINE void operator()(const uint32_t currentStep)
            {
                DataConnector& dc = Environment<>::get().DataConnector();
                auto speciesPtr = dc.get<SpeciesType>(FrameType::getName(), true);
                speciesPtr->fillAllGaps();
                dc.releaseData(FrameType::getName());
            }
        };

    } // namespace particles
} // namespace picongpu
