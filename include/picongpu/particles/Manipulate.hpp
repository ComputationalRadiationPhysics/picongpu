/* Copyright 2014-2021 Rene Widera, Sergei Bastrakov
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
#include "picongpu/particles/manipulators/manipulators.def"
#include "picongpu/particles/filter/filter.def"

#include <pmacc/particles/algorithm/CallForEach.hpp>
#include <pmacc/meta/conversion/ToSeq.hpp>
#include <pmacc/meta/ForEach.hpp>
#include <pmacc/particles/meta/FindByNameOrType.hpp>
#include <boost/mpl/placeholders.hpp>


namespace picongpu
{
    namespace particles
    {
        namespace detail
        {
            /** Operator to create a filtered functor
             */
            template<typename T_Manipulator, typename T_Species, typename T_Filter>
            struct MakeUnaryFilteredFunctor
            {
            private:
                using Species = pmacc::particles::meta::FindByNameOrType_t<VectorAllSpecies, T_Species>;
                using SpeciesFunctor = typename bmpl::apply1<T_Manipulator, Species>::type;
                using ParticleFilter = typename bmpl::apply1<T_Filter, Species>::type;

            public:
                using type = manipulators::IUnary<SpeciesFunctor, ParticleFilter>;
            };
        } // namespace detail

        /** Run a user defined manipulation for each particle of a species
         *
         * Allows to manipulate attributes of existing particles in a species with
         * arbitrary unary functors ("manipulators").
         *
         * @warning Does NOT call FillAllGaps after manipulation! If the
         *          manipulation deactivates particles or creates "gaps" in any
         *          other way, FillAllGaps needs to be called for the
         *          `T_Species` manually in the next step!
         *
         * @tparam T_Manipulator unary lambda functor accepting one particle
         *                       species,
         *                       @see picongpu::particles::manipulators
         * @tparam T_Species type or name as boost::mpl::string of the used species
         * @tparam T_Filter picongpu::particles::filter, particle filter type to
         *                  select particles in `T_Species` to manipulate
         */
        template<typename T_Manipulator, typename T_Species = bmpl::_1, typename T_Filter = filter::All>
        struct Manipulate
            : public pmacc::particles::algorithm::CallForEach<
                  pmacc::particles::meta::FindByNameOrType<VectorAllSpecies, T_Species>,
                  detail::MakeUnaryFilteredFunctor<T_Manipulator, T_Species, T_Filter>>
        {
        };


        /** Apply a manipulation for each particle of a species or a sequence of
         *  species
         *
         * This function provides a high-level interface to particle manipulation
         * from simulation stages and plugins, but not .param files. The common
         * workflow is as follows:
         * - select the species to manipulate, often by filtering VectorAllSpecies
         * - define a manipulator type; in case the manipulator has a species type
         * as a template parameter, use the bmpl::_1 placeholder instead
         * - define a filter type when necessary
         * - call manipulate()
         *
         * This is a function-style wrapper around creating a Manipulate object and
         * calling its operator(). Unlike Manipulate, it supports both single
         * species and sequences of species.
         *
         * @tparam T_Manipulator unary lambda functor accepting one particle
         *                       species, @see picongpu::particles::manipulators
         * @tparam T_Species a single species or a sequence of species; in both
         *                   cases each species is defined by a type or a name
         * @tparam T_Filter picongpu::particles::filter, particle filter type to
         *                  select particles in `T_Species` to manipulate via
         *                  `T_DestSpeciesType`
         *
         * @param currentStep index of the current time iteration
         */
        template<typename T_Manipulator, typename T_Species, typename T_Filter = filter::All>
        inline void manipulate(uint32_t const currentStep)
        {
            using SpeciesSeq = typename pmacc::ToSeq<T_Species>::type;
            using Functor = Manipulate<T_Manipulator, bmpl::_1, T_Filter>;
            pmacc::meta::ForEach<SpeciesSeq, Functor> forEach;
            forEach(currentStep);
        }
    } // namespace particles
} // namespace picongpu
