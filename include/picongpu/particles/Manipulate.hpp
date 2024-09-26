/* Copyright 2014-2023 Rene Widera, Sergei Bastrakov
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/Manipulate.def"
#include "picongpu/particles/filter/filter.def"
#include "picongpu/particles/manipulators/manipulators.def"
#include "picongpu/particles/param.hpp"

#include <pmacc/meta/ForEach.hpp>
#include <pmacc/meta/conversion/ToSeq.hpp>
#include <pmacc/particles/algorithm/CallForEach.hpp>
#include <pmacc/particles/meta/FindByNameOrType.hpp>

#include <boost/mpl/placeholders.hpp>

#include <cstdint>
#include <type_traits>


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
                using SpeciesFunctor = typename boost::mpl::apply1<T_Manipulator, Species>::type;
                using ParticleFilter = typename boost::mpl::apply1<T_Filter, Species>::type;

            public:
                using type = manipulators::IUnary<SpeciesFunctor, ParticleFilter>;
            };
        } // namespace detail

        /** Run a user defined manipulation for each particle of a species in an area
         *
         * Allows to manipulate attributes of existing particles in a species with
         * arbitrary unary functors ("manipulators").
         *
         * Provides two versions of operator() to either operate on T_Area or a custom area,
         * @see pmacc::particles::algorithm::CallForEach.
         *
         * @warning Does NOT call FillAllGaps after manipulation! If the
         *          manipulation deactivates particles or creates "gaps" in any
         *          other way, FillAllGaps needs to be called for the
         *          `T_Species` manually in the next step!
         *
         * @tparam T_Manipulator unary lambda functor accepting one particle
         *                       species,
         *                       @see picongpu::particles::manipulators
         * @tparam T_Species type or name as PMACC_CSTRING of the used species
         * @tparam T_Filter picongpu::particles::filter, particle filter type to
         *                  select particles in `T_Species` to manipulate
         * @tparam T_Area area to process particles in operator()(currentStep),
         *                wrapped into std::integral_constant for boost::mpl::apply to work;
         *                does not affect operator()(currentStep, areaMapperFactory)
         */
        template<
            typename T_Manipulator,
            typename T_Species = boost::mpl::_1,
            typename T_Filter = filter::All,
            typename T_Area = std::integral_constant<uint32_t, CORE + BORDER>>
        struct Manipulate
            : public pmacc::particles::algorithm::CallForEach<
                  pmacc::particles::meta::FindByNameOrType<VectorAllSpecies, T_Species>,
                  detail::MakeUnaryFilteredFunctor<T_Manipulator, T_Species, T_Filter>,
                  T_Area::value>
        {
        };

        template<typename T_Manipulator, typename T_Species, typename T_Filter, uint32_t T_area>
        void manipulate(uint32_t const currentStep)
        {
            using SpeciesSeq = pmacc::ToSeq<T_Species>;
            using Functor
                = Manipulate<T_Manipulator, boost::mpl::_1, T_Filter, std::integral_constant<uint32_t, T_area>>;
            pmacc::meta::ForEach<SpeciesSeq, Functor> forEach;
            forEach(currentStep);
        }

        template<typename T_Manipulator, typename T_Species, typename T_AreaMapperFactory, typename T_Filter>
        void manipulate(uint32_t const currentStep, T_AreaMapperFactory const& areaMapperFactory)
        {
            using SpeciesSeq = pmacc::ToSeq<T_Species>;
            using Functor = Manipulate<T_Manipulator, boost::mpl::_1, T_Filter>;
            pmacc::meta::ForEach<SpeciesSeq, Functor> forEach;
            forEach(currentStep, areaMapperFactory);
        }

        /** @} */

    } // namespace particles
} // namespace picongpu
