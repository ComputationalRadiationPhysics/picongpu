/* Copyright 2019-2024 Rene Widera, Pawel Ordyna, Filip Optolowicz
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
#include "picongpu/particles/fusion/Collider.def"
#include "picongpu/particles/fusion/WithPeer.hpp"

#include <pmacc/meta/ForEach.hpp>
#include <pmacc/meta/accessors/First.hpp>
#include <pmacc/meta/accessors/Second.hpp>
#include <pmacc/meta/conversion/ApplyGuard.hpp>
#include <pmacc/meta/conversion/ToSeq.hpp>

namespace picongpu::particles::fusion
{
    namespace detail
    {
        // "For each" implementation for calling a collider for each species pair in a list of reactants
        template<
            typename T_SpeciesPairListReactants,
            typename T_SpeciesPairListProducts,
            typename T_Collider,
            uint32_t colliderId>
        struct CallColliderForAPair
        {
            template<size_t... I>
            HINLINE void operator()(
                std::index_sequence<I...>,
                std::shared_ptr<DeviceHeap> const& deviceHeap,
                uint32_t currentStep)
            {
                (fusion::WithPeer<
                     typename T_Collider::Functor,
                     typename pmacc::mp_at_c<T_SpeciesPairListReactants, I>::first,
                     typename pmacc::mp_at_c<T_SpeciesPairListReactants, I>::second,
                     typename pmacc::mp_at_c<T_SpeciesPairListProducts, I>::first,
                     typename pmacc::mp_at_c<T_SpeciesPairListProducts, I>::second,
                     typename T_Collider::FilterPair,
                     colliderId,
                     I>{}(deviceHeap, currentStep),
                 ...);
            }
        };
    } // namespace detail

    template<typename T_Collider, uint32_t colliderId>
    struct CallCollider
    {
        void operator()(std::shared_ptr<DeviceHeap> const& deviceHeap, uint32_t currentStep)
        {
            using SpeciesPairListReactants = pmacc::ToSeq<typename T_Collider::SpeciesPairsReactants>;
            using SpeciesPairListProducts = pmacc::ToSeq<typename T_Collider::SpeciesPairsProducts>;
            constexpr size_t numPairs = pmacc::mp_size<SpeciesPairListReactants>::value;
            std::make_index_sequence<numPairs> index{};
            detail::CallColliderForAPair<SpeciesPairListReactants, SpeciesPairListProducts, T_Collider, colliderId>{}(
                index,
                deviceHeap,
                currentStep);
        }
    };
} // namespace picongpu::particles::fusion
