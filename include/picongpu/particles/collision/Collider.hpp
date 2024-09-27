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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/collision/Collider.def"
#include "picongpu/particles/collision/WithPeer.hpp"

#include <pmacc/meta/ForEach.hpp>
#include <pmacc/meta/accessors/First.hpp>
#include <pmacc/meta/accessors/Second.hpp>
#include <pmacc/meta/conversion/ApplyGuard.hpp>
#include <pmacc/meta/conversion/ToSeq.hpp>


namespace picongpu
{
    namespace particles
    {
        namespace collision
        {
            namespace detail
            {
                //! For each implementation for calling a collider for each species pair with a loop index
                template<typename T_SpeciesPairList, typename T_Collider, uint32_t colliderId>
                struct CallColliderForAPair
                {
                    template<size_t... I>
                    HINLINE void operator()(
                        std::index_sequence<I...>,
                        std::shared_ptr<DeviceHeap> const& deviceHeap,
                        uint32_t currentStep)
                    {
                        (collision::WithPeer<
                             typename T_Collider::Functor,
                             typename pmacc::mp_at_c<T_SpeciesPairList, I>::first,
                             typename pmacc::mp_at_c<T_SpeciesPairList, I>::second,
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
                    using SpeciesPairList = pmacc::ToSeq<typename T_Collider::SpeciesPairs>;
                    constexpr size_t numPairs = pmacc::mp_size<SpeciesPairList>::value;
                    std::make_index_sequence<numPairs> index{};
                    detail::CallColliderForAPair<SpeciesPairList, T_Collider, colliderId>{}(
                        index,
                        deviceHeap,
                        currentStep);
                }
            };
        } // namespace collision
    } // namespace particles
} // namespace picongpu
