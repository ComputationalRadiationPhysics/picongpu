/* Copyright 2019-2021 Rene Widera, Pawel Ordyna
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
            template<typename T_Collider>
            struct CallCollider
            {
                void operator()(std::shared_ptr<DeviceHeap> const& deviceHeap, uint32_t currentStep)
                {
                    using SpeciesPairList = typename pmacc::ToSeq<typename T_Collider::SpeciesPairs>::type;

                    pmacc::meta::ForEach<
                        SpeciesPairList,
                        WithPeer<
                            ApplyGuard<typename T_Collider::Functor>,
                            pmacc::meta::accessors::First<bmpl::_1>,
                            pmacc::meta::accessors::Second<bmpl::_1>,
                            typename T_Collider::Params,
                            typename T_Collider::FilterPair>>{}(deviceHeap, currentStep);
                }
            };
        } // namespace collision
    } // namespace particles
} // namespace picongpu
