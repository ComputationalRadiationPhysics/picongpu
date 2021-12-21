/* Copyright 2020-2021 Sergei Bastrakov
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

#include "picongpu/particles/pusher/particlePusherComposite.hpp"

#include <pmacc/traits/IsBaseTemplateOf.hpp>

#include <type_traits>


namespace picongpu
{
    namespace particles
    {
        namespace pusher
        {
            /** Check if pusher type is composite (use several underlying pushers)
             *
             * The only composite pusher types are children of
             * particlePusherComposite::Push template classes
             *
             * @tparam T_Pusher pusher type
             * @treturn ::type std::true_type or std::false_type
             */
            template<typename T_Pusher>
            struct IsComposite : public pmacc::traits::IsBaseTemplateOf_t<particlePusherComposite::Push, T_Pusher>
            {
            };

        } // namespace pusher
    } // namespace particles
} // namespace picongpu
