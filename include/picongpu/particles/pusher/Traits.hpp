/*
 * SPDX-FileCopyrightText: Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
