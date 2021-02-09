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

#include "picongpu/simulation_defines.hpp"

#include <pmacc/math/vector/compile-time/Vector.hpp>

#include <cstdint>
#include <string>


namespace picongpu
{
    namespace particlePusherComposite
    {
        /** Concept for an activation functor for a composite pusher
         *
         * This concept defines an interface for the corresponding template
         * argument. This class is not supposed to be used directly.
         * However, a helper activator class to be reused is provided below.
         */
        struct ActivationFunctor
        {
            /** Return a 1-based index of which pusher of the composite to use
             *
             * Return value out of the range [1, #pushers] means no pusher to be used.
             *
             * @param currentStep current time iteration
             */
            HDINLINE uint32_t operator()(uint32_t const currentStep) const;
        };

        /** Helper activation functor for a composite of two pushers
         *
         * Uses the first pusher for currentStep < T_switchTimeStep and the second
         * one otherwise.
         */
        template<uint32_t T_switchTimeStep>
        struct BinarySwitchActivationFunctor
        {
            HDINLINE constexpr uint32_t operator()(uint32_t const currentStep) const
            {
                return currentStep < T_switchTimeStep ? 1 : 2;
            }
        };

        /** Composite of two particle pushers, each implementing the pusher concept.
         *
         * The decision which pusher to use is made by the activation functor.
         * The composite pushers implement the pusher concept themselves, however
         * for performance reasons special treatment is recommended during the
         * particle push simulation stage.
         *
         * @tparam T_FirstPusher first pusher type
         * @tparam T_SecondPusher second pusher type
         * @tparam T_ActivationFunctor activation functor to decide which pusher to use,
         *                             implements the ActivationFunctor concept
         */
        template<typename T_FirstPusher, typename T_SecondPusher, typename T_ActivationFunctor>
        struct Push
            : public T_FirstPusher
            , T_SecondPusher
        {
            using FirstPusher = T_FirstPusher;
            using SecondPusher = T_SecondPusher;
            using ActivationFunctor = T_ActivationFunctor;

            /* These are done logically correct, but should not be used directly for
             * the particle push stage.
             */
            using LowerMargin = typename pmacc::math::CT::max<
                typename traits::GetLowerMargin<FirstPusher>::type,
                typename traits::GetLowerMargin<SecondPusher>::type>::type;
            using UpperMargin = typename pmacc::math::CT::
                max<typename GetUpperMargin<FirstPusher>::type, typename GetUpperMargin<SecondPusher>::type>::type;

            /** Get active pusher 1-based index
             *
             * Result other than 1 or 2 means no pusher should be used
             *
             * @param currentStep current time iteration
             */
            static HDINLINE uint32_t activePusherIdx(uint32_t const currentStep)
            {
                return ActivationFunctor{}(currentStep);
            }

            /** Push one particle, this is compatibility-only
             *
             * Should not be used for the particle push stage due to shared memory
             * and register consumption.
             */
            template<typename T_FunctorFieldE, typename T_FunctorFieldB, typename T_Particle, typename T_Pos>
            HDINLINE void operator()(
                T_FunctorFieldB const functorBField,
                T_FunctorFieldE const functorEField,
                T_Particle& particle,
                T_Pos& pos,
                uint32_t const currentStep) const
            {
                auto const pusherIdx = activePusherIdx(currentStep);
                if(pusherIdx == 1)
                    FirstPusher::operator()(functorBField, functorEField, particle, pos, currentStep);
                else if(pusherIdx == 2)
                    SecondPusher::operator()(functorBField, functorEField, particle, pos, currentStep);
            }

            static pmacc::traits::StringProperty getStringProperties()
            {
                auto firstProperty = FirstPusher::getStringProperties();
                auto secondProperty = SecondPusher::getStringProperties();
                pmacc::traits::StringProperty propList(
                    "name",
                    std::string("Composite of ") + firstProperty["name"].value + " and "
                        + secondProperty["name"].value);
                return propList;
            }
        };

    } // namespace particlePusherComposite
} // namespace picongpu
