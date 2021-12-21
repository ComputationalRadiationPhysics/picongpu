/* Copyright 2013-2021 Rene Widera, Benjamin Worpitz, Alexander Grund
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <utility>

namespace pmacc
{
    namespace meta
    {
        namespace detail
        {
            /** Conditional functor execution
             *
             * @tparam T_condition condition for executing a functor
             */
            template<bool T_condition>
            struct IfElse;

            /** Specialization if condition is true
             *
             * Execute trueFunctor passed to operator()
             */
            template<>
            struct IfElse<true>
            {
                PMACC_NO_NVCC_HDWARNING
                template<typename T_ConditionTrueFunctor, typename T_ConditionFalseFunctor, typename... T_Args>
                HDINLINE void operator()(
                    T_ConditionTrueFunctor&& trueFunctor,
                    T_ConditionFalseFunctor&&,
                    T_Args&&... args) const
                {
                    trueFunctor(std::forward<T_Args>(args)...);
                }
            };

            /** Specialization if condition is false
             *
             * Execute falseFunctor passed to operator()
             */
            template<>
            struct IfElse<false>
            {
                PMACC_NO_NVCC_HDWARNING
                template<typename T_ConditionTrueFunctor, typename T_ConditionFalseFunctor, typename... T_Args>
                HDINLINE void operator()(
                    T_ConditionTrueFunctor&&,
                    T_ConditionFalseFunctor&& falseFunctor,
                    T_Args&&... args) const
                {
                    falseFunctor(std::forward<T_Args>(args)...);
                }
            };
        } // namespace detail

        /** Conditional functor invocation
         *
         * Evaluate and invoke a functor based on the conditional compile-time argument.
         * The functor implementations must have valid C++ syntax but is only evaluated on invocation.
         *
         * @tparam T_condition invoke trueFunctor if the condition is true, else no operation is performed
         * @tparam T_ConditionTrueFunctor functor type, the functor is only evaluated if T_condition is true
         * @tparam T_Args types of the arguments forwarded to the trueFunctor
         * @param trueFunctor functor instance which is called with the arguments args if T_condition is true
         * @param args arguments forwarded to the functor
         */
        PMACC_NO_NVCC_HDWARNING
        template<bool T_condition, typename T_ConditionTrueFunctor, typename... T_Args>
        HDINLINE void invokeIf(T_ConditionTrueFunctor&& trueFunctor, T_Args&&... args)
        {
            detail::IfElse<T_condition>{}(
                trueFunctor,
                /* T_Args instead of auto for arguments is required to avoid a nvcc compiler bug shown when compiling
                 * for A100. https://github.com/ComputationalRadiationPhysics/picongpu/pull/3714#issuecomment-914349936
                 */
                [](T_Args&&...) {},
                std::forward<T_Args>(args)...);
        }

        /** Conditional functor invocation
         *
         * Evaluate and invoke a functor based on the conditional compile-time argument.
         * The functor implementations must have valid C++ syntax but is only evaluated on invocation.
         *
         * @tparam T_condition invoke trueFunctor if the condition is true, else falseFunctor
         * @tparam T_ConditionTrueFunctor functor type, the functor is only evaluated if T_condition is true
         * @tparam T_ConditionFalseFunctor functor type, the functor is only evaluated if T_condition is false
         * @tparam T_Args types of the arguments forwarded to the functor
         * @param trueFunctor functor instance which is called with the arguments args if T_condition is true
         * @param falseFunctor functor instance which is called with the arguments args if T_condition is false
         * @param args arguments forwarded to the functor
         */
        PMACC_NO_NVCC_HDWARNING
        template<
            bool T_condition,
            typename T_ConditionTrueFunctor,
            typename T_ConditionFalseFunctor,
            typename... T_Args>
        HDINLINE void invokeIfElse(
            T_ConditionTrueFunctor&& trueFunctor,
            T_ConditionFalseFunctor&& falseFunctor,
            T_Args&&... args)
        {
            detail::IfElse<T_condition>{}(trueFunctor, falseFunctor, std::forward<T_Args>(args)...);
        }

    } // namespace meta
} // namespace pmacc
