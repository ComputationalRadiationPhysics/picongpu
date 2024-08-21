/* Copyright 2013-2023 Rene Widera, Axel Huebl
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

#include "picongpu/simulation_defines.hpp"

#include <type_traits>
#include <utility>


namespace picongpu
{
    namespace particles
    {
        namespace functor
        {
            template<typename T_Functor>
            struct User : public T_Functor
            {
                using Functor = T_Functor;

                /** constructor
                 *
                 * This constructor is only compiled if the user functor has
                 * a host side constructor with one (uint32_t) argument.
                 *
                 * @tparam DeferFunctor is used to defer the functor type evaluation to enable/disable
                 *                      the constructor
                 * @param currentStep current simulation time step
                 * @param is used to enable/disable the constructor (do not pass any value to this parameter)
                 */
                template<typename DeferFunctor = Functor>
                HINLINE User(
                    uint32_t currentStep,
                    IdGenerator,
                    std::enable_if_t<
                        !std::is_default_constructible_v<
                            DeferFunctor> && std::is_constructible_v<DeferFunctor, uint32_t>>* = 0)
                    : Functor(currentStep)
                {
                }

                template<typename DeferFunctor = Functor>
                HINLINE User(
                    uint32_t currentStep,
                    IdGenerator idGen,
                    std::enable_if_t<
                        !std::is_default_constructible_v<
                            DeferFunctor> && std::is_constructible_v<DeferFunctor, uint32_t, IdGenerator>>* = 0)
                    : Functor(currentStep, idGen)
                {
                }

                /** constructor
                 *
                 * This constructor is only compiled if the user functor has a default constructor.
                 *
                 * @tparam DeferFunctor is used to defer the functor type evaluation to enable/disable
                 *                      the constructor
                 * @param current simulation time step
                 * @param is used to enable/disable the constructor (do not pass any value to this parameter)
                 */
                template<typename DeferFunctor = Functor>
                HINLINE User(
                    uint32_t,
                    IdGenerator,
                    std::enable_if_t<std::is_default_constructible_v<DeferFunctor>>* = nullptr)
                    : Functor()
                {
                }
            };
        } // namespace functor
    } // namespace particles
} // namespace picongpu
