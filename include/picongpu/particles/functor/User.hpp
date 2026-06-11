/*
 * SPDX-FileCopyrightText: Rene Widera, Axel Huebl
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"

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
                        !std::is_default_constructible_v<DeferFunctor>
                        && std::is_constructible_v<DeferFunctor, uint32_t>>* = 0)
                    : Functor(currentStep)
                {
                }

                template<typename DeferFunctor = Functor>
                HINLINE User(
                    uint32_t currentStep,
                    IdGenerator idGen,
                    std::enable_if_t<
                        !std::is_default_constructible_v<DeferFunctor>
                        && std::is_constructible_v<DeferFunctor, uint32_t, IdGenerator>>* = 0)
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
