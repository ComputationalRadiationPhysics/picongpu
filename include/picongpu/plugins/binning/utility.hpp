/* Copyright 2023 Tapish Narwal
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

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/memory/STLTuple.hpp>

#include <tuple>
#include <utility>

namespace picongpu
{
    namespace plugins::binning
    {
        namespace detail
        {
            template<typename TFunc, typename... TArgs, std::size_t... Is>
            HDINLINE constexpr auto apply_impl(
                TFunc&& f,
                pmacc::memory::tuple::Tuple<TArgs...>&& t,
                std::index_sequence<Is...>)
            {
                // @todo give Is as a param to f, so compiler has knowledge
                return f(pmacc::memory::tuple::get<Is>(std::forward<pmacc::memory::tuple::Tuple<TArgs...>&&>(t))...);
            }
        } // namespace detail

        template<typename TFunc, typename... TArgs>
        HDINLINE constexpr auto apply(TFunc&& f, pmacc::memory::tuple::Tuple<TArgs...> t)
        {
            return detail::apply_impl(
                std::forward<TFunc>(f),
                std::forward<pmacc::memory::tuple::Tuple<TArgs...>&&>(t),
                std::make_index_sequence<sizeof...(TArgs)>{});
        }

        namespace detail
        {
            template<size_t... Is, typename... Args, typename Functor>
            constexpr auto tupleMapHelper(
                std::index_sequence<Is...>,
                const std::tuple<Args...>& tuple,
                const Functor& functor)
            {
                return pmacc::memory::tuple::make_tuple(functor(std::get<Is>(tuple))...);
            }
        } // namespace detail

        /**
         * @brief create an alpaka tuple from a standard tuple by applying a functor
         */
        template<typename... Args, typename Functor>
        constexpr auto tupleMap(const std::tuple<Args...>& tuple, const Functor& functor)
        {
            return detail::tupleMapHelper(std::make_index_sequence<sizeof...(Args)>{}, tuple, functor);
        }

        template<typename... Args>
        HDINLINE auto createTuple(Args const&... args)
        {
            return std::make_tuple(args...);
        }

    } // namespace plugins::binning
} // namespace picongpu
