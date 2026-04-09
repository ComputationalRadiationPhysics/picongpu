/***************************************************************************
 *
 *  Copyright (C) 2018 Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a copy of the License has been included in this
 *  repository.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 * STLTuple.h
 *
 * @brief:
 *  Minimal implementation of std::tuple that is standard layout.
 *
  Authors:
 *
 *    Mehdi Goli    Codeplay Software Ltd.
 *    Ralph Potter  Codeplay Software Ltd.
 *    Luke Iwanski  Codeplay Software Ltd.
 *
 * This file has been modified by Tapish Narwal, adding forward as tuple,
 * tie, passing paramters by forwarding, and using stl instead of
 * self-written metafunctions.
 **************************************************************************/

#pragma once

#include "pmacc/attribute/FunctionSpecifier.hpp"

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/core/Common.hpp>

#include <type_traits>
#include <utility>

// suppress warnings as this is third-party code
#if BOOST_COMP_CLANG
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdocumentation"
#    pragma clang diagnostic ignored "-Wdocumentation-unknown-command"
#endif
#if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
#    pragma warning(push)
#    pragma warning(disable : 4003) // not enough arguments for function-like macro invocation
#endif

namespace pmacc
{
    namespace memory
    {
        namespace tuple
        {
            /// @struct Tuple
            /// @brief A fixed-size collection of heterogeneous values, implemented using recursive templates.
            /// This structure allows accessing elements by index, and supports various utilities like concatenation
            /// and removal of types.
            /// @tparam Ts...  Types of the elements that the tuple stores. Empty list is supported.
            template<typename... Ts>
            struct Tuple;

            /// @brief Specialization of the @ref Tuple class when the tuple has at least one element.
            /// This recursive structure stores the first element (`head`) and delegates the rest to the `tail`.
            /// @tparam T The type of the first element in the tuple.
            /// @tparam Ts... The types of the remaining elements in the tuple.
            template<typename T, typename... Ts>
            struct Tuple<T, Ts...>
            {
                template<typename U, typename... Us>
                requires(!std::same_as<std::remove_cvref_t<U>, Tuple> && sizeof...(Us) == sizeof...(Ts))
                HDINLINE constexpr Tuple(U&& u, Us&&... us) noexcept
                    : head(std::forward<U>(u))
                    , tail(std::forward<Us>(us)...)
                {
                }

                HDINLINE constexpr Tuple(Tuple const&) noexcept = default;
                HDINLINE constexpr Tuple(Tuple&&) noexcept = default;
                HDINLINE constexpr Tuple& operator=(Tuple const&) noexcept = default;
                HDINLINE constexpr Tuple& operator=(Tuple&&) noexcept = default;

                T head;
                Tuple<Ts...> tail;
            };

            // Base case for empty tuple
            template<>
            struct Tuple<>
            {
            };

            // Deduction guide for Tuple to contruct with values
            template<typename T, typename... Ts>
            Tuple(T&&, Ts&&...) -> Tuple<std::remove_cvref_t<T>, std::remove_cvref_t<Ts>...>;

            /// @brief Extracts the Kth element from the tuple.
            /// @tparam K The index of the element to extract (0-based).
            /// @tparam T The type of the (sizeof...(Types) -(K+1))th element.
            /// @tparam Ts... The types of the elements in the tuple.
            /// @param t The tuple from which to extract the element.
            /// @return The extracted element by reference.
            template<size_t k, typename T, typename... Ts>
            HDINLINE constexpr decltype(auto) get(Tuple<T, Ts...>& t)
            {
                if constexpr(k == 0)
                {
                    return t.head;
                }
                else
                {
                    return get<k - 1>(t.tail);
                }
            }

            /// Const version of `get`
            template<size_t k, typename T, typename... Ts>
            HDINLINE constexpr decltype(auto) get(Tuple<T, Ts...> const& t)
            {
                if constexpr(k == 0)
                {
                    return t.head;
                }
                else
                {
                    return get<k - 1>(t.tail);
                }
            }

            /// @brief Creates a tuple object by forwarding the provided arguments.
            /// The types of the arguments are deduced from the argument types.
            /// @tparam Args The types of the arguments to construct the tuple from.
            /// @param args The arguments to construct the tuple.
            /// @return A tuple containing the provided arguments.
            template<typename... Args>
            HDINLINE constexpr auto make_tuple(Args&&... args)
            {
                return Tuple<std::remove_cvref_t<Args>...>(std::forward<Args>(args)...);
            }

            /// @brief Creates a tuple of forwarding references to the provided arguments.
            /// This ensures the value category (lvalue/rvalue) of each argument is preserved.
            /// @tparam Args Types of the arguments to construct the tuple from.
            /// @param args Zero or more arguments to construct the tuple.
            /// @return A tuple of forwarding references to the provided arguments.
            template<typename... Args>
            HDINLINE constexpr auto forward_as_tuple(Args&&... args)
            {
                return Tuple<Args&&...>(std::forward<Args>(args)...);
            }

            /// @brief Creates a tuple of references to the provided variables.
            /// This is similar to std::tie but for the custom Tuple class.
            /// @tparam Args Types of the arguments to construct the tuple from.
            /// @param args Variables to create the tuple of references from.
            /// @return Tuple<Args&...> A tuple of references to the provided variables.
            template<typename... Args>
            HDINLINE constexpr auto tie(Args&... args)
            {
                return Tuple<Args&...>(args...);
            }

            /// @brief Creates a tuple of references to the provided const variables.
            /// This is a const reference version for constant variables.
            /// @tparam Args Types of the arguments to construct the tuple from.
            /// @param args Variables to create the tuple of references from.
            /// @return Tuple<Args&...> A tuple of references to the provided const variables.
            template<typename... Args>
            HDINLINE constexpr auto tie(Args const&... args)
            {
                return Tuple<Args const&...>(args...);
            }

            /// @struct tuple_size
            /// @brief A helper structure to get the size of a tuple. Handles cv-qualifiers and refs
            template<typename T>
            struct tuple_size : tuple_size<std::remove_cvref_t<T>>
            {
            };

            /// Specialization of `tuple_size` for the `Tuple` type.
            template<typename... Args>
            struct tuple_size<Tuple<Args...>> : std::integral_constant<std::size_t, sizeof...(Args)>
            {
            };

            /// Helper variable template to get the size of a tuple.
            template<typename T>
            constexpr std::size_t tuple_size_v = tuple_size<T>::value;

            template<typename Tuple, typename T, std::size_t... Is>
            HDINLINE constexpr auto append_base(Tuple&& t, T&& a, std::index_sequence<Is...>)
            {
                return make_tuple(get<Is>(std::forward<Tuple>(t))..., std::forward<T>(a));
            }

            /// @brief A function to append a new element to the end of a tuple.
            /// @tparam Args... The types of the elements inside the tuple.
            /// @tparam T The type of the new element to append.
            /// @param t The tuple to which the new element will be added.
            /// @param a The new element to add.
            /// @return A new tuple containing all elements from the original tuple followed by the new element.
            template<typename... Args, typename T>
            HDINLINE constexpr auto append(Tuple<Args...>& t, T&& a)
            {
                return append_base(t, std::forward<T>(a), std::make_index_sequence<sizeof...(Args)>{});
            }

            template<typename... Args1, typename... Args2, std::size_t... Is1, std::size_t... Is2>
            HDINLINE constexpr auto append_base(
                Tuple<Args1...>& t1,
                Tuple<Args2...>& t2,
                std::index_sequence<Is1...>,
                std::index_sequence<Is2...>)
            {
                return make_tuple(get<Is1>(t1)..., get<Is2>(t2)...);
            }

            /// @brief Concatenates two tuples into one tuple.
            /// @tparam Args1... The types of the elements inside the first tuple.
            /// @tparam Args2... The types of the elements inside the second tuple.
            /// @param t1 The first tuple to append.
            /// @param t2 The second tuple to append.
            /// @return A new tuple that contains all elements from both input tuples.
            template<typename... Args1, typename... Args2>
            HDINLINE constexpr auto append(Tuple<Args1...>& t1, Tuple<Args2...>& t2)
            {
                return append_base(
                    t1,
                    t2,
                    std::make_index_sequence<sizeof...(Args1)>{},
                    std::make_index_sequence<sizeof...(Args2)>{});
            }

        } // namespace tuple
    } // namespace memory
} // namespace pmacc

#if BOOST_COMP_CLANG
#    pragma clang diagnostic pop
#endif
#if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
#    pragma warning(pop)
#endif
