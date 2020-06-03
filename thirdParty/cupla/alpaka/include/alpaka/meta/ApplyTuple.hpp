/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Unused.hpp>

#include <alpaka/meta/IntegerSequence.hpp>

#include <boost/config.hpp>

#include <utility>
#include <tuple>
#include <type_traits>

namespace alpaka
{
    namespace meta
    {
        //-----------------------------------------------------------------------------
        // C++17 std::invoke
        namespace detail
        {
            template<class F, class... Args>
            inline auto invoke_impl(F && f, Args &&... args)
            -> decltype(std::forward<F>(f)(std::forward<Args>(args)...))
            {
                return std::forward<F>(f)(std::forward<Args>(args)...);
            }

            template<class Base, class T, class Derived>
            inline auto invoke_impl(T Base::*pmd, Derived && ref)
            -> decltype(std::forward<Derived>(ref).*pmd)
            {
                return std::forward<Derived>(ref).*pmd;
            }

            template<class PMD, class Pointer>
            inline auto invoke_impl(PMD pmd, Pointer && ptr)
            -> decltype((*std::forward<Pointer>(ptr)).*pmd)
            {
                return (*std::forward<Pointer>(ptr)).*pmd;
            }

            template<class Base, class T, class Derived, class... Args>
            inline auto invoke_impl(T Base::*pmf, Derived && ref, Args &&... args)
            -> decltype((std::forward<Derived>(ref).*pmf)(std::forward<Args>(args)...))
            {
                return (std::forward<Derived>(ref).*pmf)(std::forward<Args>(args)...);
            }

            template<class PMF, class Pointer, class... Args>
            inline auto invoke_impl(PMF pmf, Pointer && ptr, Args &&... args)
            -> decltype(((*std::forward<Pointer>(ptr)).*pmf)(std::forward<Args>(args)...))
            {
                return ((*std::forward<Pointer>(ptr)).*pmf)(std::forward<Args>(args)...);
            }
        }

        template< class F, class... ArgTypes>
        auto invoke(F && f, ArgTypes &&... args)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(detail::invoke_impl(std::forward<F>(f), std::forward<ArgTypes>(args)...))
#endif
        {
            return detail::invoke_impl(std::forward<F>(f), std::forward<ArgTypes>(args)...);
        }

        //-----------------------------------------------------------------------------
        // C++17 std::apply
        namespace detail
        {
            template<class F, class Tuple, std::size_t... I>
            auto apply_impl( F && f, Tuple && t, meta::IndexSequence<I...> )
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
            -> decltype(
                meta::invoke(
                    std::forward<F>(f),
                    std::get<I>(std::forward<Tuple>(t))...))
#endif
            {
                // If the the index sequence is empty, t will not be used at all.
                alpaka::ignore_unused(t);

                return
                    meta::invoke(
                        std::forward<F>(f),
                        std::get<I>(std::forward<Tuple>(t))...);
            }
        }

        template<class F, class Tuple>
        auto apply(F && f, Tuple && t)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(
            detail::apply_impl(
                std::forward<F>(f),
                std::forward<Tuple>(t),
                meta::MakeIndexSequence<std::tuple_size<typename std::decay<Tuple>::type>::value>{}))
#endif
        {
            return
                detail::apply_impl(
                    std::forward<F>(f),
                    std::forward<Tuple>(t),
                    meta::MakeIndexSequence<std::tuple_size<typename std::decay<Tuple>::type>::value>{});
        }
    }
}
