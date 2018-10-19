/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/core/Common.hpp>

#include <alpaka/meta/IntegerSequence.hpp>

#if !BOOST_ARCH_CUDA_DEVICE
    #include <boost/core/ignore_unused.hpp>
#endif
#include <boost/config.hpp>

#include <utility>
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
#if !BOOST_ARCH_CUDA_DEVICE
                boost::ignore_unused(t);
#endif
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
