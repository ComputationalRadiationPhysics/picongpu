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

#include <boost/config.hpp>

#if !BOOST_ARCH_CUDA_DEVICE
    #include <boost/core/ignore_unused.hpp>
#endif

#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
    #include <type_traits>
#endif

namespace alpaka
{
    namespace meta
    {
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TFnObj,
            typename T>
        ALPAKA_FN_HOST_ACC auto foldr(
#if BOOST_ARCH_CUDA_DEVICE
            TFnObj const &,
#else
            TFnObj const & f,
#endif
            T const & t)
        -> T
        {
#if !BOOST_ARCH_CUDA_DEVICE
            boost::ignore_unused(f);
#endif
            return t;
        }
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        namespace detail
        {
            //#############################################################################
            template<
                typename TFnObj,
                typename... T>
            struct TypeOfFold;
            //#############################################################################
            template<
                typename TFnObj,
                typename T>
            struct TypeOfFold<
                TFnObj,
                T>
            {
                using type = T;
            };
            //#############################################################################
            template<
                typename TFnObj,
                typename T,
                typename... P>
            struct TypeOfFold<
                TFnObj,
                T,
                P...>
            {
                using type =
                    typename std::result_of<
                        TFnObj(T, typename TypeOfFold<TFnObj, P...>::type)>::type;
            };
        }

        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TFnObj,
            typename T0,
            typename T1,
            typename... Ts>
        ALPAKA_FN_HOST_ACC auto foldr(
            TFnObj const & f,
            T0 const & t0,
            T1 const & t1,
            Ts const & ... ts)
        // NOTE: The following line is not allowed because the point of function declaration is after the trailing return type.
        // Thus the function itself is not available inside its return type declaration.
        // http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_closed.html#1433
        // http://stackoverflow.com/questions/3744400/trailing-return-type-using-decltype-with-a-variadic-template-function
        // http://stackoverflow.com/questions/11596898/variadic-template-and-inferred-return-type-in-concat/11597196#11597196
        //-> decltype(f(t0, foldr(f, t1, ts...)))
        -> typename detail::TypeOfFold<TFnObj, T0, T1, Ts...>::type
        {
            return f(t0, foldr(f, t1, ts...));
        }
#else
        //-----------------------------------------------------------------------------
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TFnObj,
            typename T0,
            typename T1,
            typename... Ts>
        ALPAKA_FN_HOST_ACC auto foldr(
            TFnObj const & f,
            T0 const & t0,
            T1 const & t1,
            Ts const & ... ts)
        {
            return f(t0, foldr(f, t1, ts...));
        }
#endif
    }
}
