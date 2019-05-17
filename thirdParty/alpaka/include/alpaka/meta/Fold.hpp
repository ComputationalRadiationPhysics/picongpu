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

#include <boost/config.hpp>

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
            TFnObj const & f,
            T const & t)
        -> T
        {
            alpaka::ignore_unused(f);

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
