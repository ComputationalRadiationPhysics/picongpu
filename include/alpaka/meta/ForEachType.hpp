/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/core/Common.hpp>
#include <alpaka/core/Unused.hpp>

#include <utility>

namespace alpaka
{
    namespace meta
    {
        namespace detail
        {
            //#############################################################################
            template<
                typename TList>
            struct ForEachTypeHelper;
            //#############################################################################
            template<
                template<typename...> class TList>
            struct ForEachTypeHelper<
                TList<>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TFnObj,
                    typename... TArgs>
                ALPAKA_FN_HOST_ACC static auto forEachTypeHelper(
                    TFnObj && f,
                    TArgs && ... args)
                -> void
                {
                    alpaka::ignore_unused(f);
                    alpaka::ignore_unused(args...);
                }
            };
            //#############################################################################
            template<
                template<typename...> class TList,
                typename T,
                typename... Ts>
            struct ForEachTypeHelper<
                TList<T, Ts...>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TFnObj,
                    typename... TArgs>
                ALPAKA_FN_HOST_ACC static auto forEachTypeHelper(
                    TFnObj && f,
                    TArgs && ... args)
                -> void
                {
                    // Call the function object template call operator.
#if BOOST_COMP_MSVC && !BOOST_COMP_NVCC
                    f.operator()<T>(
                        std::forward<TArgs>(args)...);
#else
                    f.template operator()<T>(
                        std::forward<TArgs>(args)...);
#endif
                    ForEachTypeHelper<
                        TList<Ts...>>
                    ::forEachTypeHelper(
                        std::forward<TFnObj>(f),
                        std::forward<TArgs>(args)...);
                }
            };
        }

        //-----------------------------------------------------------------------------
        //! Equivalent to boost::mpl::for_each but does not require the types of the sequence to be default constructible.
        //! This function does not create instances of the types instead it passes the types as template parameter.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TList,
            typename TFnObj,
            typename... TArgs>
        ALPAKA_FN_HOST_ACC auto forEachType(
            TFnObj && f,
            TArgs && ... args)
        -> void
        {
            detail::ForEachTypeHelper<
                TList>
            ::forEachTypeHelper(
                std::forward<TFnObj>(f),
                std::forward<TArgs>(args)...);
        }
    }
}
