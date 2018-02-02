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

#include <boost/predef.h>
#if !BOOST_ARCH_CUDA_DEVICE
    #include <boost/core/ignore_unused.hpp>
#endif

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
#if !BOOST_ARCH_CUDA_DEVICE
                    TFnObj && f,
                    TArgs && ... args)
#else
                    TFnObj &&,
                    TArgs && ...)
#endif
                -> void
                {
#if !BOOST_ARCH_CUDA_DEVICE
                    boost::ignore_unused(f);
                    boost::ignore_unused(args...);
#endif
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
