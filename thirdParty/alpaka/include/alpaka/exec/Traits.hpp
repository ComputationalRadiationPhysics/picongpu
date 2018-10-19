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

#include <alpaka/dim/Traits.hpp>
#include <alpaka/size/Traits.hpp>

#include <alpaka/core/Debug.hpp>
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
    #include <alpaka/workdiv/Traits.hpp>
#endif

#include <alpaka/core/Common.hpp>

#include <type_traits>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The executor specifics.
    namespace exec
    {
        //-----------------------------------------------------------------------------
        //! The execution traits.
        namespace traits
        {
            //#############################################################################
            //! The executor type trait.
            template<
                typename TExec,
                typename TKernelFnObj,
                typename... TArgs/*,
                typename TSfinae = void*/>
            struct ExecType;
        }

        //#############################################################################
        //! The executor type trait alias template to remove the ::type.
        template<
            typename TExec,
            typename TKernelFnObj,
            typename... TArgs>
        using Exec = typename traits::ExecType<TExec, TKernelFnObj, TArgs...>::type;

        //-----------------------------------------------------------------------------
        //! \return An executor.
        template<
            typename TAcc,
            typename TWorkDiv,
            typename TKernelFnObj,
            typename... TArgs>
        ALPAKA_FN_HOST auto create(
            TWorkDiv && workDiv,
            TKernelFnObj && kernelFnObj,
            TArgs && ... args)
        -> Exec<
            TAcc,
            typename std::decay<TKernelFnObj>::type,
            typename std::decay<TArgs>::type...>
        {
            static_assert(
                dim::Dim<typename std::decay<TWorkDiv>::type>::value == dim::Dim<TAcc>::value,
                "The dimensions of TAcc and TWorkDiv have to be identical!");
            static_assert(
                std::is_same<size::Size<typename std::decay<TWorkDiv>::type>, size::Size<TAcc>>::value,
                "The size type of TAcc and the size type of TWorkDiv have to be identical!");

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            std::cout << BOOST_CURRENT_FUNCTION
                << " gridBlockExtent: " << workdiv::getWorkDiv<Grid, Blocks>(workDiv)
                << ", blockThreadExtent: " << workdiv::getWorkDiv<Block, Threads>(workDiv)
                << std::endl;
#endif
            return
                Exec<
                    TAcc,
                    typename std::decay<TKernelFnObj>::type,
                    typename std::decay<TArgs>::type...>(
                        std::forward<TWorkDiv>(workDiv),
                        std::forward<TKernelFnObj>(kernelFnObj),
                        std::forward<TArgs>(args)...);
        }
    }
}
