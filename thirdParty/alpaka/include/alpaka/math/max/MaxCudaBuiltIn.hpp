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

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include <alpaka/core/Common.hpp>       // ALPAKA_FN_*, BOOST_LANG_CUDA

#if !BOOST_LANG_CUDA
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#include <alpaka/math/max/Traits.hpp>   // Max

//#include <boost/core/ignore_unused.hpp> // boost::ignore_unused

#include <type_traits>                  // std::enable_if, std::is_arithmetic, std::is_integral
#include <math_functions.hpp>           // ::max

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library max.
        //#############################################################################
        class MaxCudaBuiltIn
        {
        public:
            using MaxBase = MaxCudaBuiltIn;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library integral max trait specialization.
            //#############################################################################
            template<
                typename Tx,
                typename Ty>
            struct Max<
                MaxCudaBuiltIn,
                Tx,
                Ty,
                typename std::enable_if<
                    std::is_integral<Tx>::value
                    && std::is_integral<Ty>::value>::type>
            {
                ALPAKA_FN_ACC_CUDA_ONLY static auto max(
                    MaxCudaBuiltIn const & /*max*/,
                    Tx const & x,
                    Ty const & y)
                -> decltype(::max(x, y))
                {
                    //boost::ignore_unused(max);
                    return ::max(x, y);
                }
            };
            //#############################################################################
            //! The standard library mixed integral floating point max trait specialization.
            //#############################################################################
            template<
                typename Tx,
                typename Ty>
            struct Max<
                MaxCudaBuiltIn,
                Tx,
                Ty,
                typename std::enable_if<
                    std::is_arithmetic<Tx>::value
                    && std::is_arithmetic<Ty>::value
                    && !(std::is_integral<Tx>::value
                        && std::is_integral<Ty>::value)>::type>
            {
                ALPAKA_FN_ACC_CUDA_ONLY static auto max(
                    MaxCudaBuiltIn const & /*max*/,
                    Tx const & x,
                    Ty const & y)
                -> decltype(::fmax(x, y))
                {
                    //boost::ignore_unused(max);
                    return ::fmax(x, y);
                }
            };
        }
    }
}

#endif
