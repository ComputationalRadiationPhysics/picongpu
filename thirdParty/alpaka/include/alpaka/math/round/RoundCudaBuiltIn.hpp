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

#include <alpaka/core/Common.hpp>

#if !BOOST_LANG_CUDA
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#include <alpaka/math/round/Traits.hpp>

//#include <boost/core/ignore_unused.hpp>

#include <type_traits>
#if BOOST_COMP_NVCC >= BOOST_VERSION_NUMBER(9, 1, 0)
    #include <crt/math_functions.hpp>
#else
    #include <math_functions.hpp>
#endif

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library round.
        class RoundCudaBuiltIn
        {
        public:
            using RoundBase = RoundCudaBuiltIn;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library round trait specialization.
            template<
                typename TArg>
            struct Round<
                RoundCudaBuiltIn,
                TArg,
                typename std::enable_if<
                    std::is_floating_point<TArg>::value>::type>
            {
                ALPAKA_FN_ACC_CUDA_ONLY static auto round(
                    RoundCudaBuiltIn const & /*round*/,
                    TArg const & arg)
                -> decltype(::round(arg))
                {
                    //boost::ignore_unused(round);
                    return ::round(arg);
                }
            };
            //#############################################################################
            //! The standard library round trait specialization.
            template<
                typename TArg>
            struct Lround<
                RoundCudaBuiltIn,
                TArg,
                typename std::enable_if<
                    std::is_floating_point<TArg>::value>::type>
            {
                ALPAKA_FN_ACC_CUDA_ONLY static auto lround(
                    RoundCudaBuiltIn const & /*lround*/,
                    TArg const & arg)
                -> long int
                {
                    //boost::ignore_unused(lround);
                    return ::lround(arg);
                }
            };
            //#############################################################################
            //! The standard library round trait specialization.
            template<
                typename TArg>
            struct Llround<
                RoundCudaBuiltIn,
                TArg,
                typename std::enable_if<
                    std::is_floating_point<TArg>::value>::type>
            {
                ALPAKA_FN_ACC_CUDA_ONLY static auto llround(
                    RoundCudaBuiltIn const & /*llround*/,
                    TArg const & arg)
                -> long int
                {
                    //boost::ignore_unused(llround);
                    return ::llround(arg);
                }
            };
        }
    }
}

#endif
