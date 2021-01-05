/* Copyright 2015-2021 Alexander Grund, Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/types.hpp"
#include "pmacc/static_assert.hpp"

#if(BOOST_LANG_CUDA)
#    include <curand_kernel.h>
#elif(BOOST_LANG_HIP)
#    include <hiprand_kernel.h>
#else
#    include "pmacc/random/methods/AlpakaRand.hpp"
#endif


namespace pmacc
{
    namespace random
    {
        namespace methods
        {
#if(ALPAKA_ACC_GPU_CUDA_ENABLED || ALPAKA_ACC_GPU_HIP_ENABLED)
            //! Uses the CUDA XORWOW RNG but does not store state members required for normal distribution
            template<typename T_Acc = cupla::Acc>
            class XorMin
            {
#    if(BOOST_LANG_HIP)
                using NativeStateType = hiprandStateXORWOW_t;
#    elif(BOOST_LANG_CUDA)
                using NativeStateType = curandStateXORWOW_t;
#    endif

            public:
                class StateType
                {
                public:
                    PMACC_ALIGN(d, unsigned int);
                    PMACC_ALIGN(v[5], unsigned int);

                    HDINLINE StateType()
                    {
                    }

                    DINLINE StateType(NativeStateType const& other)
                    {
#    if(BOOST_LANG_HIP)
                        // @todo avoid using pointer casts to copy the rng state
                        auto baseObjectPtr
                            = reinterpret_cast<typename NativeStateType::xorwow_state const* const>(&other);
                        d = baseObjectPtr->d;
                        auto const* nativeStateArray = baseObjectPtr->x;
                        PMACC_STATIC_ASSERT_MSG(sizeof(v) == sizeof(baseObjectPtr->x), Unexpected_sizes);
#    elif(BOOST_LANG_CUDA)
                        d = other.d;
                        auto const* nativeStateArray = other.v;
                        PMACC_STATIC_ASSERT_MSG(sizeof(v) == sizeof(other.v), Unexpected_sizes);
#    endif
                        for(unsigned i = 0; i < sizeof(v) / sizeof(v[0]); i++)
                            v[i] = nativeStateArray[i];
                    }
                };

                DINLINE void init(T_Acc const& acc, StateType& state, uint32_t seed, uint32_t subsequence = 0) const
                {
                    NativeStateType tmpState;

#    if(ALPAKA_ACC_GPU_HIP_ENABLED == 1)
#        define PMACC_RNG_INIT_FN hiprand_init
#    elif(ALPAKA_ACC_GPU_CUDA_ENABLED == 1)
#        define PMACC_RNG_INIT_FN curand_init
#    endif

                    PMACC_RNG_INIT_FN(seed, subsequence, 0, &tmpState);

#    undef PMACC_RNG_INIT_FN

                    state = tmpState;
                }

                DINLINE uint32_t get32Bits(T_Acc const& acc, StateType& state) const
                {
                    /* This generator uses the xorwow formula of
                     * www.jstatsoft.org/v08/i14/paper page 5
                     * Has period 2^192 - 2^32.
                     */
                    uint32_t t;
                    t = (state.v[0] ^ (state.v[0] >> 2));
                    state.v[0] = state.v[1];
                    state.v[1] = state.v[2];
                    state.v[2] = state.v[3];
                    state.v[3] = state.v[4];
                    state.v[4] = (state.v[4] ^ (state.v[4] << 4)) ^ (t ^ (t << 1));
                    state.d += 362437;
                    return state.v[4] + state.d;
                }

                DINLINE uint64_t get64Bits(T_Acc const& acc, StateType& state) const
                {
                    // two 32bit values are packed into a 64bit value
                    uint64_t result = get32Bits(acc, state);
                    result <<= 32;
                    result ^= get32Bits(acc, state);
                    return result;
                }

                static std::string getName()
                {
                    return "XorMin";
                }
            };
#else
            //! fallback to alpaka RNG if a cpu accelerator is used
            template<typename T_Acc = cupla::Acc>
            using XorMin = AlpakaRand<T_Acc>;
#endif
        } // namespace methods
    } // namespace random
} // namespace pmacc
