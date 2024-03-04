/* Copyright 2022 Benjamin Worpitz, René Widera, Andrea Bocci, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/BoostPredef.hpp"
#include "alpaka/core/Concepts.hpp"
#include "alpaka/core/Cuda.hpp"
#include "alpaka/core/Hip.hpp"
#include "alpaka/dev/DevUniformCudaHipRt.hpp"
#include "alpaka/rand/Traits.hpp"

#include <type_traits>

#if(defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)) && !defined(ALPAKA_DISABLE_VENDOR_RNG)

#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#        include <curand_kernel.h>
#    elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#        if BOOST_COMP_CLANG
#            pragma clang diagnostic push
#            pragma clang diagnostic ignored "-Wduplicate-decl-specifier"
#        endif

#        if HIP_VERSION >= 50'200'000
#            include <hiprand/hiprand_kernel.h>
#        else
#            include <hiprand_kernel.h>
#        endif

#        if BOOST_COMP_CLANG
#            pragma clang diagnostic pop
#        endif
#    endif

namespace alpaka::rand
{
    //! The CUDA/HIP rand implementation.
    template<typename TApi>
    class RandUniformCudaHipRand : public concepts::Implements<ConceptRand, RandUniformCudaHipRand<TApi>>
    {
    };

#    if !defined(ALPAKA_HOST_ONLY)

#        if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
#            error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#        endif

#        if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
#            error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#        endif

    namespace distribution::uniform_cuda_hip
    {
        //! The CUDA/HIP random number floating point normal distribution.
        template<typename T>
        class NormalReal;

        //! The CUDA/HIP random number floating point uniform distribution.
        template<typename T>
        class UniformReal;

        //! The CUDA/HIP random number integer uniform distribution.
        template<typename T>
        class UniformUint;
    } // namespace distribution::uniform_cuda_hip

    namespace engine::uniform_cuda_hip
    {
        //! The CUDA/HIP Xor random number generator engine.
        class Xor
        {
        public:
            // After calling this constructor the instance is not valid initialized and
            // need to be overwritten with a valid object
            Xor() = default;

            __device__ Xor(
                std::uint32_t const& seed,
                std::uint32_t const& subsequence = 0,
                std::uint32_t const& offset = 0)
            {
#        ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                curand_init(seed, subsequence, offset, &state);
#        else
                hiprand_init(seed, subsequence, offset, &state);
#        endif
            }

        private:
            template<typename T>
            friend class distribution::uniform_cuda_hip::NormalReal;
            template<typename T>
            friend class distribution::uniform_cuda_hip::UniformReal;
            template<typename T>
            friend class distribution::uniform_cuda_hip::UniformUint;

#        ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
            curandStateXORWOW_t state = curandStateXORWOW_t{};
#        else
            hiprandStateXORWOW_t state = hiprandStateXORWOW_t{};
#        endif

        public:
            // STL UniformRandomBitGenerator concept. This is not strictly necessary as the distributions
            // contained in this file are aware of the API specifics of the CUDA/HIP XORWOW engine and STL
            // distributions might not work on the device, but it servers a compatibility bridge to other
            // potentially compatible alpaka distributions.
#        ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
            using result_type = decltype(curand(&state));
#        else
            using result_type = decltype(hiprand(&state));
#        endif
            ALPAKA_FN_HOST_ACC static constexpr result_type min()
            {
                return std::numeric_limits<result_type>::min();
            }

            ALPAKA_FN_HOST_ACC static constexpr result_type max()
            {
                return std::numeric_limits<result_type>::max();
            }

            __device__ result_type operator()()
            {
#        ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                return curand(&state);
#        else
                return hiprand(&state);
#        endif
            }
        };
    } // namespace engine::uniform_cuda_hip

    namespace distribution::uniform_cuda_hip
    {
        //! The CUDA/HIP random number float normal distribution.
        template<>
        class NormalReal<float>
        {
        public:
            template<typename TEngine>
            __device__ auto operator()(TEngine& engine) -> float
            {
#        ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                return curand_normal(&engine.state);
#        else
                return hiprand_normal(&engine.state);
#        endif
            }
        };

        //! The CUDA/HIP random number float normal distribution.
        template<>
        class NormalReal<double>
        {
        public:
            template<typename TEngine>
            __device__ auto operator()(TEngine& engine) -> double
            {
#        ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                return curand_normal_double(&engine.state);
#        else
                return hiprand_normal_double(&engine.state);
#        endif
            }
        };

        //! The CUDA/HIP random number float uniform distribution.
        template<>
        class UniformReal<float>
        {
        public:
            template<typename TEngine>
            __device__ auto operator()(TEngine& engine) -> float
            {
                // (0.f, 1.0f]
#        ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                float const fUniformRand(curand_uniform(&engine.state));
#        else
                float const fUniformRand(hiprand_uniform(&engine.state));
#        endif
                // NOTE: (1.0f - curand_uniform) does not work, because curand_uniform seems to return
                // denormalized floats around 0.f. [0.f, 1.0f)
                return fUniformRand * static_cast<float>(fUniformRand != 1.0f);
            }
        };

        //! The CUDA/HIP random number float uniform distribution.
        template<>
        class UniformReal<double>
        {
        public:
            template<typename TEngine>
            __device__ auto operator()(TEngine& engine) -> double
            {
                // (0.f, 1.0f]
#        ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                double const fUniformRand(curand_uniform_double(&engine.state));
#        else
                double const fUniformRand(hiprand_uniform_double(&engine.state));
#        endif
                // NOTE: (1.0f - curand_uniform_double) does not work, because curand_uniform_double seems to
                // return denormalized floats around 0.f. [0.f, 1.0f)
                return fUniformRand * static_cast<double>(fUniformRand != 1.0);
            }
        };

        //! The CUDA/HIP random number unsigned integer uniform distribution.
        template<>
        class UniformUint<unsigned int>
        {
        public:
            template<typename TEngine>
            __device__ auto operator()(TEngine& engine) -> unsigned int
            {
#        ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                return curand(&engine.state);
#        else
                return hiprand(&engine.state);
#        endif
            }
        };
    } // namespace distribution::uniform_cuda_hip

    namespace distribution::trait
    {
        //! The CUDA/HIP random number float normal distribution get trait specialization.
        template<typename TApi, typename T>
        struct CreateNormalReal<RandUniformCudaHipRand<TApi>, T, std::enable_if_t<std::is_floating_point_v<T>>>
        {
            __device__ static auto createNormalReal(RandUniformCudaHipRand<TApi> const& /*rand*/)
                -> uniform_cuda_hip::NormalReal<T>
            {
                return {};
            }
        };

        //! The CUDA/HIP random number float uniform distribution get trait specialization.
        template<typename TApi, typename T>
        struct CreateUniformReal<RandUniformCudaHipRand<TApi>, T, std::enable_if_t<std::is_floating_point_v<T>>>
        {
            __device__ static auto createUniformReal(RandUniformCudaHipRand<TApi> const& /*rand*/)
                -> uniform_cuda_hip::UniformReal<T>
            {
                return {};
            }
        };

        //! The CUDA/HIP random number integer uniform distribution get trait specialization.
        template<typename TApi, typename T>
        struct CreateUniformUint<RandUniformCudaHipRand<TApi>, T, std::enable_if_t<std::is_integral_v<T>>>
        {
            __device__ static auto createUniformUint(RandUniformCudaHipRand<TApi> const& /*rand*/)
                -> uniform_cuda_hip::UniformUint<T>
            {
                return {};
            }
        };
    } // namespace distribution::trait

    namespace engine::trait
    {
        //! The CUDA/HIP random number default generator get trait specialization.
        template<typename TApi>
        struct CreateDefault<RandUniformCudaHipRand<TApi>>
        {
            __device__ static auto createDefault(
                RandUniformCudaHipRand<TApi> const& /*rand*/,
                std::uint32_t const& seed = 0,
                std::uint32_t const& subsequence = 0,
                std::uint32_t const& offset = 0) -> uniform_cuda_hip::Xor
            {
                return {seed, subsequence, offset};
            }
        };
    } // namespace engine::trait
#    endif
} // namespace alpaka::rand

#endif
