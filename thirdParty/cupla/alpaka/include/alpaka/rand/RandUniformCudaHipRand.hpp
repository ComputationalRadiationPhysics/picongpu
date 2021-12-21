/* Copyright 2019 Benjamin Worpitz, René Widera
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <alpaka/core/BoostPredef.hpp>

#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
#        error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#    endif

#    if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
#        error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#    endif

#    include <alpaka/dev/DevUniformCudaHipRt.hpp>
#    include <alpaka/rand/Traits.hpp>

// Backend specific imports.
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#        include <alpaka/core/Cuda.hpp>

#        include <curand_kernel.h>

#    elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#        include <alpaka/core/Hip.hpp>

#        pragma clang diagnostic push
#        pragma clang diagnostic ignored "-Wduplicate-decl-specifier"

#        include <hiprand_kernel.h>

#        pragma clang diagnostic pop
#    endif


#    include <type_traits>

namespace alpaka
{
    namespace rand
    {
        //#############################################################################
        //! The CUDA/HIP rand implementation.
        class RandUniformCudaHipRand : public concepts::Implements<ConceptRand, RandUniformCudaHipRand>
        {
        };

        namespace generator
        {
            namespace uniform_cuda_hip
            {
                //#############################################################################
                //! The CUDA/HIP Xor random number generator.
                class Xor
                {
                public:
                    //-----------------------------------------------------------------------------
                    // After calling this constructor the instance is not valid initialized and
                    // need to be overwritten with a valid object
                    //-----------------------------------------------------------------------------
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
                    ALPAKA_FN_HOST_ACC Xor() : m_State(curandStateXORWOW_t{})
#    else
                    ALPAKA_FN_HOST_ACC Xor() : m_State(hiprandStateXORWOW_t{})
#    endif
                    {
                    }

#    if BOOST_COMP_HIP
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST_ACC ~Xor() = default;
#    endif

                    //-----------------------------------------------------------------------------
                    __device__ Xor(
                        std::uint32_t const& seed,
                        std::uint32_t const& subsequence = 0,
                        std::uint32_t const& offset = 0)
                    {
#    ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                        curand_init(seed, subsequence, offset, &m_State);
#    else
                        hiprand_init(seed, subsequence, offset, &m_State);
#    endif
                    }

                public:
#    ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                    curandStateXORWOW_t m_State;
#    else
                    hiprandStateXORWOW_t m_State;
#    endif
                };
            } // namespace uniform_cuda_hip
        } // namespace generator
        namespace distribution
        {
            namespace uniform_cuda_hip
            {
                //#############################################################################
                //! The CUDA/HIP random number floating point normal distribution.
                template<typename T>
                class NormalReal;

                //#############################################################################
                //! The CUDA/HIP random number float normal distribution.
                template<>
                class NormalReal<float>
                {
                public:
                    //-----------------------------------------------------------------------------
                    NormalReal() = default;

                    //-----------------------------------------------------------------------------
                    template<typename TGenerator>
                    __device__ auto operator()(TGenerator& generator) -> float
                    {
#    ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                        return curand_normal(&generator.m_State);
#    else
                        return hiprand_normal(&generator.m_State);
#    endif
                    }
                };
                //#############################################################################
                //! The CUDA/HIP random number float normal distribution.
                template<>
                class NormalReal<double>
                {
                public:
                    //-----------------------------------------------------------------------------
                    NormalReal() = default;

                    //-----------------------------------------------------------------------------
                    template<typename TGenerator>
                    __device__ auto operator()(TGenerator& generator) -> double
                    {
#    ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                        return curand_normal_double(&generator.m_State);
#    else
                        return hiprand_normal_double(&generator.m_State);
#    endif
                    }
                };

                //#############################################################################
                //! The CUDA/HIP random number floating point uniform distribution.
                template<typename T>
                class UniformReal;

                //#############################################################################
                //! The CUDA/HIP random number float uniform distribution.
                template<>
                class UniformReal<float>
                {
                public:
                    //-----------------------------------------------------------------------------
                    UniformReal() = default;

                    //-----------------------------------------------------------------------------
                    template<typename TGenerator>
                    __device__ auto operator()(TGenerator& generator) -> float
                    {
                        // (0.f, 1.0f]
#    ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                        float const fUniformRand(curand_uniform(&generator.m_State));
#    else
                        float const fUniformRand(hiprand_uniform(&generator.m_State));
#    endif
                        // NOTE: (1.0f - curand_uniform) does not work, because curand_uniform seems to return
                        // denormalized floats around 0.f. [0.f, 1.0f)
                        return fUniformRand * static_cast<float>(fUniformRand != 1.0f);
                    }
                };
                //#############################################################################
                //! The CUDA/HIP random number float uniform distribution.
                template<>
                class UniformReal<double>
                {
                public:
                    //-----------------------------------------------------------------------------
                    UniformReal() = default;

                    //-----------------------------------------------------------------------------
                    template<typename TGenerator>
                    __device__ auto operator()(TGenerator& generator) -> double
                    {
                        // (0.f, 1.0f]
#    ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                        double const fUniformRand(curand_uniform_double(&generator.m_State));
#    else
                        double const fUniformRand(hiprand_uniform_double(&generator.m_State));
#    endif
                        // NOTE: (1.0f - curand_uniform_double) does not work, because curand_uniform_double seems to
                        // return denormalized floats around 0.f. [0.f, 1.0f)
                        return fUniformRand * static_cast<double>(fUniformRand != 1.0);
                    }
                };

                //#############################################################################
                //! The CUDA/HIP random number integer uniform distribution.
                template<typename T>
                class UniformUint;

                //#############################################################################
                //! The CUDA/HIP random number unsigned integer uniform distribution.
                template<>
                class UniformUint<unsigned int>
                {
                public:
                    //-----------------------------------------------------------------------------
                    UniformUint() = default;

                    //-----------------------------------------------------------------------------
                    template<typename TGenerator>
                    __device__ auto operator()(TGenerator& generator) -> unsigned int
                    {
#    ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
                        return curand(&generator.m_State);
#    else
                        return hiprand(&generator.m_State);
#    endif
                    }
                };
            } // namespace uniform_cuda_hip
        } // namespace distribution

        namespace distribution
        {
            namespace traits
            {
                //#############################################################################
                //! The CUDA/HIP random number float normal distribution get trait specialization.
                template<typename T>
                struct CreateNormalReal<RandUniformCudaHipRand, T, std::enable_if_t<std::is_floating_point<T>::value>>
                {
                    //-----------------------------------------------------------------------------
                    __device__ static auto createNormalReal(RandUniformCudaHipRand const& /*rand*/)
                        -> rand::distribution::uniform_cuda_hip::NormalReal<T>
                    {
                        return rand::distribution::uniform_cuda_hip::NormalReal<T>();
                    }
                };
                //#############################################################################
                //! The CUDA/HIP random number float uniform distribution get trait specialization.
                template<typename T>
                struct CreateUniformReal<RandUniformCudaHipRand, T, std::enable_if_t<std::is_floating_point<T>::value>>
                {
                    //-----------------------------------------------------------------------------
                    __device__ static auto createUniformReal(RandUniformCudaHipRand const& /*rand*/)
                        -> rand::distribution::uniform_cuda_hip::UniformReal<T>
                    {
                        return rand::distribution::uniform_cuda_hip::UniformReal<T>();
                    }
                };
                //#############################################################################
                //! The CUDA/HIP random number integer uniform distribution get trait specialization.
                template<typename T>
                struct CreateUniformUint<RandUniformCudaHipRand, T, std::enable_if_t<std::is_integral<T>::value>>
                {
                    //-----------------------------------------------------------------------------
                    __device__ static auto createUniformUint(RandUniformCudaHipRand const& /*rand*/)
                        -> rand::distribution::uniform_cuda_hip::UniformUint<T>
                    {
                        return rand::distribution::uniform_cuda_hip::UniformUint<T>();
                    }
                };
            } // namespace traits
        } // namespace distribution
        namespace generator
        {
            namespace traits
            {
                //#############################################################################
                //! The CUDA/HIP random number default generator get trait specialization.
                template<>
                struct CreateDefault<RandUniformCudaHipRand>
                {
                    //-----------------------------------------------------------------------------
                    __device__ static auto createDefault(
                        RandUniformCudaHipRand const& /*rand*/,
                        std::uint32_t const& seed,
                        std::uint32_t const& subsequence) -> rand::generator::uniform_cuda_hip::Xor
                    {
                        return rand::generator::uniform_cuda_hip::Xor(seed, subsequence);
                    }
                };
            } // namespace traits
        } // namespace generator
    } // namespace rand
} // namespace alpaka

#endif
