/* Copyright 2019 Benjamin Worpitz, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

#include <alpaka/core/Common.hpp>

#if !BOOST_LANG_CUDA
    #error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#include <alpaka/rand/Traits.hpp>

#include <alpaka/dev/DevCudaRt.hpp>

#include <alpaka/core/Cuda.hpp>

#include <curand_kernel.h>

#include <type_traits>

namespace alpaka
{
    namespace rand
    {
        //#############################################################################
        //! The CUDA rand implementation.
        class RandCuRand
        {
        public:
            using RandBase = RandCuRand;
        };

        namespace generator
        {
            namespace cuda
            {
                //#############################################################################
                //! The CUDA Xor random number generator.
                class Xor
                {
                public:

                    //-----------------------------------------------------------------------------
                    //! After calling this constructor the instance is not valid initialized and
                    //! need to be overwritten with a valid object
                    Xor() = default;

                    //-----------------------------------------------------------------------------
                    __device__ Xor(
                        std::uint32_t const & seed,
                        std::uint32_t const & subsequence = 0,
                        std::uint32_t const & offset = 0)
                    {
                        curand_init(
                            seed,
                            subsequence,
                            offset,
                            &m_State);
                    }

                public:
                    curandStateXORWOW_t m_State;
                };
            }
        }
        namespace distribution
        {
            namespace cuda
            {
                //#############################################################################
                //! The CUDA random number floating point normal distribution.
                template<
                    typename T>
                class NormalReal;

                //#############################################################################
                //! The CUDA random number float normal distribution.
                template<>
                class NormalReal<
                    float>
                {
                public:
                    //-----------------------------------------------------------------------------
                    NormalReal() = default;

                    //-----------------------------------------------------------------------------
                    template<
                        typename TGenerator>
                    __device__ auto operator()(
                        TGenerator & generator)
                    -> float
                    {
                        return curand_normal(&generator.m_State);
                    }
                };
                //#############################################################################
                //! The CUDA random number float normal distribution.
                template<>
                class NormalReal<
                    double>
                {
                public:
                    //-----------------------------------------------------------------------------
                    NormalReal() = default;

                    //-----------------------------------------------------------------------------
                    template<
                        typename TGenerator>
                    __device__ auto operator()(
                        TGenerator & generator)
                    -> double
                    {
                        return curand_normal_double(&generator.m_State);
                    }
                };

                //#############################################################################
                //! The CUDA random number floating point uniform distribution.
                template<
                    typename T>
                class UniformReal;

                //#############################################################################
                //! The CUDA random number float uniform distribution.
                template<>
                class UniformReal<
                    float>
                {
                public:
                    //-----------------------------------------------------------------------------
                    UniformReal() = default;

                    //-----------------------------------------------------------------------------
                    template<
                        typename TGenerator>
                    __device__ auto operator()(
                        TGenerator & generator)
                    -> float
                    {
                        // (0.f, 1.0f]
                        float const fUniformRand(curand_uniform(&generator.m_State));
                        // NOTE: (1.0f - curand_uniform) does not work, because curand_uniform seems to return denormalized floats around 0.f.
                        // [0.f, 1.0f)
                        return fUniformRand * static_cast<float>( fUniformRand != 1.0f );
                    }
                };
                //#############################################################################
                //! The CUDA random number float uniform distribution.
                template<>
                class UniformReal<
                    double>
                {
                public:
                    //-----------------------------------------------------------------------------
                    UniformReal() = default;

                    //-----------------------------------------------------------------------------
                    template<
                        typename TGenerator>
                    __device__ auto operator()(
                        TGenerator & generator)
                    -> double
                    {
                        // (0.f, 1.0f]
                        double const fUniformRand(curand_uniform_double(&generator.m_State));
                        // NOTE: (1.0f - curand_uniform_double) does not work, because curand_uniform_double seems to return denormalized floats around 0.f.
                        // [0.f, 1.0f)
                        return fUniformRand * static_cast<double>( fUniformRand != 1.0 );
                    }
                };

                //#############################################################################
                //! The CUDA random number integer uniform distribution.
                template<
                    typename T>
                class UniformUint;

                //#############################################################################
                //! The CUDA random number unsigned integer uniform distribution.
                template<>
                class UniformUint<
                    unsigned int>
                {
                public:
                    //-----------------------------------------------------------------------------
                    UniformUint() = default;

                    //-----------------------------------------------------------------------------
                    template<
                        typename TGenerator>
                    __device__ auto operator()(
                        TGenerator & generator)
                    -> unsigned int
                    {
                        return curand(&generator.m_State);
                    }
                };
            }
        }

        namespace distribution
        {
            namespace traits
            {
                //#############################################################################
                //! The CUDA random number float normal distribution get trait specialization.
                template<
                    typename T>
                struct CreateNormalReal<
                    RandCuRand,
                    T,
                    typename std::enable_if<
                        std::is_floating_point<T>::value>::type>
                {
                    //-----------------------------------------------------------------------------
                    __device__ static auto createNormalReal(
                        RandCuRand const & /*rand*/)
                    -> rand::distribution::cuda::NormalReal<T>
                    {
                        return rand::distribution::cuda::NormalReal<T>();
                    }
                };
                //#############################################################################
                //! The CUDA random number float uniform distribution get trait specialization.
                template<
                    typename T>
                struct CreateUniformReal<
                    RandCuRand,
                    T,
                    typename std::enable_if<
                        std::is_floating_point<T>::value>::type>
                {
                    //-----------------------------------------------------------------------------
                    __device__ static auto createUniformReal(
                        RandCuRand const & /*rand*/)
                    -> rand::distribution::cuda::UniformReal<T>
                    {
                        return rand::distribution::cuda::UniformReal<T>();
                    }
                };
                //#############################################################################
                //! The CUDA random number integer uniform distribution get trait specialization.
                template<
                    typename T>
                struct CreateUniformUint<
                    RandCuRand,
                    T,
                    typename std::enable_if<
                        std::is_integral<T>::value>::type>
                {
                    //-----------------------------------------------------------------------------
                    __device__ static auto createUniformUint(
                        RandCuRand const & /*rand*/)
                    -> rand::distribution::cuda::UniformUint<T>
                    {
                        return rand::distribution::cuda::UniformUint<T>();
                    }
                };
            }
        }
        namespace generator
        {
            namespace traits
            {
                //#############################################################################
                //! The CUDA random number default generator get trait specialization.
                template<>
                struct CreateDefault<
                    RandCuRand>
                {
                    //-----------------------------------------------------------------------------
                    __device__ static auto createDefault(
                        RandCuRand const & /*rand*/,
                        std::uint32_t const & seed,
                        std::uint32_t const & subsequence)
                    -> rand::generator::cuda::Xor
                    {
                        return rand::generator::cuda::Xor(
                            seed,
                            subsequence);
                    }
                };
            }
        }
    }
}

#endif
