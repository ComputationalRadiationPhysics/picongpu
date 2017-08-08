/**
* \file
* Copyright 2015-2016 Benjamin Worpitz
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

// This is not currently supported by the clang native CUDA compiler.
#if !BOOST_COMP_CLANG_CUDA

#include <alpaka/rand/Traits.hpp>       // CreateNormalReal, ...

#include <alpaka/dev/DevCudaRt.hpp>     // dev::DevCudaRt

#include <alpaka/core/Cuda.hpp>         // ALPAKA_CUDA_RT_CHECK

#include <curand_kernel.h>              // curand_init, ...

#include <type_traits>                  // std::enable_if

namespace alpaka
{
    namespace rand
    {
        //#############################################################################
        //! The CUDA rand implementation.
        //#############################################################################
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
                //#############################################################################
                class Xor
                {
                public:

                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //
                    // After calling this constructor the instance is not valid initialized and
                    // need to be overwritten with a valid object
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_CUDA_ONLY Xor() = default;

                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_CUDA_ONLY Xor(
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
                //#############################################################################
                template<
                    typename T>
                class NormalReal;

                //#############################################################################
                //! The CUDA random number float normal distribution.
                //#############################################################################
                template<>
                class NormalReal<
                    float>
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_CUDA_ONLY NormalReal() = default;

                    //-----------------------------------------------------------------------------
                    //! Call operator.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TGenerator>
                    ALPAKA_FN_ACC_CUDA_ONLY auto operator()(
                        TGenerator & generator)
                    -> float
                    {
                        return curand_normal(&generator.m_State);
                    }
                };
                //#############################################################################
                //! The CUDA random number float normal distribution.
                //#############################################################################
                template<>
                class NormalReal<
                    double>
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_CUDA_ONLY NormalReal() = default;

                    //-----------------------------------------------------------------------------
                    //! Call operator.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TGenerator>
                    ALPAKA_FN_ACC_CUDA_ONLY auto operator()(
                        TGenerator & generator)
                    -> double
                    {
                        return curand_normal_double(&generator.m_State);
                    }
                };

                //#############################################################################
                //! The CUDA random number floating point uniform distribution.
                //#############################################################################
                template<
                    typename T>
                class UniformReal;

                //#############################################################################
                //! The CUDA random number float uniform distribution.
                //#############################################################################
                template<>
                class UniformReal<
                    float>
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_CUDA_ONLY UniformReal() = default;

                    //-----------------------------------------------------------------------------
                    //! Call operator.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TGenerator>
                    ALPAKA_FN_ACC_CUDA_ONLY auto operator()(
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
                //#############################################################################
                template<>
                class UniformReal<
                    double>
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_CUDA_ONLY UniformReal() = default;

                    //-----------------------------------------------------------------------------
                    //! Call operator.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TGenerator>
                    ALPAKA_FN_ACC_CUDA_ONLY auto operator()(
                        TGenerator & generator)
                    -> double
                    {
                        // (0.f, 1.0f]
                        double const fUniformRand(curand_uniform_double(&generator.m_State));
                        // NOTE: (1.0f - curand_uniform_double) does not work, because curand_uniform_double seems to return denormalized floats around 0.f.
                        // [0.f, 1.0f)
                        return fUniformRand * static_cast<double>( fUniformRand != 1.0f );
                    }
                };

                //#############################################################################
                //! The CUDA random number integer uniform distribution.
                //#############################################################################
                template<
                    typename T>
                class UniformUint;

                //#############################################################################
                //! The CUDA random number unsigned integer uniform distribution.
                //#############################################################################
                template<>
                class UniformUint<
                    unsigned int>
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_CUDA_ONLY UniformUint() = default;

                    //-----------------------------------------------------------------------------
                    //! Call operator.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TGenerator>
                    ALPAKA_FN_ACC_CUDA_ONLY auto operator()(
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
                //#############################################################################
                template<
                    typename T>
                struct CreateNormalReal<
                    RandCuRand,
                    T,
                    typename std::enable_if<
                        std::is_floating_point<T>::value>::type>
                {
                    //-----------------------------------------------------------------------------
                    //
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_CUDA_ONLY static auto createNormalReal(
                        RandCuRand const & /*rand*/)
                    -> rand::distribution::cuda::NormalReal<T>
                    {
                        return rand::distribution::cuda::NormalReal<T>();
                    }
                };
                //#############################################################################
                //! The CUDA random number float uniform distribution get trait specialization.
                //#############################################################################
                template<
                    typename T>
                struct CreateUniformReal<
                    RandCuRand,
                    T,
                    typename std::enable_if<
                        std::is_floating_point<T>::value>::type>
                {
                    //-----------------------------------------------------------------------------
                    //
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_CUDA_ONLY static auto createUniformReal(
                        RandCuRand const & /*rand*/)
                    -> rand::distribution::cuda::UniformReal<T>
                    {
                        return rand::distribution::cuda::UniformReal<T>();
                    }
                };
                //#############################################################################
                //! The CUDA random number integer uniform distribution get trait specialization.
                //#############################################################################
                template<
                    typename T>
                struct CreateUniformUint<
                    RandCuRand,
                    T,
                    typename std::enable_if<
                        std::is_integral<T>::value>::type>
                {
                    //-----------------------------------------------------------------------------
                    //
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_CUDA_ONLY static auto createUniformUint(
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
                //#############################################################################
                template<>
                struct CreateDefault<
                    RandCuRand>
                {
                    //-----------------------------------------------------------------------------
                    //
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_CUDA_ONLY static auto createDefault(
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
#endif
