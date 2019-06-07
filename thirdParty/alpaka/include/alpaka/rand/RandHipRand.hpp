/* Copyright 2019 Benjamin Worpitz, Matthias Werner, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

#include <alpaka/core/Common.hpp>

#if !BOOST_LANG_HIP
    #error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

#include <alpaka/rand/Traits.hpp>

#include <alpaka/dev/DevHipRt.hpp>

#include <alpaka/core/Hip.hpp>

#include <hiprand_kernel.h>

#include <type_traits>

namespace alpaka
{
    namespace rand
    {
        //#############################################################################
        //! The HIP rand implementation.
        class RandHipRand
        {
        public:
            using RandBase = RandHipRand;
        };

        namespace generator
        {
            namespace hip
            {
                //#############################################################################
                //! The HIP Xor random number generator.
                class Xor
                {
                public:

                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //
                    // After calling this constructor the instance is not valid initialized and
                    // need to be overwritten with a valid object
                    //-----------------------------------------------------------------------------
                    Xor() = default;

                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    __device__ Xor(
                        std::uint32_t const & seed,
                        std::uint32_t const & subsequence = 0,
                        std::uint32_t const & offset = 0)
                    {
                        hiprand_init(
                            seed,
                            subsequence,
                            offset,
                            &m_State);
                    }

                public:
                    hiprandStateXORWOW_t m_State;
                };
            }
        }
        namespace distribution
        {
            namespace hip
            {
                //#############################################################################
                //! The HIP random number floating point normal distribution.
                template<
                    typename T>
                class NormalReal;

                //#############################################################################
                //! The HIP random number float normal distribution.
                template<>
                class NormalReal<
                    float>
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    NormalReal() = default;

                    //-----------------------------------------------------------------------------
                    //! Call operator.
                    template<
                        typename TGenerator>
                    __device__ auto operator()(
                        TGenerator & generator)
                    -> float
                    {
                        return hiprand_normal(&generator.m_State);
                    }
                };
                //#############################################################################
                //! The HIP random number float normal distribution.
                template<>
                class NormalReal<
                    double>
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    NormalReal() = default;

                    //-----------------------------------------------------------------------------
                    //! Call operator.
                    template<
                        typename TGenerator>
                    __device__ auto operator()(
                        TGenerator & generator)
                    -> double
                    {
                        return hiprand_normal_double(&generator.m_State);
                    }
                };

                //#############################################################################
                //! The HIP random number floating point uniform distribution.
                template<
                    typename T>
                class UniformReal;

                //#############################################################################
                //! The HIP random number float uniform distribution.
                template<>
                class UniformReal<
                    float>
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    UniformReal() = default;

                    //-----------------------------------------------------------------------------
                    //! Call operator.
                    template<
                        typename TGenerator>
                    __device__ auto operator()(
                        TGenerator & generator)
                    -> float
                    {
                        // (0.f, 1.0f]
                        float const fUniformRand(hiprand_uniform(&generator.m_State));
                        // NOTE: (1.0f - hiprand_uniform) does not work, because hiprand_uniform seems to return denormalized floats around 0.f.
                        // [0.f, 1.0f)
                        return fUniformRand * static_cast<float>( fUniformRand != 1.0f );
                    }
                };
                //#############################################################################
                //! The HIP random number float uniform distribution.
                template<>
                class UniformReal<
                    double>
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    UniformReal() = default;

                    //-----------------------------------------------------------------------------
                    //! Call operator.
                    template<
                        typename TGenerator>
                    __device__ auto operator()(
                        TGenerator & generator)
                    -> double
                    {
                        // (0.f, 1.0f]
                        double const fUniformRand(hiprand_uniform_double(&generator.m_State));
                        // NOTE: (1.0f - hiprand_uniform_double) does not work, because hiprand_uniform_double seems to return denormalized floats around 0.f.
                        // [0.f, 1.0f)
                        return fUniformRand * static_cast<double>( fUniformRand != 1.0f );
                    }
                };

                //#############################################################################
                //! The HIP random number integer uniform distribution.
                template<
                    typename T>
                class UniformUint;

                //#############################################################################
                //! The HIP random number unsigned integer uniform distribution.
                template<>
                class UniformUint<
                    unsigned int>
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    UniformUint() = default;

                    //-----------------------------------------------------------------------------
                    //! Call operator.
                    template<
                        typename TGenerator>
                    __device__ auto operator()(
                        TGenerator & generator)
                    -> unsigned int
                    {
                        return hiprand(&generator.m_State);
                    }
                };
            }
        }

        namespace distribution
        {
            namespace traits
            {
                //#############################################################################
                //! The HIP random number float normal distribution get trait specialization.
                template<
                    typename T>
                struct CreateNormalReal<
                    RandHipRand,
                    T,
                    typename std::enable_if<
                        std::is_floating_point<T>::value>::type>
                {
                    //-----------------------------------------------------------------------------

                    ALPAKA_FN_HOST_ACC static auto createNormalReal(
                        RandHipRand const & /*rand*/)
                    -> rand::distribution::hip::NormalReal<T>
                    {
                        return rand::distribution::hip::NormalReal<T>();
                    }
                };
                //#############################################################################
                //! The HIP random number float uniform distribution get trait specialization.
                template<
                    typename T>
                struct CreateUniformReal<
                    RandHipRand,
                    T,
                    typename std::enable_if<
                        std::is_floating_point<T>::value>::type>
                {
                    //-----------------------------------------------------------------------------

                    ALPAKA_FN_HOST_ACC static auto createUniformReal(
                        RandHipRand const & /*rand*/)
                    -> rand::distribution::hip::UniformReal<T>
                    {
                        return rand::distribution::hip::UniformReal<T>();
                    }
                };
                //#############################################################################
                //! The HIP random number integer uniform distribution get trait specialization.
                template<
                    typename T>
                struct CreateUniformUint<
                    RandHipRand,
                    T,
                    typename std::enable_if<
                        std::is_integral<T>::value>::type>
                {
                    //-----------------------------------------------------------------------------

                    ALPAKA_FN_HOST_ACC static auto createUniformUint(
                        RandHipRand const & /*rand*/)
                    -> rand::distribution::hip::UniformUint<T>
                    {
                        return rand::distribution::hip::UniformUint<T>();
                    }
                };
            }
        }
        namespace generator
        {
            namespace traits
            {
                //#############################################################################
                //! The HIP random number default generator get trait specialization.
                template<>
                struct CreateDefault<
                    RandHipRand>
                {
                    //-----------------------------------------------------------------------------

                    __device__ static auto createDefault(
                        RandHipRand const & /*rand*/,
                        std::uint32_t const & seed,
                        std::uint32_t const & subsequence)
                    -> rand::generator::hip::Xor
                    {
                        return rand::generator::hip::Xor(
                            seed,
                            subsequence);
                    }
                };
            }
        }
    }
}

#endif
