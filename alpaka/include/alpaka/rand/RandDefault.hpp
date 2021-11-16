/* Copyright 2021 Jeffrey Kelling
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/math/cos/Traits.hpp>
#include <alpaka/math/isnan/Traits.hpp>
#include <alpaka/math/log/Traits.hpp>
#include <alpaka/math/sin/Traits.hpp>
#include <alpaka/math/sqrt/Traits.hpp>
#include <alpaka/rand/RandPhilox.hpp>
#include <alpaka/rand/Traits.hpp>

#include <algorithm>
#include <limits>
#include <type_traits>

namespace alpaka
{
    namespace rand
    {
        class RandDefault : public concepts::Implements<ConceptRand, RandDefault>
        {
        };

        namespace distribution
        {
            namespace gpu
            {
                namespace detail
                {
                    template<typename TFloat>
                    struct BitsType;

                    template<>
                    struct BitsType<float>
                    {
                        using type = std::uint32_t;
                    };
                    template<>
                    struct BitsType<double>
                    {
                        using type = std::uint64_t;
                    };
                } // namespace detail

                //! The GPU random number normal distribution.
                template<typename T>
                class UniformUint
                {
                    static_assert(std::is_integral<T>::value, "Return type of UniformUint must be integral.");

                public:
                    UniformUint() = default;

                    template<typename TEngine>
                    ALPAKA_FN_HOST_ACC auto operator()(TEngine& engine) -> T
                    {
                        using BitsT = typename TEngine::result_type;
                        T ret = 0;
                        constexpr auto N = sizeof(T) / sizeof(BitsT);
                        for(unsigned int a = 0; a < N; ++a)
                        {
                            ret
                                ^= (static_cast<T>(engine())
                                    << (sizeof(BitsT) * std::numeric_limits<unsigned char>::digits * a));
                        }
                        return ret;
                    }
                };

                //! The GPU random number uniform distribution.
                template<typename T>
                class UniformReal
                {
                    static_assert(
                        std::is_floating_point<T>::value,
                        "Return type of UniformReal must be floating point.");

                    using BitsT = typename detail::BitsType<T>::type;

                public:
                    UniformReal() = default;

                    template<typename TEngine>
                    ALPAKA_FN_HOST_ACC auto operator()(TEngine& engine) -> T
                    {
                        constexpr BitsT limit = static_cast<BitsT>(1) << std::numeric_limits<T>::digits;
                        const BitsT b = UniformUint<BitsT>()(engine);
                        const auto ret = static_cast<T>(b & (limit - 1)) / limit;
                        return ret;
                    }
                };

                /*! The GPU random number normal distribution.
                 *
                 * \note
                 * This type contains state and is not thread-safe: To be used
                 * per thread, not shared.
                 *
                 * \note When reproducibility is a concern, each instance of
                 * this class should be used with only on random engine
                 * instance, or two consecutive number should be generated with
                 * each engine used. This is due to the implicit caching of one
                 * Gaussian random number.
                 */
                template<typename Acc, typename T>
                class NormalReal
                {
                    static_assert(
                        std::is_floating_point<T>::value,
                        "Return type of NormalReal must be floating point.");

                    const Acc& m_acc;
                    T m_cache = std::numeric_limits<T>::quiet_NaN();

                public:
                    /*! \warning Retains a reference to \p acc, thus must not
                     * outlive it.
                     */
                    NormalReal(const Acc& acc) : m_acc(acc)
                    {
                    }

                    NormalReal(const NormalReal& o) = delete;

                    //! The move ctor clears `m_cache` of source.
                    //! \todo This is to be deleted when moving to C++17
                    NormalReal(NormalReal&& o) : m_acc(o.m_acc), m_cache(o.m_cache)
                    {
                        o.m_cache = std::numeric_limits<T>::quiet_NaN();
                    }

                    template<typename TEngine>
                    ALPAKA_FN_HOST_ACC auto operator()(TEngine& engine) -> T
                    {
                        constexpr T sigma = 1., mu = 0.;
                        if(math::isnan(m_acc, m_cache))
                        {
                            UniformReal<T> uni;

                            T u1, u2;
                            do
                            {
                                u1 = uni(engine);
                                u2 = uni(engine);
                            } while(u1 <= std::numeric_limits<T>::epsilon());

                            // compute z0 and z1
                            const T mag = sigma * math::sqrt(m_acc, static_cast<T>(-2.) * math::log(m_acc, u1));
                            constexpr T twoPi = static_cast<T>(2. * M_PI);
                            // getting two normal number out of this, store one for later
                            m_cache = mag * static_cast<T>(math::cos(m_acc, twoPi * u2)) + mu;

                            return mag * static_cast<T>(math::sin(m_acc, twoPi * u2)) + mu;
                        }
                        else
                        {
                            const T ret = m_cache;
                            m_cache = std::numeric_limits<T>::quiet_NaN();
                            return ret;
                        }
                    }
                };
            } // namespace gpu
        } // namespace distribution

        namespace distribution
        {
            namespace traits
            {
                //! The GPU device random number float normal distribution get trait specialization.
                template<typename T>
                struct CreateNormalReal<RandDefault, T, std::enable_if_t<std::is_floating_point<T>::value>>
                {
                    template<typename TAcc>
                    ALPAKA_FN_HOST_ACC static auto createNormalReal(TAcc const& acc)
                    {
                        return rand::distribution::gpu::NormalReal<TAcc, T>(acc);
                    }
                };
                //! The GPU device random number float uniform distribution get trait specialization.
                template<typename T>
                struct CreateUniformReal<RandDefault, T, std::enable_if_t<std::is_floating_point<T>::value>>
                {
                    ALPAKA_FN_HOST_ACC static auto createUniformReal(RandDefault const& rand)
                    {
                        alpaka::ignore_unused(rand);
                        return rand::distribution::gpu::UniformReal<T>();
                    }
                };
                //! The GPU device random number integer uniform distribution get trait specialization.
                template<typename T>
                struct CreateUniformUint<RandDefault, T, std::enable_if_t<std::is_integral<T>::value>>
                {
                    ALPAKA_FN_HOST_ACC static auto createUniformUint(RandDefault const& rand)
                    {
                        alpaka::ignore_unused(rand);
                        return rand::distribution::gpu::UniformUint<T>();
                    }
                };
            } // namespace traits
        } // namespace distribution
        namespace engine
        {
            namespace traits
            {
                //! The GPU device random number default generator get trait specialization.
                template<>
                struct CreateDefault<RandDefault>
                {
                    template<typename TAcc>
                    ALPAKA_FN_HOST_ACC static auto createDefault(
                        TAcc const& acc,
                        std::uint32_t const& seed,
                        std::uint32_t const& subsequence,
                        std::uint32_t const& offset) -> rand::Philox4x32x10<TAcc>
                    {
                        alpaka::ignore_unused(acc);
                        return rand::Philox4x32x10<TAcc>(seed, subsequence, offset);
                    }
                };

            } // namespace traits
        } // namespace engine
    } // namespace rand
} // namespace alpaka
