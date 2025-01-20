/* Copyright 2022 Jeffrey Kelling, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/math/Traits.hpp"
#include "alpaka/rand/RandPhilox.hpp"
#include "alpaka/rand/Traits.hpp"

#include <algorithm>
#include <limits>
#include <type_traits>

namespace alpaka::rand
{
    class RandDefault : public concepts::Implements<ConceptRand, RandDefault>
    {
    };

    namespace distribution::gpu
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
            static_assert(std::is_integral_v<T>, "Return type of UniformUint must be integral.");

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
            static_assert(std::is_floating_point_v<T>, "Return type of UniformReal must be floating point.");

            using BitsT = typename detail::BitsType<T>::type;

        public:
            UniformReal() = default;

            template<typename TEngine>
            ALPAKA_FN_HOST_ACC auto operator()(TEngine& engine) -> T
            {
                constexpr BitsT limit = static_cast<BitsT>(1) << std::numeric_limits<T>::digits;
                BitsT const b = UniformUint<BitsT>()(engine);
                auto const ret = static_cast<T>(b & (limit - 1)) / limit;
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
            static_assert(std::is_floating_point_v<T>, "Return type of NormalReal must be floating point.");

            Acc const* m_acc;
            T m_cache = std::numeric_limits<T>::quiet_NaN();

        public:
            /*! \warning Retains a reference to \p acc, thus must not outlive it.
             */
            ALPAKA_FN_HOST_ACC constexpr NormalReal(Acc const& acc) : m_acc(&acc)
            {
            }

            // All copy operations (and thus also move since we don't declare those and they fall back to copy) do NOT
            // copy m_cache. This way we can ensure that the following holds:
            // NormalReal<Acc> a(acc), b(acc);
            // Engine<Acc> e(acc);
            // assert(a(e) != b(e)); // because of two engine invocations
            // b = a;
            // assert(a(e) != b(e)); // because of two engine invocations

            ALPAKA_FN_HOST_ACC constexpr NormalReal(NormalReal const& other) : m_acc(other.m_acc)
            {
            }

            ALPAKA_FN_HOST_ACC constexpr auto operator=(NormalReal const& other) -> NormalReal&
            {
                m_acc = other.m_acc;
                return *this;
            }

            template<typename TEngine>
            ALPAKA_FN_HOST_ACC auto operator()(TEngine& engine) -> T
            {
                constexpr auto sigma = T{1};
                constexpr auto mu = T{0};
                if(math::isnan(*m_acc, m_cache))
                {
                    UniformReal<T> uni;

                    T u1, u2;
                    do
                    {
                        u1 = uni(engine);
                        u2 = uni(engine);
                    } while(u1 <= std::numeric_limits<T>::epsilon());

                    // compute z0 and z1
                    T const mag = sigma * math::sqrt(*m_acc, static_cast<T>(-2.) * math::log(*m_acc, u1));
                    constexpr T twoPi = static_cast<T>(2. * math::constants::pi);
                    // getting two normal number out of this, store one for later
                    m_cache = mag * static_cast<T>(math::cos(*m_acc, twoPi * u2)) + mu;

                    return mag * static_cast<T>(math::sin(*m_acc, twoPi * u2)) + mu;
                }

                T const ret = m_cache;
                m_cache = std::numeric_limits<T>::quiet_NaN();
                return ret;
            }
        };
    } // namespace distribution::gpu

    namespace distribution::trait
    {
        //! The GPU device random number float normal distribution get trait specialization.
        template<typename T>
        struct CreateNormalReal<RandDefault, T, std::enable_if_t<std::is_floating_point_v<T>>>
        {
            template<typename TAcc>
            ALPAKA_FN_HOST_ACC static auto createNormalReal(TAcc const& acc) -> gpu::NormalReal<TAcc, T>
            {
                return {acc};
            }
        };

        //! The GPU device random number float uniform distribution get trait specialization.
        template<typename T>
        struct CreateUniformReal<RandDefault, T, std::enable_if_t<std::is_floating_point_v<T>>>
        {
            ALPAKA_FN_HOST_ACC static auto createUniformReal(RandDefault const& /* rand */) -> gpu::UniformReal<T>
            {
                return {};
            }
        };

        //! The GPU device random number integer uniform distribution get trait specialization.
        template<typename T>
        struct CreateUniformUint<RandDefault, T, std::enable_if_t<std::is_integral_v<T>>>
        {
            ALPAKA_FN_HOST_ACC static auto createUniformUint(RandDefault const& /* rand */) -> gpu::UniformUint<T>
            {
                return {};
            }
        };
    } // namespace distribution::trait

    namespace engine::trait
    {
        //! The GPU device random number default generator get trait specialization.
        template<>
        struct CreateDefault<RandDefault>
        {
            template<typename TAcc>
            ALPAKA_FN_HOST_ACC static auto createDefault(
                TAcc const& /* acc */,
                std::uint32_t const& seed,
                std::uint32_t const& subsequence,
                std::uint32_t const& offset) -> Philox4x32x10
            {
                return {seed, subsequence, offset};
            }
        };
    } // namespace engine::trait
} // namespace alpaka::rand
