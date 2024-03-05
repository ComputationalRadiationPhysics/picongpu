/* Copyright 2023 Luca Ferragina, Aurora Perego, Jan Stephan, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/BoostPredef.hpp"
#include "alpaka/core/Concepts.hpp"
#include "alpaka/dev/DevGenericSycl.hpp"
#include "alpaka/rand/Traits.hpp"

#if defined(ALPAKA_ACC_SYCL_ENABLED) && !defined(ALPAKA_DISABLE_VENDOR_RNG)

// Backend specific imports.
#    include <sycl/sycl.hpp>
#    if BOOST_COMP_CLANG
#        pragma clang diagnostic push
#        pragma clang diagnostic ignored "-Wcast-align"
#        pragma clang diagnostic ignored "-Wcast-qual"
#        pragma clang diagnostic ignored "-Wextra-semi"
#        pragma clang diagnostic ignored "-Wfloat-equal"
#        pragma clang diagnostic ignored "-Wold-style-cast"
#        pragma clang diagnostic ignored "-Wreserved-identifier"
#        pragma clang diagnostic ignored "-Wreserved-macro-identifier"
#        pragma clang diagnostic ignored "-Wsign-compare"
#        pragma clang diagnostic ignored "-Wundef"
#    endif
#    include <oneapi/dpl/random>
#    if BOOST_COMP_CLANG
#        pragma clang diagnostic pop
#    endif

#    include <type_traits>

namespace alpaka::rand
{
    //! The SYCL rand implementation.
    template<typename TDim>
    struct RandGenericSycl : concepts::Implements<ConceptRand, RandGenericSycl<TDim>>
    {
        explicit RandGenericSycl(sycl::nd_item<TDim::value> my_item) : m_item_rand{my_item}
        {
        }

        sycl::nd_item<TDim::value> m_item_rand;
    };

#    if !defined(ALPAKA_HOST_ONLY)
    namespace distribution::sycl_rand
    {
        //! The SYCL random number floating point normal distribution.
        template<typename T>
        struct NormalReal;

        //! The SYCL random number uniform distribution.
        template<typename T>
        struct Uniform;
    } // namespace distribution::sycl_rand

    namespace engine::sycl_rand
    {
        //! The SYCL linear congruential random number generator engine.
        template<typename TDim>
        class Minstd
        {
        public:
            // After calling this constructor the instance is not valid initialized and
            // need to be overwritten with a valid object
            Minstd() = default;

            Minstd(RandGenericSycl<TDim> rand, std::uint32_t const& seed)
            {
                oneapi::dpl::minstd_rand engine(seed, rand.m_item_rand.get_global_linear_id());
                rng_engine = engine;
            }

        private:
            template<typename T>
            friend struct distribution::sycl_rand::NormalReal;
            template<typename T>
            friend struct distribution::sycl_rand::Uniform;

            oneapi::dpl::minstd_rand rng_engine;

        public:
            using result_type = float;

            ALPAKA_FN_HOST_ACC static result_type min()
            {
                return std::numeric_limits<result_type>::min();
            }

            ALPAKA_FN_HOST_ACC static result_type max()
            {
                return std::numeric_limits<result_type>::max();
            }

            result_type operator()()
            {
                oneapi::dpl::uniform_real_distribution<float> distr;
                return distr(rng_engine);
            }
        };
    } // namespace engine::sycl_rand

    namespace distribution::sycl_rand
    {

        //! The SYCL random number double normal distribution.
        template<typename F>
        struct NormalReal
        {
            static_assert(std::is_floating_point_v<F>);

            template<typename TEngine>
            auto operator()(TEngine& engine) -> F
            {
                oneapi::dpl::normal_distribution<F> distr;
                return distr(engine.rng_engine);
            }
        };

        //! The SYCL random number float uniform distribution.
        template<typename T>
        struct Uniform
        {
            static_assert(std::is_floating_point_v<T> || std::is_unsigned_v<T>);

            template<typename TEngine>
            auto operator()(TEngine& engine) -> T
            {
                if constexpr(std::is_floating_point_v<T>)
                {
                    oneapi::dpl::uniform_real_distribution<T> distr;
                    return distr(engine.rng_engine);
                }
                else
                {
                    oneapi::dpl::uniform_int_distribution<T> distr;
                    return distr(engine.rng_engine);
                }
            }
        };
    } // namespace distribution::sycl_rand

    namespace distribution::trait
    {
        //! The SYCL random number float normal distribution get trait specialization.
        template<typename TDim, typename T>
        struct CreateNormalReal<RandGenericSycl<TDim>, T, std::enable_if_t<std::is_floating_point_v<T>>>
        {
            static auto createNormalReal(RandGenericSycl<TDim> const& /*rand*/) -> sycl_rand::NormalReal<T>
            {
                return {};
            }
        };

        //! The SYCL random number float uniform distribution get trait specialization.
        template<typename TDim, typename T>
        struct CreateUniformReal<RandGenericSycl<TDim>, T, std::enable_if_t<std::is_floating_point_v<T>>>
        {
            static auto createUniformReal(RandGenericSycl<TDim> const& /*rand*/) -> sycl_rand::Uniform<T>
            {
                return {};
            }
        };

        //! The SYCL random number integer uniform distribution get trait specialization.
        template<typename TDim, typename T>
        struct CreateUniformUint<RandGenericSycl<TDim>, T, std::enable_if_t<std::is_integral_v<T>>>
        {
            static auto createUniformUint(RandGenericSycl<TDim> const& /*rand*/) -> sycl_rand::Uniform<T>
            {
                return {};
            }
        };
    } // namespace distribution::trait

    namespace engine::trait
    {
        //! The SYCL random number default generator get trait specialization.
        template<typename TDim>
        struct CreateDefault<RandGenericSycl<TDim>>
        {
            static auto createDefault(
                RandGenericSycl<TDim> const& rand,
                std::uint32_t const& seed = 0,
                std::uint32_t const& /* subsequence */ = 0,
                std::uint32_t const& /* offset */ = 0) -> sycl_rand::Minstd<TDim>
            {
                return {rand, seed};
            }
        };
    } // namespace engine::trait
#    endif
} // namespace alpaka::rand

#endif
