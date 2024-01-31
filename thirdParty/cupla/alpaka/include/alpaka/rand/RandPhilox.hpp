/* Copyright 2022 Jiří Vyskočil, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/meta/IsArrayOrVector.hpp"
#include "alpaka/rand/Philox/PhiloxSingle.hpp"
#include "alpaka/rand/Philox/PhiloxVector.hpp"
#include "alpaka/rand/Traits.hpp"

#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>

namespace alpaka::rand
{
    /** Most common Philox engine variant, outputs single number
     *
     * This is a variant of the Philox engine generator which outputs a single float. The counter size is \f$4
     * \times 32 = 128\f$ bits. Since the engine returns a single number, the generated result, which has the same
     * size as the counter, has to be stored between invocations. Additionally a 32 bit pointer is stored. The
     * total size of the state is 352 bits = 44 bytes.
     *
     * Ref.: J. K. Salmon, M. A. Moraes, R. O. Dror and D. E. Shaw, "Parallel random numbers: As easy as 1, 2, 3,"
     * SC '11: Proceedings of 2011 International Conference for High Performance Computing, Networking, Storage and
     * Analysis, 2011, pp. 1-12, doi: 10.1145/2063384.2063405.
     *
     * @tparam TAcc Accelerator type as defined in alpaka/acc
     */
    template<typename TAcc>
    class Philox4x32x10 : public concepts::Implements<ConceptRand, Philox4x32x10<TAcc>>
    {
    public:
        using EngineParams = engine::PhiloxParams<4, 32, 10>; ///< Philox algorithm: 10 rounds, 4 numbers of size 32.
        using EngineVariant = engine::PhiloxSingle<TAcc, EngineParams>; ///< Engine outputs a single number

        /** Initialize a new Philox engine
         *
         * @param seed Set the Philox generator key
         * @param subsequence Select a subsequence of size 2^64
         * @param offset Skip \a offset numbers form the start of the subsequence
         */
        ALPAKA_FN_HOST_ACC Philox4x32x10(
            std::uint64_t const seed = 0,
            std::uint64_t const subsequence = 0,
            std::uint64_t const offset = 0)
            : engineVariant(seed, subsequence, offset)
        {
        }

        // STL UniformRandomBitGenerator concept
        // https://en.cppreference.com/w/cpp/named_req/UniformRandomBitGenerator
        using result_type = std::uint32_t;

        ALPAKA_FN_HOST_ACC constexpr auto min() -> result_type
        {
            return 0;
        }

        ALPAKA_FN_HOST_ACC constexpr auto max() -> result_type
        {
            return std::numeric_limits<result_type>::max();
        }

        ALPAKA_FN_HOST_ACC auto operator()() -> result_type
        {
            return engineVariant();
        }

    private:
        EngineVariant engineVariant;
    };

    /** Most common Philox engine variant, outputs a 4-vector of floats
     *
     * This is a variant of the Philox engine generator which outputs a vector containing 4 floats. The counter
     * size is \f$4 \times 32 = 128\f$ bits. Since the engine returns the whole generated vector, it is up to the
     * user to extract individual floats as they need. The benefit is smaller state size since the state does not
     * contain the intermediate results. The total size of the state is 192 bits = 24 bytes.
     *
     * Ref.: J. K. Salmon, M. A. Moraes, R. O. Dror and D. E. Shaw, "Parallel random numbers: As easy as 1, 2, 3,"
     * SC '11: Proceedings of 2011 International Conference for High Performance Computing, Networking, Storage and
     * Analysis, 2011, pp. 1-12, doi: 10.1145/2063384.2063405.
     *
     * @tparam TAcc Accelerator type as defined in alpaka/acc
     */
    template<typename TAcc>
    class Philox4x32x10Vector : public concepts::Implements<ConceptRand, Philox4x32x10Vector<TAcc>>
    {
    public:
        using EngineParams = engine::PhiloxParams<4, 32, 10>;
        using EngineVariant = engine::PhiloxVector<TAcc, EngineParams>;

        /** Initialize a new Philox engine
         *
         * @param seed Set the Philox generator key
         * @param subsequence Select a subsequence of size 2^64
         * @param offset Number of numbers to skip form the start of the subsequence.
         */
        ALPAKA_FN_HOST_ACC Philox4x32x10Vector(
            std::uint32_t const seed = 0,
            std::uint32_t const subsequence = 0,
            std::uint32_t const offset = 0)
            : engineVariant(seed, subsequence, offset)
        {
        }

        template<typename TScalar>
        using ResultContainer = typename EngineVariant::template ResultContainer<TScalar>;

        using ResultInt = std::uint32_t;
        using ResultVec = decltype(std::declval<EngineVariant>()());

        ALPAKA_FN_HOST_ACC constexpr auto min() -> ResultInt
        {
            return 0;
        }

        ALPAKA_FN_HOST_ACC constexpr auto max() -> ResultInt
        {
            return std::numeric_limits<ResultInt>::max();
        }

        ALPAKA_FN_HOST_ACC auto operator()() -> ResultVec
        {
            return engineVariant();
        }

    private:
        EngineVariant engineVariant;
    };

    // The following exists because you "cannot call __device__ function from a __host__ __device__ function"
    // directly, but wrapping that call in a struct is just fine.
    template<typename TEngine>
    struct EngineCallHostAccProxy
    {
        ALPAKA_FN_HOST_ACC auto operator()(TEngine& engine) -> decltype(engine())
        {
            return engine();
        }
    };

    /// TEMP: Distributions to be decided on later. The generator should be compatible with STL as of now.
    template<typename TResult, typename TSfinae = void>
    class UniformReal : public concepts::Implements<ConceptRand, UniformReal<TResult>>
    {
        template<typename TRes, typename TEnable = void>
        struct ResultType
        {
            using type = TRes;
        };

        template<typename TRes>
        struct ResultType<TRes, std::enable_if_t<meta::IsArrayOrVector<TRes>::value>>
        {
            using type = typename TRes::value_type;
        };

        using T = typename ResultType<TResult>::type;
        static_assert(std::is_floating_point_v<T>, "Only floating-point types are supported");

    public:
        ALPAKA_FN_HOST_ACC UniformReal() : UniformReal(0, 1)
        {
        }

        ALPAKA_FN_HOST_ACC UniformReal(T min, T max) : _min(min), _max(max), _range(_max - _min)
        {
        }

        template<typename TEngine>
        ALPAKA_FN_HOST_ACC auto operator()(TEngine& engine) -> TResult
        {
            if constexpr(meta::IsArrayOrVector<TResult>::value)
            {
                auto result = engine();
                T scale = static_cast<T>(1) / engine.max() * _range;
                TResult ret{
                    static_cast<T>(result[0]) * scale + _min,
                    static_cast<T>(result[1]) * scale + _min,
                    static_cast<T>(result[2]) * scale + _min,
                    static_cast<T>(result[3]) * scale + _min};
                return ret;
            }
            else
            {
                // Since it's possible to get a host-only engine here, the call has to go through proxy
                return static_cast<T>(EngineCallHostAccProxy<TEngine>{}(engine)) / engine.max() * _range + _min;
            }

            ALPAKA_UNREACHABLE(TResult{});
        }

    private:
        const T _min;
        const T _max;
        const T _range;
    };
} // namespace alpaka::rand
