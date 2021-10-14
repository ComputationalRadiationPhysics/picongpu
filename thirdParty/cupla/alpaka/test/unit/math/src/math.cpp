/** Copyright 2019 Jakob Krude, Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "Buffer.hpp"
#include "DataGen.hpp"
#include "Defines.hpp"
#include "Functor.hpp"

#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/catch.hpp>

using TestAccs = alpaka::test::EnabledAccs<alpaka::DimInt<1u>, std::size_t>;

using Functors
    = alpaka::meta::Concatenate<alpaka::test::unit::math::UnaryFunctors, alpaka::test::unit::math::BinaryFunctors>;

using TestAccFunctorTuples = alpaka::meta::CartesianProduct<std::tuple, TestAccs, Functors>;

using DataTypes = std::tuple<float, double>;

template<std::size_t TCapacity>
struct TestKernel
{
    //! @tparam TAcc Accelerator.
    //! @tparam TFunctor Functor defined in Functor.hpp.
    //! @param acc Accelerator given from alpaka.
    //! @param functor Accessible with operator().
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc, typename TResults, typename TFunctor, typename TArgs>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TResults* results, TFunctor const& functor, TArgs const* args)
        const noexcept -> void
    {
        for(size_t i = 0; i < TCapacity; ++i)
        {
            results[i] = functor(args[i], acc);
        }
    }
};

template<typename TAcc, typename TFunctor>
struct TestTemplate
{
    template<typename TData>
    auto operator()() -> void
    {
        std::random_device rd{};
        auto const seed = rd();
        std::cout << "testing"
                  << " acc:" << typeid(TAcc).name() << " data type:" << typeid(TData).name()
                  << " functor:" << typeid(TFunctor).name() << " seed:" << seed << std::endl;

        // SETUP (defines and initialising)
        // DevAcc and DevHost are defined in Buffer.hpp too.
        using DevAcc = alpaka::Dev<TAcc>;
        using DevHost = alpaka::DevCpu;
        using PltfAcc = alpaka::Pltf<DevAcc>;
        using PltfHost = alpaka::Pltf<DevHost>;

        using Dim = alpaka::DimInt<1u>;
        using Idx = std::size_t;
        using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;
        using QueueAcc = alpaka::test::DefaultQueue<DevAcc>;
        using TArgsItem = alpaka::test::unit::math::ArgsItem<TData, TFunctor::arity>;

        static constexpr auto capacity = 1000;

        using Args = alpaka::test::unit::math::Buffer<TAcc, TArgsItem, capacity>;
        using Results = alpaka::test::unit::math::Buffer<TAcc, TData, capacity>;

        // Every functor is executed individual on one kernel.
        static constexpr size_t elementsPerThread = 1u;
        static constexpr size_t sizeExtent = 1u;

        DevAcc const devAcc = alpaka::getDevByIdx<PltfAcc>(0u);
        DevHost const devHost = alpaka::getDevByIdx<PltfHost>(0u);

        QueueAcc queue{devAcc};

        TestKernel<capacity> kernel;
        TFunctor functor;
        Args args{devAcc};
        Results results{devAcc};

        WorkDiv const workDiv = alpaka::getValidWorkDiv<TAcc>(
            devAcc,
            sizeExtent,
            elementsPerThread,
            false,
            alpaka::GridBlockExtentSubDivRestrictions::Unrestricted);
        // SETUP COMPLETED.

        // Fill the buffer with random test-numbers.
        alpaka::test::unit::math::fillWithRndArgs<TData>(args, functor, seed);
        for(size_t i = 0; i < Results::capacity; ++i)
            results(i) = static_cast<TData>(std::nan(""));

        // Copy both buffer to the device
        args.copyToDevice(queue);
        results.copyToDevice(queue);

        auto const taskKernel
            = alpaka::createTaskKernel<TAcc>(workDiv, kernel, results.pDevBuffer, functor, args.pDevBuffer);
        // Enqueue the kernel execution task.
        alpaka::enqueue(queue, taskKernel);
        // Copy back the results (encapsulated in the buffer class).
        results.copyFromDevice(queue);
        alpaka::wait(queue);
        std::cout.precision(std::numeric_limits<TData>::digits10 + 1);

        INFO("Operator: " << functor)
        INFO("Type: " << typeid(TData).name()) // Compiler specific.
#if ALPAKA_DEBUG_FULL
        INFO("The args buffer: \n" << std::setprecision(std::numeric_limits<TData>::digits10 + 1) << args << "\n")
#endif
        for(size_t i = 0; i < Args::capacity; ++i)
        {
            INFO("Idx i: " << i)
            TData std_result = functor(args(i));
            REQUIRE(results(i) == Approx(std_result));
        }
    }
};

TEMPLATE_LIST_TEST_CASE("mathOps", "[math] [operator]", TestAccFunctorTuples)
{
    /*
     * All alpaka::math:: functions are tested here except sincos.
     * The function will be called with a buffer from the custom Buffer class.
     * This argument Buffer contains ArgsItems from Defines.hpp and can be
     * accessed with the overloaded operator().
     * The args Buffer looks similar like [[0, 1], [2, 3], [4, 5]],
     * where every sub-list makes one functor-call so the result Buffer would be:
     * [f(0, 1), f(2, 3), f(4, 5)].
     * The results are saved in a different Buffer witch contains plain data.
     * The results are than compared to the result of a std:: implementation.
     * The default result is nan and should fail a test.
     *
     * BE AWARE that:
     * - ALPAKA_CUDA_FAST_MATH should be disabled
     * - not all casts between float and double can be detected.
     * - no explicit edge cases are tested, rather than 0, maximum and minimum
     *   - but it is easy to add a new Range:: enum-type with custom edge cases
     *  - some tests may fail if ALPAKA_CUDA_FAST_MATH is turned on
     * - nan typically fails every test, but could be normal defined behaviour
     * - inf/-inf typically dont fail a test
     * - for easy debugging the << operator is overloaded for Buffer objects
     * - arguments are generated between 0 and 1000
     *     and the default argument-buffer-extent is 1000
     * The arguments are generated in DataGen.hpp and can easily be modified.
     * The arguments depend on the Range:: enum-type specified for each functor.
     * ----------------------------------------------------------------------
     * - each functor has an arity and a array of ranges
     *     - there is one args Buffer and one results Buffer
     *         - each buffer encapsulated the host/device communication
     *         - as well as the data access and the initialisation
     * - all operators are tested independent, one per kernel
     * - tests the results against the std implementation ( catch REQUIRES)
     *
     * TestKernel
     * - uses the alpaka::math:: option from the functor
     * - uses the device-buffer  option from the args
     *
     * EXTENSIBILITY:
     * - Add new operators in Functor.hpp and add them to the ...Functors tuple.
     * - Add a new Range:: enum-type in Defines.hpp
     *     - specify a fill-method in DataGen.hpp
     * - Add a new Arity:: enum-type in Defines.hpp
     *     - add a matching operator() function in Functor.hpp,
     *     - add a new ...Functors tuple
     *     - call alpaka::meta::forEachType with the tuple in ForEachFunctor
     */

    using Acc = std::tuple_element_t<0u, TestType>;
    using Functor = std::tuple_element_t<1u, TestType>;

    alpaka::meta::forEachType<DataTypes>(TestTemplate<Acc, Functor>());
}

namespace custom
{
    enum Custom
    {
        Abs,
        Acos,
        Asin,
        Atan,
        Atan2,
        Cbrt,
        Ceil,
        Cos,
        Erf,
        Exp,
        Floor,
        Fmod,
        Log,
        Max,
        Min,
        Pow,
        Remainder,
        Round,
        Lround,
        Llround,
        Rsqrt,
        Sin,
        Sincos,
        Sqrt,
        Tan,
        Trunc,

        Arg1 = 1024,
        Arg2 = 2048,
        Arg3 = 4096,
    };

    // struct Custom
    //{
    //};

    ALPAKA_FN_HOST_ACC auto abs(Custom c);
    ALPAKA_FN_HOST_ACC auto abs(Custom c)
    {
        return Custom::Abs | c;
    }

    ALPAKA_FN_HOST_ACC auto acos(Custom c);
    ALPAKA_FN_HOST_ACC auto acos(Custom c)
    {
        return Custom::Acos | c;
    }

    ALPAKA_FN_HOST_ACC auto asin(Custom c);
    ALPAKA_FN_HOST_ACC auto asin(Custom c)
    {
        return Custom::Asin | c;
    }

    ALPAKA_FN_HOST_ACC auto atan(Custom c);
    ALPAKA_FN_HOST_ACC auto atan(Custom c)
    {
        return Custom::Atan | c;
    }

    ALPAKA_FN_HOST_ACC auto atan2(Custom a, Custom b);
    ALPAKA_FN_HOST_ACC auto atan2(Custom a, Custom b)
    {
        return Custom::Atan2 | a | b;
    }

    ALPAKA_FN_HOST_ACC auto cbrt(Custom c);
    ALPAKA_FN_HOST_ACC auto cbrt(Custom c)
    {
        return Custom::Cbrt | c;
    }

    ALPAKA_FN_HOST_ACC auto ceil(Custom c);
    ALPAKA_FN_HOST_ACC auto ceil(Custom c)
    {
        return Custom::Ceil | c;
    }

    ALPAKA_FN_HOST_ACC auto cos(Custom c);
    ALPAKA_FN_HOST_ACC auto cos(Custom c)
    {
        return Custom::Cos | c;
    }

    ALPAKA_FN_HOST_ACC auto erf(Custom c);
    ALPAKA_FN_HOST_ACC auto erf(Custom c)
    {
        return Custom::Erf | c;
    }

    ALPAKA_FN_HOST_ACC auto exp(Custom c);
    ALPAKA_FN_HOST_ACC auto exp(Custom c)
    {
        return Custom::Exp | c;
    }

    ALPAKA_FN_HOST_ACC auto floor(Custom c);
    ALPAKA_FN_HOST_ACC auto floor(Custom c)
    {
        return Custom::Floor | c;
    }

    ALPAKA_FN_HOST_ACC auto fmod(Custom a, Custom b);
    ALPAKA_FN_HOST_ACC auto fmod(Custom a, Custom b)
    {
        return Custom::Fmod | a | b;
    }

    ALPAKA_FN_HOST_ACC auto log(Custom c);
    ALPAKA_FN_HOST_ACC auto log(Custom c)
    {
        return Custom::Log | c;
    }

    ALPAKA_FN_HOST_ACC auto max(Custom a, Custom b);
    ALPAKA_FN_HOST_ACC auto max(Custom a, Custom b)
    {
        return Custom::Max | a | b;
    }

    ALPAKA_FN_HOST_ACC auto min(Custom a, Custom b);
    ALPAKA_FN_HOST_ACC auto min(Custom a, Custom b)
    {
        return Custom::Min | a | b;
    }

    ALPAKA_FN_HOST_ACC auto pow(Custom a, Custom b);
    ALPAKA_FN_HOST_ACC auto pow(Custom a, Custom b)
    {
        return Custom::Pow | a | b;
    }

    ALPAKA_FN_HOST_ACC auto remainder(Custom a, Custom b);
    ALPAKA_FN_HOST_ACC auto remainder(Custom a, Custom b)
    {
        return Custom::Remainder | a | b;
    }

    ALPAKA_FN_HOST_ACC auto round(Custom c);
    ALPAKA_FN_HOST_ACC auto round(Custom c)
    {
        return Custom::Round | c;
    }

    ALPAKA_FN_HOST_ACC auto lround(Custom c);
    ALPAKA_FN_HOST_ACC auto lround(Custom c)
    {
        return Custom::Lround | c;
    }

    ALPAKA_FN_HOST_ACC auto llround(Custom c);
    ALPAKA_FN_HOST_ACC auto llround(Custom c)
    {
        return Custom::Llround | c;
    }

    ALPAKA_FN_HOST_ACC auto rsqrt(Custom c);
    ALPAKA_FN_HOST_ACC auto rsqrt(Custom c)
    {
        return Custom::Rsqrt | c;
    }

    ALPAKA_FN_HOST_ACC auto sin(Custom c);
    ALPAKA_FN_HOST_ACC auto sin(Custom c)
    {
        return Custom::Sin | c;
    }

    ALPAKA_FN_HOST_ACC void sincos(Custom c, Custom& a, Custom& b);
    ALPAKA_FN_HOST_ACC void sincos(Custom c, Custom& a, Custom& b)
    {
        a = static_cast<Custom>(Custom::Sincos | c | Custom::Arg2);
        b = static_cast<Custom>(Custom::Sincos | c | Custom::Arg3);
    }

    ALPAKA_FN_HOST_ACC auto sqrt(Custom c);
    ALPAKA_FN_HOST_ACC auto sqrt(Custom c)
    {
        return Custom::Sqrt | c;
    }

    ALPAKA_FN_HOST_ACC auto tan(Custom c);
    ALPAKA_FN_HOST_ACC auto tan(Custom c)
    {
        return Custom::Tan | c;
    }

    ALPAKA_FN_HOST_ACC auto trunc(Custom c);
    ALPAKA_FN_HOST_ACC auto trunc(Custom c)
    {
        return Custom::Trunc | c;
    }
} // namespace custom

struct AdlKernel
{
    template<typename Acc>
    ALPAKA_FN_ACC void operator()(Acc const& acc, bool* success) const noexcept
    {
        using custom::Custom;

        ALPAKA_CHECK(*success, alpaka::math::abs(acc, Custom::Arg1) == (Custom::Abs | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::acos(acc, Custom::Arg1) == (Custom::Acos | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::asin(acc, Custom::Arg1) == (Custom::Asin | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::atan(acc, Custom::Arg1) == (Custom::Atan | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::cbrt(acc, Custom::Arg1) == (Custom::Cbrt | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::ceil(acc, Custom::Arg1) == (Custom::Ceil | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::cos(acc, Custom::Arg1) == (Custom::Cos | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::erf(acc, Custom::Arg1) == (Custom::Erf | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::exp(acc, Custom::Arg1) == (Custom::Exp | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::floor(acc, Custom::Arg1) == (Custom::Floor | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::log(acc, Custom::Arg1) == (Custom::Log | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::round(acc, Custom::Arg1) == (Custom::Round | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::lround(acc, Custom::Arg1) == (Custom::Lround | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::llround(acc, Custom::Arg1) == (Custom::Llround | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::rsqrt(acc, Custom::Arg1) == (Custom::Rsqrt | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::sin(acc, Custom::Arg1) == (Custom::Sin | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::sqrt(acc, Custom::Arg1) == (Custom::Sqrt | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::tan(acc, Custom::Arg1) == (Custom::Tan | Custom::Arg1));
        ALPAKA_CHECK(*success, alpaka::math::trunc(acc, Custom::Arg1) == (Custom::Trunc | Custom::Arg1));

        ALPAKA_CHECK(
            *success,
            alpaka::math::atan2(acc, Custom::Arg1, Custom::Arg2) == (Custom::Atan2 | Custom::Arg1 | Custom::Arg2));
        ALPAKA_CHECK(
            *success,
            alpaka::math::fmod(acc, Custom::Arg1, Custom::Arg2) == (Custom::Fmod | Custom::Arg1 | Custom::Arg2));
        ALPAKA_CHECK(
            *success,
            alpaka::math::max(acc, Custom::Arg1, Custom::Arg2) == (Custom::Max | Custom::Arg1 | Custom::Arg2));
        ALPAKA_CHECK(
            *success,
            alpaka::math::min(acc, Custom::Arg1, Custom::Arg2) == (Custom::Min | Custom::Arg1 | Custom::Arg2));
        ALPAKA_CHECK(
            *success,
            alpaka::math::pow(acc, Custom::Arg1, Custom::Arg2) == (Custom::Pow | Custom::Arg1 | Custom::Arg2));
        ALPAKA_CHECK(
            *success,
            alpaka::math::remainder(acc, Custom::Arg1, Custom::Arg2)
                == (Custom::Remainder | Custom::Arg1 | Custom::Arg2));

        Custom a, b;
        alpaka::math::sincos(acc, Custom::Arg1, a, b);
        ALPAKA_CHECK(*success, a == (Custom::Sincos | Custom::Arg1 | Custom::Arg2));
        ALPAKA_CHECK(*success, b == (Custom::Sincos | Custom::Arg1 | Custom::Arg3));
    }
};

TEMPLATE_LIST_TEST_CASE("mathOps", "[math] [operator] [adl]", TestAccs)
{
    using Acc = TestType;
    using Dim = alpaka::Dim<Acc>;
    using Idx = alpaka::Idx<Acc>;
    auto fixture = alpaka::test::KernelExecutionFixture<Acc>{alpaka::Vec<Dim, Idx>::ones()};
    REQUIRE(fixture(AdlKernel{}));
}
