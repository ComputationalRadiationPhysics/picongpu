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

//#############################################################################
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

        DevAcc const devAcc{alpaka::getDevByIdx<PltfAcc>(0u)};
        DevHost const devHost{alpaka::getDevByIdx<PltfHost>(0u)};

        QueueAcc queue{devAcc};

        TestKernel<capacity> kernel;
        TFunctor functor;
        Args args{devAcc};
        Results results{devAcc};

        WorkDiv const workDiv{alpaka::getValidWorkDiv<TAcc>(
            devAcc,
            sizeExtent,
            elementsPerThread,
            false,
            alpaka::GridBlockExtentSubDivRestrictions::Unrestricted)};
        // SETUP COMPLETED.

        // Fill the buffer with random test-numbers.
        alpaka::test::unit::math::fillWithRndArgs<TData>(args, functor, seed);
        for(size_t i = 0; i < Results::capacity; ++i)
            results(i) = static_cast<TData>(std::nan(""));

        // Copy both buffer to the device
        args.copyToDevice(queue);
        results.copyToDevice(queue);

        auto const taskKernel(
            alpaka::createTaskKernel<TAcc>(workDiv, kernel, results.pDevBuffer, functor, args.pDevBuffer));
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
