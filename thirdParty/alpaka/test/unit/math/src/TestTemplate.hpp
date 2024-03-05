/* Copyright 2022 Jakob Krude, Benjamin Worpitz, Bernhard Manfred Gruber, Sergei Bastrakov, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include "Buffer.hpp"
#include "DataGen.hpp"
#include "Defines.hpp"
#include "Functor.hpp"

#include <alpaka/core/DemangleTypeNames.hpp>
#include <alpaka/math/Complex.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/queue/Queue.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

namespace mathtest
{
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

    //! Helper trait to determine underlying type of real and complex numbers
    template<typename T>
    struct UnderlyingType
    {
        using type = T;
    };

    //! Specialization for complex
    template<typename T>
    struct UnderlyingType<alpaka::Complex<T>>
    {
        using type = T;
    };

    //! Base test template for math unit tests
    //! @tparam TAcc Accelerator.
    //! @tparam TFunctor Functor defined in Functor.hpp.
    template<typename TAcc, typename TFunctor>
    struct TestTemplate
    {
        //! wrappedFunctor is either a TFunctor{} or TFunctor{} wrapped into a host-device lambda
        template<typename TData, typename TWrappedFunctor = TFunctor>
        auto operator()(TWrappedFunctor const& wrappedFunctor = TWrappedFunctor{}) -> void
        {
            std::random_device rd{};
            auto const seed = rd();
            INFO(
                "testing"
                << " acc:" << alpaka::core::demangled<TAcc> << " data type:" << alpaka::core::demangled<TData>
                << " functor:" << alpaka::core::demangled<TWrappedFunctor> << " seed:" << seed);

            // SETUP (defines and initialising)
            // DevAcc is defined in Buffer.hpp too.
            using DevAcc = alpaka::Dev<TAcc>;

            using Dim = alpaka::DimInt<1u>;
            using Idx = std::size_t;
            using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;
            using QueueAcc = alpaka::test::DefaultQueue<DevAcc>;
            using TArgsItem = ArgsItem<TData, TFunctor::arity>;

            static constexpr auto capacity = 1000;

            using Args = Buffer<TAcc, TArgsItem, capacity>;
            using Results = Buffer<TAcc, TData, capacity>;

            // Every functor is executed individual on one kernel.
            static constexpr size_t elementsPerThread = 1u;
            static constexpr size_t sizeExtent = 1u;

            auto const platformAcc = alpaka::Platform<TAcc>{};
            auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);
            auto const platformHost = alpaka::PlatformCpu{};
            auto const devHost = alpaka::getDevByIdx(platformHost, 0);

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
            fillWithRndArgs<TData>(args, functor, seed);
            using Underlying = typename UnderlyingType<TData>::type;
            for(size_t i = 0; i < Results::capacity; ++i)
                results(i) = static_cast<Underlying>(std::nan(""));

            // Copy both buffer to the device
            args.copyToDevice(queue);
            results.copyToDevice(queue);

            // Enqueue the kernel execution task.
            auto const taskKernel
                = alpaka::createTaskKernel<TAcc>(workDiv, kernel, results.pDevBuffer, wrappedFunctor, args.pDevBuffer);
            alpaka::enqueue(queue, taskKernel);

            // Copy back the results (encapsulated in the buffer class).
            results.copyFromDevice(queue);
            alpaka::wait(queue);
            std::cout.precision(std::numeric_limits<Underlying>::digits10 + 1);

            INFO("Operator: " << functor);
            INFO("Type: " << alpaka::core::demangled<TData>); // Compiler specific.
#if ALPAKA_DEBUG_FULL
            INFO(
                "The args buffer: \n"
                << std::setprecision(std::numeric_limits<Underlying>::digits10 + 1) << args << "\n");
#endif
            for(size_t i = 0; i < Args::capacity; ++i)
            {
                TData std_result = functor(args(i));
                INFO("Idx i: " << i << " computed : " << results(i) << " vs expected: " << std_result);
                REQUIRE(isApproxEqual(results(i), std_result));
            }
        }

        //! Approximate comparison of real numbers
        template<typename T>
        static bool isApproxEqual(T const& a, T const& b)
        {
            return a == Catch::Approx(b).margin(std::numeric_limits<T>::epsilon());
        }

        //! Is complex number considered finite for math testing.
        //! Complex numbers with absolute value close to max() of underlying type are considered infinite.
        //! The reason is, CUDA/HIP implementation cannot guarantee correct treatment of such values due to
        //! implementing some math functions via calls to others. For extreme values of arguments, it could cause
        //! intermediate results to become infinite or NaN. So in this function we consider all large enough values to
        //! be effectively infinite and equivalent to one another. Thus, the tests do not concern accuracy for extreme
        //! values. However, they still check the implementation for "reasonable" values.
        template<typename T>
        static bool isFinite(alpaka::Complex<T> const& z)
        {
            auto const absValue = abs(z);
            auto const maxAbs = static_cast<T>(0.1) * std::numeric_limits<T>::max();
            return std::isfinite(absValue) && (absValue < maxAbs);
        }

        //! Approximate comparison of complex numbers
        template<typename T>
        static bool isApproxEqual(alpaka::Complex<T> const& a, alpaka::Complex<T> const& b)
        {
            // Consider all infinite values equal, @see comment at isFinite()
            if(!isFinite(a) && !isFinite(b))
                return true;
            // For the same reason use relative difference comparison with a large margin
            auto const scalingFactor = static_cast<T>(std::is_same_v<T, float> ? 1.1e4 : 1.1e6);
            auto const marginValue = scalingFactor * std::numeric_limits<T>::epsilon();
            return (a.real() == Catch::Approx(b.real()).margin(marginValue).epsilon(marginValue))
                   && (a.imag() == Catch::Approx(b.imag()).margin(marginValue).epsilon(marginValue));
        }
    };
} // namespace mathtest
