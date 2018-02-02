/**
 * \file
 * Copyright 2015 Benjamin Worpitz
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

// \Hack: Boost.MPL defines BOOST_MPL_CFG_GPU_ENABLED to __host__ __device__ if nvcc is used.
// BOOST_AUTO_TEST_CASE_TEMPLATE and its internals are not GPU enabled but is using boost::mpl::for_each internally.
// For each template parameter this leads to:
// /home/travis/build/boost/boost/mpl/for_each.hpp(78): warning: calling a __host__ function from a __host__ __device__ function is not allowed
// because boost::mpl::for_each has the BOOST_MPL_CFG_GPU_ENABLED attribute but the test internals are pure host methods.
// Because we do not use MPL within GPU code here, we can disable the MPL GPU support.
#define BOOST_MPL_CFG_GPU_ENABLED

#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/Acc.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/stream/Stream.hpp>

#include <boost/predef.h>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wunused-parameter"
#endif
#include <boost/test/unit_test.hpp>
#if BOOST_COMP_CLANG
    #pragma clang diagnostic pop
#endif

BOOST_AUTO_TEST_SUITE(viewStaticAccMem)

using Elem = std::uint32_t;
using Dim = alpaka::dim::DimInt<2u>;
using Size = std::uint32_t;

// These forward declarations are only necessary when you want to access those variables
// from a different compilation unit and should be moved to a common header.
// Here they are used to silence clang`s -Wmissing-variable-declarations warning
// that forces every non-static variable to be declared with extern before the are defined.
extern ALPAKA_STATIC_DEV_MEM_CONSTANT Elem g_constantMemory2DInitialized[3][2];
extern ALPAKA_STATIC_DEV_MEM_CONSTANT Elem g_constantMemory2DUninitialized[3][2];

ALPAKA_STATIC_DEV_MEM_CONSTANT Elem g_constantMemory2DInitialized[3][2] =
    {
        {0u, 1u},
        {2u, 3u},
        {4u, 5u}
    };

ALPAKA_STATIC_DEV_MEM_CONSTANT Elem g_constantMemory2DUninitialized[3][2];

//#############################################################################
//! Uses static device memory on the defined globally for the whole compilation unit.
struct StaticDeviceMemoryTestKernel
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc,
        typename TElem>
    ALPAKA_FN_ACC void operator()(
        TAcc const & acc,
        TElem const * const pConstantMem) const
    {
        auto const gridThreadExtent =
            alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
        auto const gridThreadIdx =
            alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);

        auto const offset = gridThreadExtent[1u] * gridThreadIdx[0u] + gridThreadIdx[1u];
        auto const val = offset;

        BOOST_VERIFY(val == *(pConstantMem + offset));
    }
};

using TestAccs = alpaka::test::acc::EnabledAccs<Dim, Size>;

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    staticDeviceMemoryConstant,
    TAcc,
    TestAccs)
{
    using DevAcc = alpaka::dev::Dev<TAcc>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    DevAcc devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(0u));

    alpaka::vec::Vec<Dim, Size> const extent(3u, 2u);

    alpaka::test::KernelExecutionFixture<TAcc> fixture(extent);

    StaticDeviceMemoryTestKernel kernel;

    //-----------------------------------------------------------------------------
    // initialized static constant device memory
    {
        auto const viewConstantMemInitialized(
            alpaka::mem::view::createStaticDevMemView(
                &g_constantMemory2DInitialized[0u][0u],
                devAcc,
                extent));

        BOOST_REQUIRE_EQUAL(
            true,
            fixture(
                kernel,
                alpaka::mem::view::getPtrNative(viewConstantMemInitialized)));
    }

    //-----------------------------------------------------------------------------
    // uninitialized static constant device memory
    {
        using PltfHost = alpaka::pltf::PltfCpu;
        auto devHost(alpaka::pltf::getDevByIdx<PltfHost>(0u));

        using StreamAcc = alpaka::test::stream::DefaultStream<DevAcc>;
        StreamAcc streamAcc(devAcc);

        std::vector<Elem> const data{0u, 1u, 2u, 3u, 4u, 5u};
        alpaka::mem::view::ViewPlainPtr<decltype(devHost), const Elem, Dim, Size> bufHost(data.data(), devHost, extent);

        auto viewConstantMemUninitialized(
            alpaka::mem::view::createStaticDevMemView(
                &g_constantMemory2DUninitialized[0u][0u],
                devAcc,
                extent));

        alpaka::mem::view::copy(streamAcc, viewConstantMemUninitialized, bufHost, extent);
        alpaka::wait::wait(streamAcc);

        BOOST_REQUIRE_EQUAL(
            true,
            fixture(
                kernel,
                alpaka::mem::view::getPtrNative(viewConstantMemUninitialized)));
    }
}

// These forward declarations are only necessary when you want to access those variables
// from a different compilation unit and should be moved to a common header.
// Here they are used to silence clang`s -Wmissing-variable-declarations warning
// that forces every non-static variable to be declared with extern before the are defined.
extern ALPAKA_STATIC_DEV_MEM_GLOBAL Elem g_globalMemory2DInitialized[3][2];
extern ALPAKA_STATIC_DEV_MEM_GLOBAL Elem g_globalMemory2DUninitialized[3][2];

ALPAKA_STATIC_DEV_MEM_GLOBAL Elem g_globalMemory2DInitialized[3][2] =
    {
        {0u, 1u},
        {2u, 3u},
        {4u, 5u}
    };

ALPAKA_STATIC_DEV_MEM_GLOBAL Elem g_globalMemory2DUninitialized[3][2];

//-----------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE_TEMPLATE(
    staticDeviceMemoryGlobal,
    TAcc,
    TestAccs)
{
    using DevAcc = alpaka::dev::Dev<TAcc>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    DevAcc devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(0u));

    alpaka::vec::Vec<Dim, Size> const extent(3u, 2u);

    alpaka::test::KernelExecutionFixture<TAcc> fixture(extent);

    StaticDeviceMemoryTestKernel kernel;

    //-----------------------------------------------------------------------------
    // initialized static global device memory
    {
        auto const viewGlobalMemInitialized(
            alpaka::mem::view::createStaticDevMemView(
                &g_globalMemory2DInitialized[0u][0u],
                devAcc,
                extent));

        BOOST_REQUIRE_EQUAL(
            true,
            fixture(
                kernel,
                alpaka::mem::view::getPtrNative(viewGlobalMemInitialized)));
    }

    //-----------------------------------------------------------------------------
    // uninitialized static global device memory
    {
        using PltfHost = alpaka::pltf::PltfCpu;
        auto devHost(alpaka::pltf::getDevByIdx<PltfHost>(0u));

        using StreamAcc = alpaka::test::stream::DefaultStream<DevAcc>;
        StreamAcc streamAcc(devAcc);

        std::vector<Elem> const data{0u, 1u, 2u, 3u, 4u, 5u};
        alpaka::mem::view::ViewPlainPtr<decltype(devHost), const Elem, Dim, Size> bufHost(data.data(), devHost, extent);

        auto viewGlobalMemUninitialized(
            alpaka::mem::view::createStaticDevMemView(
                &g_globalMemory2DUninitialized[0u][0u],
                devAcc,
                extent));

        alpaka::mem::view::copy(streamAcc, viewGlobalMemUninitialized, bufHost, extent);
        alpaka::wait::wait(streamAcc);

        BOOST_REQUIRE_EQUAL(
            true,
            fixture(
                kernel,
                alpaka::mem::view::getPtrNative(viewGlobalMemUninitialized)));
    }
}

BOOST_AUTO_TEST_SUITE_END()
