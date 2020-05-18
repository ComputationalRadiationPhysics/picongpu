/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/core/Common.hpp>

#include <alpaka/test/acc/TestAccs.hpp>
#include <alpaka/test/KernelExecutionFixture.hpp>
#include <alpaka/test/queue/Queue.hpp>
#include <alpaka/meta/ForEachType.hpp>
#include <alpaka/core/BoostPredef.hpp>

#include <catch2/catch.hpp>

using Elem = std::uint32_t;
using Dim = alpaka::dim::DimInt<2u>;
using Idx = std::uint32_t;

// These forward declarations are only necessary when you want to access those variables
// from a different compilation unit and should be moved to a common header.
// Here they are used to silence clang`s -Wmissing-variable-declarations warning
// that forces every non-static variable to be declared with extern before the are defined.
extern ALPAKA_STATIC_ACC_MEM_CONSTANT Elem g_constantMemory2DInitialized[3][2];
extern ALPAKA_STATIC_ACC_MEM_CONSTANT Elem g_constantMemory2DUninitialized[3][2];

ALPAKA_STATIC_ACC_MEM_CONSTANT Elem g_constantMemory2DInitialized[3][2] =
    {
        {0u, 1u},
        {2u, 3u},
        {4u, 5u}
    };

ALPAKA_STATIC_ACC_MEM_CONSTANT Elem g_constantMemory2DUninitialized[3][2];

//#############################################################################
//! Uses static device memory on the accelerator defined globally for the whole compilation unit.
struct StaticDeviceMemoryTestKernel
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc,
        typename TElem>
    ALPAKA_FN_ACC void operator()(
        TAcc const & acc,
        bool * success,
        TElem const * const pConstantMem) const
    {
        auto const gridThreadExtent =
            alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
        auto const gridThreadIdx =
            alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);

        auto const offset = gridThreadExtent[1u] * gridThreadIdx[0u] + gridThreadIdx[1u];
        auto const val = offset;

        ALPAKA_CHECK(*success, val == *(pConstantMem + offset));
    }
};

using TestAccs = alpaka::test::acc::EnabledAccs<Dim, Idx>;

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE( "staticDeviceMemoryGlobal", "[viewStaticAccMem]", TestAccs)
{
    using Acc = TestType;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    DevAcc devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(0u));

    alpaka::vec::Vec<Dim, Idx> const extent(3u, 2u);

    alpaka::test::KernelExecutionFixture<Acc> fixture(extent);

    StaticDeviceMemoryTestKernel kernel;

    //-----------------------------------------------------------------------------
    // FIXME: constant memory in HIP is still not working
#if !BOOST_COMP_HIP
    // initialized static constant device memory
    {
        auto const viewConstantMemInitialized(
            alpaka::mem::view::createStaticDevMemView(
                &g_constantMemory2DInitialized[0u][0u],
                devAcc,
                extent));

        REQUIRE(fixture(
            kernel,
            alpaka::mem::view::getPtrNative(viewConstantMemInitialized)));
    }
    //-----------------------------------------------------------------------------
    // uninitialized static constant device memory
    {
        using PltfHost = alpaka::pltf::PltfCpu;
        auto devHost(alpaka::pltf::getDevByIdx<PltfHost>(0u));

        using QueueAcc = alpaka::test::queue::DefaultQueue<DevAcc>;
        QueueAcc queueAcc(devAcc);

        std::vector<Elem> const data{0u, 1u, 2u, 3u, 4u, 5u};
        alpaka::mem::view::ViewPlainPtr<decltype(devHost), const Elem, Dim, Idx> bufHost(data.data(), devHost, extent);

        auto viewConstantMemUninitialized(
            alpaka::mem::view::createStaticDevMemView(
                &g_constantMemory2DUninitialized[0u][0u],
                devAcc,
                extent));

        alpaka::mem::view::copy(queueAcc, viewConstantMemUninitialized, bufHost, extent);
        alpaka::wait::wait(queueAcc);

        REQUIRE(fixture(
            kernel,
            alpaka::mem::view::getPtrNative(viewConstantMemUninitialized)));
    }
#endif
}

// These forward declarations are only necessary when you want to access those variables
// from a different compilation unit and should be moved to a common header.
// Here they are used to silence clang`s -Wmissing-variable-declarations warning
// that forces every non-static variable to be declared with extern before the are defined.
extern ALPAKA_STATIC_ACC_MEM_GLOBAL Elem g_globalMemory2DInitialized[3][2];
extern ALPAKA_STATIC_ACC_MEM_GLOBAL Elem g_globalMemory2DUninitialized[3][2];

ALPAKA_STATIC_ACC_MEM_GLOBAL Elem g_globalMemory2DInitialized[3][2] =
    {
        {0u, 1u},
        {2u, 3u},
        {4u, 5u}
    };

ALPAKA_STATIC_ACC_MEM_GLOBAL Elem g_globalMemory2DUninitialized[3][2];

//-----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE( "staticDeviceMemoryConstant", "[viewStaticAccMem]", TestAccs)
{
    using Acc = TestType;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    DevAcc devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(0u));

    alpaka::vec::Vec<Dim, Idx> const extent(3u, 2u);

    alpaka::test::KernelExecutionFixture<Acc> fixture(extent);

    StaticDeviceMemoryTestKernel kernel;

    //-----------------------------------------------------------------------------
    // FIXME: static device memory in HIP is still not working
#if !BOOST_COMP_HIP
    // initialized static global device memory
    {
        auto const viewGlobalMemInitialized(
            alpaka::mem::view::createStaticDevMemView(
                &g_globalMemory2DInitialized[0u][0u],
                devAcc,
                extent));

        REQUIRE(
            fixture(
                kernel,
                alpaka::mem::view::getPtrNative(viewGlobalMemInitialized)));
    }

    //-----------------------------------------------------------------------------
    // uninitialized static global device memory
    {
        using PltfHost = alpaka::pltf::PltfCpu;
        auto devHost(alpaka::pltf::getDevByIdx<PltfHost>(0u));

        using QueueAcc = alpaka::test::queue::DefaultQueue<DevAcc>;
        QueueAcc queueAcc(devAcc);

        std::vector<Elem> const data{0u, 1u, 2u, 3u, 4u, 5u};
        alpaka::mem::view::ViewPlainPtr<decltype(devHost), const Elem, Dim, Idx> bufHost(data.data(), devHost, extent);

        auto viewGlobalMemUninitialized(
            alpaka::mem::view::createStaticDevMemView(
                &g_globalMemory2DUninitialized[0u][0u],
                devAcc,
                extent));

        alpaka::mem::view::copy(queueAcc, viewGlobalMemUninitialized, bufHost, extent);
        alpaka::wait::wait(queueAcc);

        REQUIRE(
            fixture(
                kernel,
                alpaka::mem::view::getPtrNative(viewGlobalMemUninitialized)));
    }
#endif
}
