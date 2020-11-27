/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/test/Check.hpp>
#include <alpaka/test/queue/Queue.hpp>

namespace alpaka
{
    namespace test
    {
        //#############################################################################
        //! The fixture for executing a kernel on a given accelerator.
        template<typename TAcc>
        class KernelExecutionFixture
        {
        public:
            using Acc = TAcc;
            using Dim = alpaka::Dim<Acc>;
            using Idx = alpaka::Idx<Acc>;
            using DevAcc = alpaka::Dev<Acc>;
            using PltfAcc = alpaka::Pltf<DevAcc>;
            using QueueAcc = alpaka::test::DefaultQueue<DevAcc>;
            using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;

        public:
            //-----------------------------------------------------------------------------
            template<typename TExtent>
            KernelExecutionFixture(TExtent const& extent)
                : m_devHost(alpaka::getDevByIdx<PltfCpu>(0u))
                , m_devAcc(alpaka::getDevByIdx<PltfAcc>(0u))
                , m_queue(m_devAcc)
                , m_workDiv(alpaka::getValidWorkDiv<Acc>(
                      m_devAcc,
                      extent,
                      alpaka::Vec<Dim, Idx>::ones(),
                      false,
                      alpaka::GridBlockExtentSubDivRestrictions::Unrestricted))
            {
            }
            //-----------------------------------------------------------------------------
            KernelExecutionFixture(WorkDiv const& workDiv)
                : m_devHost(alpaka::getDevByIdx<PltfCpu>(0u))
                , m_devAcc(alpaka::getDevByIdx<PltfAcc>(0u))
                , m_queue(m_devAcc)
                , m_workDiv(workDiv)
            {
            }
            //-----------------------------------------------------------------------------
            template<typename TKernelFnObj, typename... TArgs>
            auto operator()(TKernelFnObj const& kernelFnObj, TArgs&&... args) -> bool
            {
                // Allocate the result value
                auto bufAccResult(alpaka::allocBuf<bool, Idx>(m_devAcc, static_cast<Idx>(1u)));
                alpaka::memset(m_queue, bufAccResult, static_cast<std::uint8_t>(true), bufAccResult);

                alpaka::exec<Acc>(
                    m_queue,
                    m_workDiv,
                    kernelFnObj,
                    alpaka::getPtrNative(bufAccResult),
                    std::forward<TArgs>(args)...);

                // Copy the result value to the host
                auto bufHostResult(alpaka::allocBuf<bool, Idx>(m_devHost, static_cast<Idx>(1u)));
                alpaka::memcpy(m_queue, bufHostResult, bufAccResult, bufAccResult);
                alpaka::wait(m_queue);

                auto const result(*alpaka::getPtrNative(bufHostResult));

                return result;
            }

        private:
            alpaka::DevCpu m_devHost;
            DevAcc m_devAcc;
            QueueAcc m_queue;
            WorkDiv m_workDiv;
        };
    } // namespace test
} // namespace alpaka
