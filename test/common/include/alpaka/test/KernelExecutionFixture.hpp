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

#include <alpaka/alpaka.hpp>

#include <alpaka/test/Check.hpp>
#include <alpaka/test/queue/Queue.hpp>

namespace alpaka
{
    namespace test
    {
        //#############################################################################
        //! The fixture for executing a kernel on a given accelerator.
        template<
            typename TAcc>
        class KernelExecutionFixture
        {
        public:
            using Acc = TAcc;
            using Dim = alpaka::dim::Dim<Acc>;
            using Idx = alpaka::idx::Idx<Acc>;
            using DevAcc = alpaka::dev::Dev<Acc>;
            using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
            using QueueAcc = alpaka::test::queue::DefaultQueue<DevAcc>;

        public:
            //-----------------------------------------------------------------------------
            template<
                typename TExtent>
            KernelExecutionFixture(
                TExtent const & extent) :
                    m_devHost(alpaka::pltf::getDevByIdx<pltf::PltfCpu>(0u)),
                    m_devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(0u)),
                    m_queue(m_devAcc),
                    m_workDiv(
                        alpaka::workdiv::getValidWorkDiv<Acc>(
                            m_devAcc,
                            extent,
                            alpaka::vec::Vec<Dim, Idx>::ones(),
                            false,
                            alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted))
            {}
            //-----------------------------------------------------------------------------
            template<
                typename TKernelFnObj,
                typename... TArgs>
            auto operator()(
                TKernelFnObj const & kernelFnObj,
                TArgs const & ... args)
            -> bool
            {
                // Allocate the result value
                auto bufAccResult(alpaka::mem::buf::alloc<bool, Idx>(m_devAcc, static_cast<Idx>(1u)));
                alpaka::mem::view::set(
                    m_queue,
                    bufAccResult,
                    static_cast<std::uint8_t>(true),
                    bufAccResult);

                alpaka::kernel::exec<Acc>(
                    m_queue,
                    m_workDiv,
                    kernelFnObj,
                    alpaka::mem::view::getPtrNative(bufAccResult),
                    args...);

                // Copy the result value to the host
                auto bufHostResult(alpaka::mem::buf::alloc<bool, Idx>(m_devHost, static_cast<Idx>(1u)));
                alpaka::mem::view::copy(m_queue, bufHostResult, bufAccResult, bufAccResult);
                alpaka::wait::wait(m_queue);

                auto const result(*alpaka::mem::view::getPtrNative(bufHostResult));

                return result;
            }

        private:
            alpaka::dev::DevCpu m_devHost;
            DevAcc m_devAcc;
            QueueAcc m_queue;
            alpaka::workdiv::WorkDivMembers<Dim, Idx> m_workDiv;
        };
    }
}
