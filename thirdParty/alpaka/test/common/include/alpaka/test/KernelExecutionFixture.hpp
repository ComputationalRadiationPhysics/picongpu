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
#include <alpaka/test/stream/Stream.hpp>

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
            using Size = alpaka::size::Size<Acc>;
            using DevAcc = alpaka::dev::Dev<Acc>;
            using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
            using StreamAcc = alpaka::test::stream::DefaultStream<DevAcc>;

        public:
            //-----------------------------------------------------------------------------
            template<
                typename TExtent>
            KernelExecutionFixture(
                TExtent const & extent) :
                    m_devHost(alpaka::pltf::getDevByIdx<pltf::PltfCpu>(0u)),
                    m_devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(0u)),
                    m_stream(m_devAcc),
                    m_workDiv(
                        alpaka::workdiv::getValidWorkDiv<Acc>(
                            m_devAcc,
                            extent,
                            alpaka::vec::Vec<Dim, Size>::ones(),
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
                auto const exec(
                    alpaka::exec::create<Acc>(
                        m_workDiv,
                        kernelFnObj,
                        args...));

                alpaka::stream::enqueue(m_stream, exec);

                alpaka::wait::wait(m_stream);

                return true;
            }
            //-----------------------------------------------------------------------------
            virtual ~KernelExecutionFixture()
            {}

        private:
            alpaka::dev::DevCpu m_devHost;
            DevAcc m_devAcc;
            StreamAcc m_stream;
            alpaka::workdiv::WorkDivMembers<Dim, Size> m_workDiv;
        };
    }
}
