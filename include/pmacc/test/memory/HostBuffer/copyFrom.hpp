/*
 * SPDX-FileCopyrightText: Erik Zenker
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

/* #includes in "test/memoryUT.cu" */

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/memory/buffers/HostBuffer.hpp"

namespace pmacc
{
    namespace test
    {
        namespace memory
        {
            namespace HostBuffer
            {
                /**
                 * Checks if data is copied correctly from device to
                 * host.
                 */
                struct CopyFromTest
                {
                    template<typename T_Dim>
                    void exec(T_Dim)
                    {
                        using Data = uint8_t;
                        using Extents = size_t;

                        using ::pmacc::test::memory::getElementsPerDim;

                        std::vector<size_t> nElementsPerDim = getElementsPerDim<T_Dim>();

                        for(unsigned i = 0; i < nElementsPerDim.size(); ++i)
                        {
                            auto const dataSpace = ::pmacc::DataSpace<T_Dim::value>::create(nElementsPerDim[i]);
                            auto* hostBuffer = new ::pmacc::HostBuffer<Data, T_Dim::value>(dataSpace);
                            auto* deviceBuffer = new ::pmacc::DeviceBuffer<Data, T_Dim::value>(dataSpace);

                            hostBuffer->reset();

                            auto fillBox = hostBuffer->getDataBox();
                            for(size_t i = 0; i < static_cast<size_t>(dataSpace.productOfComponents()); ++i)
                            {
                                fillBox.getPointer()[i] = static_cast<Data>(i);
                            }

                            deviceBuffer->copyFrom(*hostBuffer);
                            hostBuffer->reset();
                            hostBuffer->copyFrom(*deviceBuffer);

                            auto compareBox = hostBuffer->getDataBox();
                            for(size_t i = 0; i < static_cast<size_t>(dataSpace.productOfComponents()); ++i)
                            {
                                REQUIRE(compareBox.getPointer()[i] == static_cast<Data>(i));
                            }

                            delete hostBuffer;
                            delete deviceBuffer;
                        }
                    }

                    PMACC_NO_NVCC_HDWARNING
                    template<typename T_Dim>
                    HDINLINE void operator()(T_Dim dim)
                    {
                        exec(dim);
                    }
                };

            } // namespace HostBuffer
        } // namespace memory
    } // namespace test
} // namespace pmacc

TEST_CASE("HostBuffer::copyFrom", "[copyFrom]")
{
    using namespace pmacc::test::memory::HostBuffer;
    ::pmacc::mp_for_each<Dims>(CopyFromTest());
}
