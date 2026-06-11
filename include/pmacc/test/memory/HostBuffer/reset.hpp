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
                 * Checks if the HostBuffer is reseted correctly to zero.
                 */
                struct ResetTest
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
                            ::pmacc::HostBuffer<Data, T_Dim::value> hostBuffer(dataSpace);

                            hostBuffer.reset();

                            auto resultBox = hostBuffer.getDataBox();
                            for(size_t i = 0; i < static_cast<size_t>(dataSpace.productOfComponents()); ++i)
                            {
                                REQUIRE(resultBox.getPointer()[i] == 0);
                            }
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

TEST_CASE("HostBuffer::reset", "[reset]")
{
    using namespace pmacc::test::memory::HostBuffer;
    ::pmacc::mp_for_each<Dims>(ResetTest());
}
