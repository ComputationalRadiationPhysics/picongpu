/* Copyright 2024 Benjamin Worpitz, Andrea Bocci, Bernhard Manfred Gruber, Jan Stephan, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/alpaka.hpp"

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
#    error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
#    error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

#include "alpaka/test/Check.hpp"
#include "alpaka/test/queue/Queue.hpp"

#include <utility>

namespace alpaka::test
{
    //! The fixture for executing a kernel on a given accelerator.
    template<typename TAcc>
    class KernelExecutionFixture
    {
    public:
        using Acc = TAcc;
        using Dim = alpaka::Dim<Acc>;
        using Idx = alpaka::Idx<Acc>;
        using Platform = alpaka::Platform<Acc>;
        using Device = Dev<Acc>;
        using Queue = test::DefaultQueue<Device>;
        using WorkDiv = WorkDivMembers<Dim, Idx>;

        KernelExecutionFixture(WorkDiv workDiv) : m_queue{m_device}, m_workDiv{std::move(workDiv)}
        {
        }

        template<typename TExtent>
        KernelExecutionFixture(TExtent const& extent)
            : m_queue{m_device}
            , m_workDiv{getValidWorkDiv<Acc>(
                  m_device,
                  extent,
                  Vec<Dim, Idx>::ones(),
                  false,
                  GridBlockExtentSubDivRestrictions::Unrestricted)}
        {
        }

        KernelExecutionFixture(Queue queue, WorkDiv workDiv) : m_queue{std::move(queue)}, m_workDiv{std::move(workDiv)}
        {
        }

        template<typename TExtent>
        KernelExecutionFixture(Queue queue, TExtent const& extent)
            : m_queue{std::move(queue)}
            , m_workDiv{getValidWorkDiv<Acc>(
                  m_device,
                  extent,
                  Vec<Dim, Idx>::ones(),
                  false,
                  GridBlockExtentSubDivRestrictions::Unrestricted)}
        {
        }

        template<typename TKernelFnObj, typename... TArgs>
        auto operator()(TKernelFnObj const& kernelFnObj, TArgs&&... args) -> bool
        {
            // Allocate the result value
            auto bufAccResult = allocBuf<bool, Idx>(m_device, static_cast<Idx>(1u));
            memset(m_queue, bufAccResult, static_cast<std::uint8_t>(true));

            exec<Acc>(m_queue, m_workDiv, kernelFnObj, getPtrNative(bufAccResult), std::forward<TArgs>(args)...);

            // Copy the result value to the host
            auto bufHostResult = allocBuf<bool, Idx>(m_devHost, static_cast<Idx>(1u));
            memcpy(m_queue, bufHostResult, bufAccResult);
            wait(m_queue);

            auto const result = *getPtrNative(bufHostResult);

            return result;
        }

    private:
        PlatformCpu m_platformHost{};
        DevCpu m_devHost{getDevByIdx(m_platformHost, 0)};
        Platform m_platform{};
        Device m_device{getDevByIdx(m_platform, 0)};
        Queue m_queue;
        WorkDiv m_workDiv;
    };

} // namespace alpaka::test
