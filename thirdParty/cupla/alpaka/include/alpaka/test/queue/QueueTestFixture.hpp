/* Copyright 2019 Benjamin Worpitz
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once
#include "alpaka/alpaka.hpp"

#include <tuple>

namespace alpaka::test
{
    template<typename TDevQueue>
    struct QueueTestFixture
    {
        using Dev = std::tuple_element_t<0, TDevQueue>;
        using Queue = std::tuple_element_t<1, TDevQueue>;
        using Platform = alpaka::Platform<Dev>;

        QueueTestFixture() : m_dev(getDevByIdx(m_platform, 0)), m_queue(m_dev)
        {
        }

        Platform m_platform;
        Dev m_dev;
        Queue m_queue;
    };
} // namespace alpaka::test
