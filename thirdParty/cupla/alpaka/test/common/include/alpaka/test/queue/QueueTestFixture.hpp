/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/alpaka.hpp>

#include <tuple>

namespace alpaka
{
    namespace test
    {
        //#############################################################################
        template<typename TDevQueue>
        struct QueueTestFixture
        {
            using Dev = std::tuple_element_t<0, TDevQueue>;
            using Queue = std::tuple_element_t<1, TDevQueue>;

            using Pltf = alpaka::Pltf<Dev>;

            //-----------------------------------------------------------------------------
            QueueTestFixture() : m_dev(alpaka::getDevByIdx<Pltf>(0u)), m_queue(m_dev)
            {
            }

            Dev m_dev;
            Queue m_queue;
        };
    } // namespace test
} // namespace alpaka
