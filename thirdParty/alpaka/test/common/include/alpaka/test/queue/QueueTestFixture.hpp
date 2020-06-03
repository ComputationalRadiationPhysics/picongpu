/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/alpaka.hpp>

namespace alpaka
{
    namespace test
    {
        namespace queue
        {
            //#############################################################################
            template<
                typename TDevQueue>
            struct QueueTestFixture
            {
                using Dev = typename std::tuple_element<0, TDevQueue>::type;
                using Queue = typename std::tuple_element<1, TDevQueue>::type;

                using Pltf = alpaka::pltf::Pltf<Dev>;

                //-----------------------------------------------------------------------------
                QueueTestFixture() :
                    m_dev(alpaka::pltf::getDevByIdx<Pltf>(0u)),
                    m_queue(m_dev)
                {
                }

                Dev m_dev;
                Queue m_queue;
            };
        }
    }
}
