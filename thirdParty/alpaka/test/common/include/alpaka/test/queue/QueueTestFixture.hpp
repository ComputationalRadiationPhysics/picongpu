/**
 * \file
 * Copyright 2017 Benjamin Worpitz
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
