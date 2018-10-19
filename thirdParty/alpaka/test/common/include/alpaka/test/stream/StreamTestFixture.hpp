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
        namespace stream
        {
            //#############################################################################
            template<
                typename TDevStream>
            struct StreamTestFixture
            {
                using Dev = typename std::tuple_element<0, TDevStream>::type;
                using Stream = typename std::tuple_element<1, TDevStream>::type;

                using Pltf = alpaka::pltf::Pltf<Dev>;

                //-----------------------------------------------------------------------------
                StreamTestFixture() :
                    m_dev(alpaka::pltf::getDevByIdx<Pltf>(0u)),
                    m_stream(m_dev)
                {
                }

                Dev m_dev;
                Stream m_stream;
            };
        }
    }
}
