/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
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

#pragma once

#include <alpaka/pltf/Traits.hpp>
#include <alpaka/dev/DevCpu.hpp>

#include <sstream>
#include <vector>

namespace alpaka
{
    namespace pltf
    {
        //#############################################################################
        //! The CPU device platform.
        class PltfCpu
        {
        public:
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST PltfCpu() = delete;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU device device type trait specialization.
            template<>
            struct DevType<
                pltf::PltfCpu>
            {
                using type = dev::DevCpu;
            };
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU platform device count get trait specialization.
            template<>
            struct GetDevCount<
                pltf::PltfCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDevCount()
                -> std::size_t
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    return 1;
                }
            };

            //#############################################################################
            //! The CPU platform device get trait specialization.
            template<>
            struct GetDevByIdx<
                pltf::PltfCpu>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDevByIdx(
                    std::size_t const & devIdx)
                -> dev::DevCpu
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    std::size_t const devCount(pltf::getDevCount<pltf::PltfCpu>());
                    if(devIdx >= devCount)
                    {
                        std::stringstream ssErr;
                        ssErr << "Unable to return device handle for CPU device with index " << devIdx << " because there are only " << devCount << " devices!";
                        throw std::runtime_error(ssErr.str());
                    }

                    return {};
                }
            };
        }
    }
}
