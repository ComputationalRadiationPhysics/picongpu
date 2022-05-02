/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Concepts.hpp>
#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/pltf/Traits.hpp>

#include <sstream>
#include <vector>

namespace alpaka
{
    //! The CPU device platform.
    class PltfCpu : public concepts::Implements<ConceptPltf, PltfCpu>
    {
    public:
        ALPAKA_FN_HOST PltfCpu() = delete;
    };

    namespace trait
    {
        //! The CPU device device type trait specialization.
        template<>
        struct DevType<PltfCpu>
        {
            using type = DevCpu;
        };

        //! The CPU platform device count get trait specialization.
        template<>
        struct GetDevCount<PltfCpu>
        {
            ALPAKA_FN_HOST static auto getDevCount() -> std::size_t
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                return 1;
            }
        };

        //! The CPU platform device get trait specialization.
        template<>
        struct GetDevByIdx<PltfCpu>
        {
            ALPAKA_FN_HOST static auto getDevByIdx(std::size_t const& devIdx) -> DevCpu
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                std::size_t const devCount(getDevCount<PltfCpu>());
                if(devIdx >= devCount)
                {
                    std::stringstream ssErr;
                    ssErr << "Unable to return device handle for CPU device with index " << devIdx
                          << " because there are only " << devCount << " devices!";
                    throw std::runtime_error(ssErr.str());
                }

                return {};
            }
        };
    } // namespace trait
} // namespace alpaka
