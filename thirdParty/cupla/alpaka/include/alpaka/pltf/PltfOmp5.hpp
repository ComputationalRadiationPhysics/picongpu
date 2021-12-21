/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_ANY_BT_OMP5_ENABLED

#    if _OPENMP < 201307
#        error If ALPAKA_ACC_ANY_BT_OMP5_ENABLED is set, the compiler has to support OpenMP 4.0 or higher!
#    endif

#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/dev/DevOmp5.hpp>
#    include <alpaka/pltf/Traits.hpp>

#    include <limits>
#    include <sstream>
#    include <vector>

namespace alpaka
{
    //#############################################################################
    //! The OpenMP 5 device platform.
    class PltfOmp5 : public concepts::Implements<ConceptPltf, PltfOmp5>
    {
    public:
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST PltfOmp5() = delete;
    };

    namespace traits
    {
        //#############################################################################
        //! The OpenMP 5 device device type trait specialization.
        template<>
        struct DevType<PltfOmp5>
        {
            using type = DevOmp5;
        };

        //#############################################################################
        //! The OpenMP 5 platform device count get trait specialization.
        template<>
        struct GetDevCount<PltfOmp5>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getDevCount() -> std::size_t
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                const std::size_t count = static_cast<std::size_t>(::omp_get_num_devices());
                // runtime will report zero devices if not target device is available or if offloading is disabled
                return count > 0 ? count : 1;
            }
        };

        //#############################################################################
        //! The OpenMP 5 platform device get trait specialization.
        template<>
        struct GetDevByIdx<PltfOmp5>
        {
            //-----------------------------------------------------------------------------
            //! \param devIdx device id, less than GetDevCount or equal, yielding omp_get_initial_device()
            ALPAKA_FN_HOST static auto getDevByIdx(std::size_t devIdx) -> DevOmp5
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                std::size_t const devCount(static_cast<std::size_t>(::omp_get_num_devices()));
                int devIdxOmp5 = static_cast<int>(devIdx);
                if(devIdx == devCount || (devCount == 0 && devIdx == 1 /* getDevCount */))
                { // take this case to use the initial device
                    devIdxOmp5 = ::omp_get_initial_device();
                }
                else if(devIdx > devCount)
                {
                    std::stringstream ssErr;
                    ssErr << "Unable to return device handle for device " << devIdx << ". There are only " << devCount
                          << " target devices"
                             "and the initial device with index "
                          << devCount;
                    throw std::runtime_error(ssErr.str());
                }

                return {devIdxOmp5};
            }
        };
    } // namespace traits
} // namespace alpaka

#endif
