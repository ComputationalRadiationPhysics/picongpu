/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_ANY_BT_OACC_ENABLED

#    if _OPENACC < 201306
#        error If ALPAKA_ACC_ANY_BT_OACC_ENABLED is set, the compiler has to support OpenACC 2.0 or higher!
#    endif

#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/dev/DevOacc.hpp>
#    include <alpaka/pltf/Traits.hpp>

#    include <sstream>
#    include <vector>

namespace alpaka
{
    //#############################################################################
    //! The OpenACC device platform.
    class PltfOacc : public concepts::Implements<ConceptPltf, PltfOacc>
    {
    public:
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST PltfOacc() = delete;
    };

    namespace traits
    {
        //#############################################################################
        //! The OpenACC device device type trait specialization.
        template<>
        struct DevType<PltfOacc>
        {
            using type = DevOacc;
        };

        //#############################################################################
        //! The OpenACC platform device count get trait specialization.
        template<>
        struct GetDevCount<PltfOacc>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getDevCount() -> std::size_t
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                return static_cast<std::size_t>(::acc_get_num_devices(::acc_get_device_type()));
            }
        };

        //#############################################################################
        //! The OpenACC platform device get trait specialization.
        template<>
        struct GetDevByIdx<PltfOacc>
        {
            //-----------------------------------------------------------------------------
            //! \param devIdx device id, less than GetDevCount
            ALPAKA_FN_HOST static auto getDevByIdx(std::size_t devIdx) -> DevOacc
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                std::size_t const devCount(getDevCount<PltfOacc>());
                if(devIdx >= devCount)
                {
                    std::stringstream ssErr;
                    ssErr << "Unable to return device handle for OpenACC device with index " << devIdx
                          << " because there are only " << devCount << " devices!";
                    throw std::runtime_error(ssErr.str());
                }

                return {static_cast<int>(devIdx)};
            }
        };
    } // namespace traits
} // namespace alpaka

#endif
