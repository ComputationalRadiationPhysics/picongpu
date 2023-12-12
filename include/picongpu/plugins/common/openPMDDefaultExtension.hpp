/* Copyright 2022-2023 Sergei Bastrakov
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <string>

#include <openPMD/openPMD.hpp>

namespace picongpu
{
    namespace openPMD
    {
        enum class ExtensionPreference
        {
            ADIOS,
            HDF5
        };

        /** Get default extension for openPMD files
         *
         * Make a uniform choice when several valid backends are available.
         * Check that at least one valid backend is available.
         *
         * This function can only be compiled when openPMD is enabled.
         */
        inline std::string getDefaultExtension(ExtensionPreference ep = ExtensionPreference::ADIOS)
        {
            using EP = ExtensionPreference;
            auto getADIOSExtension = []()
            {
                auto availableExtensions = ::openPMD::getFileExtensions();
                /*
                 * Using macros for detecting the ADIOS2 version is not very nice since the openPMD-api can be
                 * recompiled against different backends and backend versions without affecting the ABI. Upgrading
                 * ADIOS2 will require recompiling openPMD-api, but only relinking PIConGPU. The below #ifdef logic
                 * will then not be updated.
                 *
                 * A functionality for doing a runtime query here is unfortunately missing in the openPMD-api at the
                 * moment.
                 * https://github.com/openPMD/openPMD-api/issues/1563
                 *
                 * The reason for requiring ADIOS2 >= v2.9.2 before enabling BP5 is this bug, fixed in version v2.9.2:
                 * https://github.com/ornladios/ADIOS2/issues/3504
                 */
#if openPMD_HAVE_ADIOS2
#    if ADIOS2_VERSION_MAJOR * 10000 + ADIOS2_VERSION_MINOR * 100 + ADIOS2_VERSION_PATCH >= 20902
                if(std::find(availableExtensions.begin(), availableExtensions.end(), "bp5")
                   != availableExtensions.end())
                {
                    // Engine available in ADIOS2 >= v2.8
                    // File extension and support for engine available in openPMD-api >= 0.15
                    return "bp5";
                }
                else
#    endif
#endif
                    if(std::find(availableExtensions.begin(), availableExtensions.end(), "bp4")
                       != availableExtensions.end())
                {
                    // Engine available and supported in all supported versions of ADIOS2 and openPMD-api
                    // File extension available in openPMD-api >= 0.15
                    return "bp4";
                }
                else
                {
                    // Extension is always available in all supported versions of ADIOS2 and openPMD-api
                    return "bp";
                }
            };
#if openPMD_HAVE_ADIOS2 && openPMD_HAVE_HDF5
            switch(ep)
            {
            case EP::ADIOS:
                return getADIOSExtension();
            case EP::HDF5:
                return "h5";
            }
            /*
             * This silences compiler warnings
             */
            return "[openPMD::getDefaultExtension()] Unreachable!";
#elif openPMD_HAVE_ADIOS2
            return getADIOSExtension();
#elif openPMD_HAVE_HDF5
            return "h5";
#else
            // Neither ADIOS2 nor HDF5 is not allowed when openPMD is enabled (we can only be here in this case)
            static_assert(
                false,
                "Error: openPMD API dependency is enabled but has neither ADIOS2 nor HDF5 backend available.");
#endif
        }
    } // namespace openPMD
} // namespace picongpu
