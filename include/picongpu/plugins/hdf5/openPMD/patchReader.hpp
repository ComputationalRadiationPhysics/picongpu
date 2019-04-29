/* Copyright 2016-2019 Axel Huebl
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

#include "picongpu/plugins/common/particlePatches.hpp"

#if( ENABLE_HDF5 == 1 )
#  include <splash/splash.h>
#endif

#include <vector>
#include <list>
#include <string>
#include <iostream>
#include <typeinfo>

namespace picongpu
{
namespace hdf5
{
namespace openPMD
{
    class PatchReader;

#if( ENABLE_HDF5 == 1 )
    /** Functor to populate and validate the list of particle patches
     */
    class PatchReader
    {
    private:
        /** Determine the variable type for `offset` and `extent`
         *
         * In particle patches, the `offset` and `extent` can be of
         * user-defined types. This function allows to determine which
         * one was used and how many patches exist.
         *
         * @note currently we force the type to be `uint64_t`,
         *       we can implement type conversions later on
         * @note currently we force the number of patches
         *       to stay constant during restarts
         *
         * @param dc parallel libSplash DataCollector
         * @param availableRanks MPI ranks in the restarted simulation
         *        that are currently waiting to find patches
         * @param id iteration in file
         * @param particlePatchPathComponent string such as
         *             "particles/e/particlePatches/numParticles" or
         *             "particles/e/particlePatches/offset/x"
         */
        void checkSpatialTypeSize(
            splash::DataCollector* const dc,
            const uint32_t availableRanks,
            const int32_t id,
            const std::string particlePatchPathComponent
        ) const;

        /** Read a specific record component of the particle patch
         *
         * Read for example: numParticles or offset/x
         *
         * @param[in]  dc pointer to an open splash::DataCollector
         * @param[in]  availableRanks MPI ranks in the restarted simulation
         *             that are currently waiting to find patches
         * @param[in]  id time step to read
         * @param[in]  particlePatchPathComponent string such as
         *             "particles/e/particlePatches/numParticles" or
         *             "particles/e/particlePatches/offset/x"
         * @param[out] dest beginning of c-array of length size()
         *             to write the patch record component to
         */
        void readPatchAttribute(
            splash::DataCollector* const dc,
            const uint32_t availableRanks,
            const int32_t id,
            const std::string particlePatchPathComponent,
            uint64_t* const dest
        ) const;

    public:
        /** Build up the global list of patches
         *
         * @param dc parallel libSplash DataCollector
         * @param availableRanks MPI ranks in the restarted simulation
         *        that are currently waiting to find patches
         * @param dimensionality the PIConGPU simDim
         * @param id iteration in file
         * @param particlePatchPath in-file path to a specific particle patch dir
         *
         * @return picongpu::openPMD::ParticlePatches struct of arrays with patches
         */
        picongpu::openPMD::ParticlePatches operator()(
            splash::DataCollector* const dc,
            const uint32_t availableRanks,
            const uint32_t dimensionality,
            const int32_t id,
            const std::string particlePatchPath
        ) const;
    };
#endif

} // namespace openPMD
} // namespace hdf5
} // namespace picongpu
