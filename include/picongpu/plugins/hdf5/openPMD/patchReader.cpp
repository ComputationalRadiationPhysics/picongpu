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

#if( ENABLE_HDF5 == 1 )

#  include "picongpu/plugins/hdf5/openPMD/patchReader.hpp"


namespace picongpu
{
namespace hdf5
{
namespace openPMD
{
    void PatchReader::checkSpatialTypeSize(
            splash::DataCollector* const dc,
            const uint32_t availableRanks,
            const int32_t id,
            const std::string particlePatchPathComponent
    ) const
    {
        // will later read into 1D buffer from first position on
        splash::Dimensions dstBuffer(availableRanks, 1, 1);
        splash::Dimensions dstOffset(0, 0, 0);
        // sizeRead will be set
        splash::Dimensions sizeRead(0, 0, 0);

        splash::CollectionType* colType = dc->readMeta(
            id,
            particlePatchPathComponent.c_str(),
            dstBuffer,
            dstOffset,
            sizeRead );

        // check if the 1D list of patches has the right length
        assert( sizeRead[0] == availableRanks );

        // currently only support uint64_t types to spare type conversation
        assert( typeid(*colType) == typeid(splash::ColTypeUInt64) );

        // free collections
        delete( colType );
        colType = nullptr;
    }

    void PatchReader::readPatchAttribute(
        splash::DataCollector* const dc,
        const uint32_t availableRanks,
        const int32_t id,
        const std::string particlePatchPathComponent,
        uint64_t* const dest
    ) const
    {
        // will later read into 1D buffer from first position on
        splash::Dimensions dstBuffer(availableRanks, 1, 1);
        splash::Dimensions dstOffset(0, 0, 0);
        // sizeRead will be set
        splash::Dimensions sizeRead(0, 0, 0);

        // check if types, number of patches and names are supported
        checkSpatialTypeSize( dc, availableRanks, id, particlePatchPathComponent.c_str() );

        // read actual offset and extent data of particle patch component
        dc->read( id,
                  particlePatchPathComponent.c_str(),
                  sizeRead,
                  (void*)dest );
    }

    picongpu::openPMD::ParticlePatches PatchReader::operator()(
        splash::DataCollector* const dc,
        const uint32_t availableRanks,
        const uint32_t dimensionality,
        const int32_t id,
        const std::string particlePatchPath
    ) const
    {
        // allocate memory for patches
        picongpu::openPMD::ParticlePatches particlePatches( availableRanks );
        const std::string name_lookup[] = {"x", "y", "z"};
        for( uint32_t d = 0; d < dimensionality; ++d )
        {
            readPatchAttribute(
                dc, availableRanks, id,
                particlePatchPath + std::string("offset/") + name_lookup[d],
                particlePatches.getOffsetComp( d )
            );
            readPatchAttribute(
                dc, availableRanks, id,
                particlePatchPath + std::string("extent/") + name_lookup[d],
                particlePatches.getExtentComp( d )
            );
        }

        // read number of particles and their starting point (offset), too
        readPatchAttribute(
            dc, availableRanks, id,
            particlePatchPath + std::string("numParticles"),
            &(*particlePatches.numParticles.begin())
        );
        readPatchAttribute(
            dc, availableRanks, id,
            particlePatchPath + std::string("numParticlesOffset"),
            &(*particlePatches.numParticlesOffset.begin())
        );

        // return struct of array with particle patches
        return particlePatches;
    }

} // namespace openPMD
} // namespace hdf5
} // namespace picongpu

#endif
