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

#include "picongpu/plugins/common/particlePatches.hpp"


namespace picongpu
{
namespace openPMD
{

    ParticlePatches::ParticlePatches( const size_t n )
    {
        /* zero particles */
        numParticles = std::vector<uint64_t>( n, 0u );
        numParticlesOffset = std::vector<uint64_t>( n, 0u );

        /* zero offsets */
        offsetX = std::vector<uint64_t>( n, 0u );
        offsetY = std::vector<uint64_t>( n, 0u );
        offsetZ = std::vector<uint64_t>( n, 0u );

        /* zero extents */
        extentX = std::vector<uint64_t>( n, 0u );
        extentY = std::vector<uint64_t>( n, 0u );
        extentZ = std::vector<uint64_t>( n, 0u );
    }

    uint64_t* ParticlePatches::getOffsetComp( const uint32_t comp )
    {
        if( comp == 0 )
            return &(*offsetX.begin());
        if( comp == 1 )
            return &(*offsetY.begin());
        if( comp == 2 )
            return &(*offsetZ.begin());

        return nullptr;
    }

    uint64_t* ParticlePatches::getExtentComp( const uint32_t comp )
    {
        if( comp == 0 )
            return &(*extentX.begin());
        if( comp == 1 )
            return &(*extentY.begin());
        if( comp == 2 )
            return &(*extentZ.begin());

        return nullptr;
    }

    size_t ParticlePatches::size() const
    {
        return numParticles.size();
    }

    void ParticlePatches::print()
    {
        std::cout << "id | numParticles numParticlesOffset "
                  << "offsetX offsetY offsetZ extentX extentY extentZ"
                  << std::endl;
        for( size_t i = 0; i < this->size(); ++i )
        {
            std::cout << i << " | "
                      << numParticles.at(i) << " "
                      << numParticlesOffset.at(i) << " "
                      << offsetX.at(i) << " "
                      << offsetY.at(i) << " "
                      << offsetZ.at(i) << " "
                      << extentX.at(i) << " "
                      << extentY.at(i) << " "
                      << extentZ.at(i) << std::endl;
        }
    }

} // namespace openPMD
} // namespace picongpu
