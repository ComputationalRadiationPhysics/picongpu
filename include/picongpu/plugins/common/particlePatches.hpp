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

#include <vector>
#include <list>
#include <iostream>
#include <cstdint>

namespace picongpu
{
namespace openPMD
{

    /** Struct for a list of particle patches
     *
     * Object for all particle patches.
     * @see https://github.com/openPMD/openPMD-standard/blob/1.0.0/STANDARD.md#sub-group-for-each-particle-species
     */
    class ParticlePatches
    {
    private:
        /** Disallow (empty) default contructor
         */
        ParticlePatches ();

    public:
        std::vector<uint64_t> numParticles;
        std::vector<uint64_t> numParticlesOffset;

        std::vector<uint64_t> offsetX;
        std::vector<uint64_t> offsetY;
        std::vector<uint64_t> offsetZ;

        std::vector<uint64_t> extentX;
        std::vector<uint64_t> extentY;
        std::vector<uint64_t> extentZ;

        /** Fill-Constructor with n empty-sized patches
         *
         * @param n number of patches to store
         */
        ParticlePatches( const size_t n );

        /** Return the beginning of one of the components of the
         *  offset as pointer
         *
         * Be aware that the pointer is pointing to the beginning
         * of a C-array of size `size()` and is only allocated as long
         * as the `ParticlePatches` object is alive.
         *
         * @param comp component (0=x, 1=y, 2=z) of offset array
         *             for the list of patches
         * @return uint64_t* pointing to the beginning of a c-array
         *                   with length as given in size()
         */
        uint64_t* getOffsetComp( const uint32_t comp );

        /** Return the beginning of one of the components of the
         *  extent as pointer
         *
         * Be aware that the pointer is pointing to the beginning
         * of a C-array of size `size()` and is only allocated as long
         * as the `ParticlePatches` object is alive.
         *
         * @param comp component (0=x, 1=y, 2=z) of extent array
         *             for the list of patches
         * @return uint64_t* pointing to the beginning of a c-array
         *                   with length as given in size()
         */
        uint64_t* getExtentComp( const uint32_t comp );

        /** Returns the number of patches
         */
        size_t size() const;

        /** Helper function printing to std::cout
         */
        void print();
    };

} // namespace openPMD
} // namespace picongpu
