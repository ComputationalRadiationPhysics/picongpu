/* Copyright 2013-2020 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch,
 *                     Klaus Steiniger, Felix Schmitt, Benjamin Worpitz,
 *                     Juncheng E, Sergei Bastrakov
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

#include "picongpu/simulation_defines.hpp"

#include "picongpu/plugins/xrayDiffraction/ComputeGlobalDomain.hpp"
#include "picongpu/plugins/xrayDiffraction/ReciprocalSpace.hpp"

#include <pmacc/Environment.hpp>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>


namespace picongpu
{
namespace plugins
{
namespace xrayDiffraction
{
namespace detail
{

    /** Writer of X-ray diffraction results to files
     *
     * Intended to be instantiated once and used from a single rank in a
     * simulation
     */
    class Writer
    {
    public:

        /** Create a writer
         *
         * @param filePrefix output file prefix
         */
        Writer( std::string const & filePrefix );

        /** Write diffraction intensity to a file
         *
         * @param globalDomainResult result aggregated for the global domain
         * @param currentStep current time iteration
         */
        void write(
            ComputeGlobalDomain const & globalDomainResult,
            uint32_t currentStep
        ) const;

    private:

        //! Directory name inside the general output directory
        std::string const directoryName;

        //! Output file prefix
        std::string const filePrefix;

    };

    Writer::Writer( std::string const & filePrefix ):
        filePrefix( filePrefix ),
        directoryName( "xrayDiffraction" )
    {
        auto & fs = Environment< simDim >::get().Filesystem();
        fs.createDirectoryWithPermissions( directoryName );
    }

    void Writer::write(
        ComputeGlobalDomain const & globalDomainResult,
        uint32_t const currentStep
    ) const
    {
        auto const fileName = directoryName + "/" + filePrefix + "_"
            + "intensity" +  std::to_string( currentStep ) + ".dat";
        auto ofile = std::ofstream{ fileName.c_str() };
        if( !ofile )
        {
            std::cerr << "Could not open file [" << fileName
            << "] for output\n";
            return;
        }
        /// TODO: clarify units, perhaps a conversion is needed
        ofile << "#\t" << "qx\t" << "qy\t" << "qz\t" << "intensity\n";
        for( size_t i = 0; i < globalDomainResult.diffractionIntensity.size();
            i++ )
        {
            auto const q = globalDomainResult.reciprocalSpace.getValue( i );
            ofile << q[ 0 ] << "\t" << q[ 1 ] << "\t" << q[ 2 ] << "\t"
                  << globalDomainResult.diffractionIntensity[ i ] << "\n";
        }
    }

} // namespace detail
} // namespace xrayDiffraction
} // namespace plugins
} // namespace picongpu
