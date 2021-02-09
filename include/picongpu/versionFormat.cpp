/* Copyright 2015-2021 Axel Huebl, Franz Poeschel
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

#include <pmacc/boost_workaround.hpp>
#include "picongpu/versionFormat.hpp"

#include <boost/version.hpp>
// work-around: mallocMC PR #142
#include <boost/config.hpp>
#include <boost/preprocessor/stringize.hpp>

#ifdef __CUDACC_VER_MAJOR__
#    include <cuda.h>
#    include <mallocMC/mallocMC.hpp>
#endif
#include <mpi.h>
#if(ENABLE_HDF5 == 1)
#    include <splash/splash.h>
#endif
#if(ENABLE_ADIOS == 1)
#    include <adios.h>
#endif
#if(PIC_ENABLE_PNG == 1)
#    include <pngwriter.h>
#endif
#if(ENABLE_OPENPMD == 1)
#    include <openPMD/openPMD.hpp>
#endif

#include <sstream>


namespace picongpu
{
    std::list<std::string> getSoftwareVersions(std::ostream& cliText)
    {
        std::string const versionNotFound("NOTFOUND");

        std::stringstream picongpu;
        picongpu << PICONGPU_VERSION_MAJOR << "." << PICONGPU_VERSION_MINOR << "." << PICONGPU_VERSION_PATCH;
        if(std::string(PICONGPU_VERSION_LABEL).size() > 0)
            picongpu << "-" << PICONGPU_VERSION_LABEL;

        std::stringstream buildType;
        buildType << BOOST_PP_STRINGIZE(CMAKE_BUILD_TYPE);

        std::stringstream os;
        os << BOOST_PP_STRINGIZE(CMAKE_SYSTEM);
        std::stringstream arch;
        arch << BOOST_PP_STRINGIZE(CMAKE_SYSTEM_PROCESSOR);

        std::stringstream cxx;
        std::stringstream cxxVersion;
        cxx << BOOST_PP_STRINGIZE(CMAKE_CXX_COMPILER_ID);
        cxxVersion << BOOST_PP_STRINGIZE(CMAKE_CXX_COMPILER_VERSION);

        std::stringstream cmake;
        cmake << BOOST_PP_STRINGIZE(CMAKE_VERSION);

#ifdef __CUDACC_VER_MAJOR__
        std::stringstream cuda;
        cuda << __CUDACC_VER_MAJOR__ << "." << __CUDACC_VER_MINOR__ << "." << __CUDACC_VER_BUILD__;

        std::stringstream mallocMC;
        mallocMC << MALLOCMC_VERSION_MAJOR << "." << MALLOCMC_VERSION_MINOR << "." << MALLOCMC_VERSION_PATCH;
#endif

        std::stringstream boost;
        boost << int(BOOST_VERSION / 100000) << "." << int(BOOST_VERSION / 100 % 1000) << "."
              << int(BOOST_VERSION % 100);

        std::stringstream mpiStandard;
        std::stringstream mpiFlavor;
        std::stringstream mpiFlavorVersion;
        mpiStandard << MPI_VERSION << "." << MPI_SUBVERSION;
#if defined(OMPI_MAJOR_VERSION)
        // includes derivates such as Bullx MPI, Sun, ...
        mpiFlavor << "OpenMPI";
        mpiFlavorVersion << OMPI_MAJOR_VERSION << "." << OMPI_MINOR_VERSION << "." << OMPI_RELEASE_VERSION;
#elif defined(MPICH_VERSION)
        /* includes MPICH2 and MPICH3 and
         * derivates such as IBM, Cray, MS, Intel, MVAPICH(2), ... */
        mpiFlavor << "MPICH";
        mpiFlavorVersion << MPICH_VERSION;
#else
        mpiFlavor << "unknown";
        mpiFlavorVersion << "unknown";
#endif

        std::stringstream pngwriter;
#if(PIC_ENABLE_PNG == 1)
        pngwriter << PNGWRITER_VERSION_MAJOR << "." << PNGWRITER_VERSION_MINOR << "." << PNGWRITER_VERSION_PATCH;
#else
        pngwriter << versionNotFound;
#endif

        std::stringstream splash;
        std::stringstream splashFormat;
#if(ENABLE_HDF5 == 1)
        splash << SPLASH_VERSION_MAJOR << "." << SPLASH_VERSION_MINOR << "." << SPLASH_VERSION_PATCH;
        splashFormat << SPLASH_FILE_FORMAT_MAJOR << "." << SPLASH_FILE_FORMAT_MINOR;
#else
        splash << versionNotFound;
        splashFormat << versionNotFound;
#endif

        std::stringstream adios;
#if(ENABLE_ADIOS == 1)
        adios << ADIOS_VERSION;
#else
        adios << versionNotFound;
#endif

#if(ENABLE_OPENPMD == 1)
        std::string openPMD = openPMD::getVersion();
#else
        std::string openPMD = versionNotFound;
#endif

        // CLI Formatting
        cliText << "PIConGPU: " << picongpu.str() << std::endl;
        cliText << "  Build-Type: " << buildType.str() << std::endl << std::endl;
        cliText << "Third party:" << std::endl;
        cliText << "  OS:         " << os.str() << std::endl;
        cliText << "  arch:       " << arch.str() << std::endl;
        cliText << "  CXX:        " << cxx.str() << " (" << cxxVersion.str() << ")" << std::endl;
        cliText << "  CMake:      " << cmake.str() << std::endl;
#ifdef __CUDACC_VER_MAJOR__
        cliText << "  CUDA:       " << cuda.str() << std::endl;
        cliText << "  mallocMC:   " << mallocMC.str() << std::endl;
#endif
        cliText << "  Boost:      " << boost.str() << std::endl;
        cliText << "  MPI:        " << std::endl
                << "    standard: " << mpiStandard.str() << std::endl
                << "    flavor:   " << mpiFlavor.str() << " (" << mpiFlavorVersion.str() << ")" << std::endl;
        cliText << "  PNGwriter:  " << pngwriter.str() << std::endl;
        cliText << "  libSplash:  " << splash.str() << " (Format " << splashFormat.str() << ")" << std::endl;
        cliText << "  ADIOS:      " << adios.str() << std::endl;
        cliText << "  openPMD:    " << openPMD << std::endl;

        // Module-like formatting of software only
        std::list<std::string> software;
        software.push_back(std::string("PIConGPU/") + picongpu.str());
        software.push_back(cxx.str() + std::string("/") + cxxVersion.str());
        software.push_back(std::string("CMake/") + cmake.str());
#ifdef __CUDACC_VER_MAJOR__
        software.push_back(std::string("CUDA/") + cuda.str());
#endif
        software.push_back(std::string("Boost/") + boost.str());
        software.push_back(mpiFlavor.str() + std::string("/") + mpiFlavorVersion.str());
#ifdef __CUDACC_VER_MAJOR__
        software.push_back(std::string("mallocMC/") + mallocMC.str());
#endif
        if(pngwriter.str().compare(versionNotFound) != 0)
            software.push_back(std::string("PNGwriter/") + pngwriter.str());
        if(splash.str().compare(versionNotFound) != 0)
            software.push_back(std::string("libSplash/") + splash.str());
        if(adios.str().compare(versionNotFound) != 0)
            software.push_back(std::string("ADIOS/") + adios.str());
        if(openPMD.compare(versionNotFound) != 0)
            software.push_back(std::string("openPMD/") + openPMD);

        return software;
    }
} // namespace picongpu
