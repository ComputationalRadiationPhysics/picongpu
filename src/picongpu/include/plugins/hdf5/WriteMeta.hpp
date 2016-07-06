/**
 * Copyright 2013-2016 Axel Huebl
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

#include "simulation_defines.hpp"

#include "plugins/hdf5/HDF5Writer.def"
#include "Environment.hpp"

#include "fields/FieldManipulator.hpp"
#include "fields/currentInterpolation/CurrentInterpolation.hpp"

#include "traits/SIBaseUnits.hpp"
#include "traits/SplashToPIC.hpp"
#include "traits/PICToSplash.hpp"

#include <string>
#include <sstream>
#include <list>

namespace picongpu
{

namespace hdf5
{
using namespace PMacc;

    struct WriteMeta
    {
        typedef PICToSplash<float_X>::type SplashFloatXType;

        void operator()(ThreadParams *threadParams)
        {
            ColTypeUInt32 ctUInt32;
            ColTypeUInt64 ctUInt64;
            ColTypeDouble ctDouble;
            SplashFloatXType splashFloatXType;

            ParallelDomainCollector *dc = threadParams->dataCollector;
            uint32_t currentStep = threadParams->currentStep;

            /* openPMD attributes */
            /*   required */
            const std::string openPMDversion( "1.0.0" );
            ColTypeString ctOpenPMDversion( openPMDversion.length() );
            dc->writeGlobalAttribute( threadParams->currentStep,
                                      ctOpenPMDversion, "openPMD",
                                      openPMDversion.c_str() );

            const uint32_t openPMDextension = 1; // ED-PIC ID
            dc->writeGlobalAttribute( threadParams->currentStep,
                                      ctUInt32, "openPMDextension",
                                      &openPMDextension );

            const std::string basePath( "/data/%T/" );
            ColTypeString ctBasePath( basePath.length() );
            dc->writeGlobalAttribute( threadParams->currentStep,
                                      ctBasePath, "basePath",
                                      basePath.c_str() );

            const std::string meshesPath( "fields/" );
            ColTypeString ctMeshesPath( meshesPath.length() );
            dc->writeGlobalAttribute( threadParams->currentStep,
                                      ctMeshesPath, "meshesPath",
                                      meshesPath.c_str() );

            const std::string particlesPath( "particles/" );
            ColTypeString ctParticlesPath( particlesPath.length() );
            dc->writeGlobalAttribute( threadParams->currentStep,
                                      ctParticlesPath, "particlesPath",
                                      particlesPath.c_str() );

            const std::string iterationEncoding( "fileBased" );
            ColTypeString ctIterationEncoding( iterationEncoding.length() );
            dc->writeGlobalAttribute( threadParams->currentStep,
                                      ctIterationEncoding, "iterationEncoding",
                                      iterationEncoding.c_str() );

            const std::string iterationFormat( threadParams->h5Filename + std::string("_%T.h5") );
            ColTypeString ctIterationFormat( iterationFormat.length() );
            dc->writeGlobalAttribute( threadParams->currentStep,
                                      ctIterationFormat, "iterationFormat",
                                      iterationFormat.c_str() );

            /*   recommended */
            const std::string author = Environment<>::get().SimulationDescription().getAuthor();
            if( author.length() > 0 )
            {
                ColTypeString ctAuthor( author.length() );
                dc->writeGlobalAttribute( threadParams->currentStep,
                                          ctAuthor, "author",
                                          author.c_str() );
            }
            const std::string software( "PIConGPU" );
            ColTypeString ctSoftware( software.length() );
            dc->writeGlobalAttribute( threadParams->currentStep,
                                      ctSoftware, "software",
                                      software.c_str() );

            std::stringstream softwareVersion;
            softwareVersion << PICONGPU_VERSION_MAJOR << "."
                            << PICONGPU_VERSION_MINOR << "."
                            << PICONGPU_VERSION_PATCH;
            ColTypeString ctSoftwareVersion( softwareVersion.str().length() );
            dc->writeGlobalAttribute( threadParams->currentStep,
                                      ctSoftwareVersion, "softwareVersion",
                                      softwareVersion.str().c_str() );

            const std::string date = helper::getDateString( "%F %T %z" );
            ColTypeString ctDate( date.length() );
            dc->writeGlobalAttribute( threadParams->currentStep,
                                      ctDate, "date",
                                      date.c_str() );
            /*   ED-PIC */
            GetStringProperties<fieldSolver::FieldSolver> fieldSolverProps;
            const std::string fieldSolver( fieldSolverProps["name"].value );
            ColTypeString ctFieldSolver( fieldSolver.length() );
            dc->writeAttribute(currentStep, ctFieldSolver, meshesPath.c_str(),
                "fieldSolver", fieldSolver.c_str());

            /* order as in axisLabels:
             *    3D: z-lower, z-upper, y-lower, y-upper, x-lower, x-upper
             *    2D: y-lower, y-upper, x-lower, x-upper
             */
            GetStringProperties<FieldManipulator> fieldBoundaryProp;
            std::list<std::string> listFieldBoundary;
            std::list<std::string> listFieldBoundaryParam;
            for( uint32_t i = NumberOfExchanges<simDim>::value - 1; i > 0; --i )
            {
                if( FRONT % i == 0 )
                {
                    listFieldBoundary.push_back(
                        fieldBoundaryProp[ExchangeTypeNames()[i]]["name"].value
                    );
                    listFieldBoundaryParam.push_back(
                        fieldBoundaryProp[ExchangeTypeNames()[i]]["param"].value
                    );
                }
            }
            helper::GetSplashArrayOfString getSplashArrayOfString;
            PMACC_AUTO(arrFieldBoundary, getSplashArrayOfString( listFieldBoundary ));
            ColTypeString ctFieldBoundaries( arrFieldBoundary.maxLen );
            PMACC_AUTO(arrFieldBoundaryParam, getSplashArrayOfString( listFieldBoundaryParam ));
            ColTypeString ctFieldBoundariesParam( arrFieldBoundaryParam.maxLen );

            dc->writeAttribute( currentStep, ctFieldBoundaries, meshesPath.c_str(),
                "fieldBoundary",
                1u, Dimensions( listFieldBoundary.size(), 0, 0 ),
                &( arrFieldBoundary.buffers.at( 0 ) )
            );
            dc->writeAttribute( currentStep, ctFieldBoundariesParam, meshesPath.c_str(),
                "fieldBoundaryParameters",
                1u, Dimensions( listFieldBoundaryParam.size(), 0, 0 ),
                &( arrFieldBoundaryParam.buffers.at( 0 ) )
            );

            if( bmpl::size<VectorAllSpecies>::type::value > 0 )
            {
                // assume all boundaries are like the first species for openPMD 1.0.0
                GetStringProperties<bmpl::at_c<VectorAllSpecies, 0>::type> particleBoundaryProp;
                std::list<std::string> listParticleBoundary;
                std::list<std::string> listParticleBoundaryParam;
                for( uint32_t i = NumberOfExchanges<simDim>::value - 1; i > 0; --i )
                {
                    if( FRONT % i == 0 )
                    {
                        listParticleBoundary.push_back(
                            particleBoundaryProp[ExchangeTypeNames()[i]]["name"].value
                        );
                        listParticleBoundaryParam.push_back(
                            particleBoundaryProp[ExchangeTypeNames()[i]]["param"].value
                        );
                    }
                }
                PMACC_AUTO(arrParticleBoundary, getSplashArrayOfString( listParticleBoundary ));
                ColTypeString ctParticleBoundary( arrParticleBoundary.maxLen );
                PMACC_AUTO(arrParticleBoundaryParam, getSplashArrayOfString( listParticleBoundaryParam ));
                ColTypeString ctParticleBoundaryParam( arrParticleBoundaryParam.maxLen );

                dc->writeAttribute( currentStep, ctParticleBoundary, meshesPath.c_str(),
                    "particleBoundary",
                    1u, Dimensions( listParticleBoundary.size(), 0, 0 ),
                    &( arrParticleBoundary.buffers.at( 0 ) )
                );
                dc->writeAttribute( currentStep, ctParticleBoundaryParam, meshesPath.c_str(),
                    "particleBoundaryParameters",
                    1u, Dimensions( listParticleBoundaryParam.size(), 0, 0 ),
                    &( arrParticleBoundaryParam.buffers.at( 0 ) )
                );
            }

            GetStringProperties<fieldSolver::CurrentInterpolation> currentSmoothingProp;
            const std::string currentSmoothing( currentSmoothingProp["name"].value );
            ColTypeString ctCurrentSmoothing( currentSmoothing.length() );
            dc->writeAttribute( currentStep, ctCurrentSmoothing, meshesPath.c_str(),
                "currentSmoothing", currentSmoothing.c_str() );

            if( currentSmoothingProp.find( "param" ) != currentSmoothingProp.end() )
            {
                const std::string currentSmoothingParam( currentSmoothingProp["param"].value );
                ColTypeString ctCurrentSmoothingParam( currentSmoothingParam.length() );
                dc->writeAttribute( currentStep, ctCurrentSmoothingParam, meshesPath.c_str(),
                    "currentSmoothingParameters", currentSmoothingParam.c_str() );
            }

            const std::string chargeCorrection( "none" );
            ColTypeString ctChargeCorrection( chargeCorrection.length() );
            dc->writeAttribute( currentStep, ctChargeCorrection, meshesPath.c_str(),
                "chargeCorrection", chargeCorrection.c_str() );

            /* write number of slides */
            const uint32_t slides = MovingWindow::getInstance().getSlideCounter(
                threadParams->currentStep
            );

            dc->writeAttribute( threadParams->currentStep,
                                ctUInt32, NULL, "sim_slides", &slides );


            /* openPMD: required time attributes */
            dc->writeAttribute( currentStep, splashFloatXType, NULL, "dt", &DELTA_T );
            const float_X time = float_X( threadParams->currentStep ) * DELTA_T;
            dc->writeAttribute( currentStep, splashFloatXType, NULL, "time", &time );
            dc->writeAttribute( currentStep, ctDouble, NULL, "timeUnitSI", &UNIT_TIME );

            /* write normed grid parameters */
            dc->writeAttribute( currentStep, splashFloatXType, NULL, "cell_width", &CELL_WIDTH );
            dc->writeAttribute( currentStep, splashFloatXType, NULL, "cell_height", &CELL_HEIGHT );
            if( simDim == DIM3 )
            {
                dc->writeAttribute( currentStep, splashFloatXType, NULL, "cell_depth", &CELL_DEPTH );
            }

            /* write base units */
            dc->writeAttribute( currentStep, ctDouble, NULL, "unit_energy", &UNIT_ENERGY );
            dc->writeAttribute( currentStep, ctDouble, NULL, "unit_length", &UNIT_LENGTH );
            dc->writeAttribute( currentStep, ctDouble, NULL, "unit_speed", &UNIT_SPEED );
            dc->writeAttribute( currentStep, ctDouble, NULL, "unit_time", &UNIT_TIME );
            dc->writeAttribute( currentStep, ctDouble, NULL, "unit_mass", &UNIT_MASS );
            dc->writeAttribute( currentStep, ctDouble, NULL, "unit_charge", &UNIT_CHARGE );
            dc->writeAttribute( currentStep, ctDouble, NULL, "unit_efield", &UNIT_EFIELD );
            dc->writeAttribute( currentStep, ctDouble, NULL, "unit_bfield", &UNIT_BFIELD );

            /* write physical constants */
            dc->writeAttribute( currentStep, splashFloatXType, NULL, "mue0", &MUE0 );
            dc->writeAttribute( currentStep, splashFloatXType, NULL, "eps0", &EPS0 );
        }
    };
} // namespace hdf5
} // namespace picongpu
