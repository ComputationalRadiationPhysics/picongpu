/* Copyright 2013-2021 Axel Huebl
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

#include "picongpu/plugins/adios/ADIOSWriter.def"
#include "picongpu/plugins/common/stringHelpers.hpp"
#include <pmacc/Environment.hpp>

#include "picongpu/fields/absorber/Absorber.hpp"
#include "picongpu/fields/currentInterpolation/CurrentInterpolation.hpp"

#include "picongpu/traits/SIBaseUnits.hpp"
#include "picongpu/traits/PICToAdios.hpp"

#include <string>
#include <sstream>
#include <list>


namespace picongpu
{
    namespace adios
    {
        using namespace pmacc;

        namespace writeMeta
        {
            /** write openPMD species meta data
             *
             * @tparam numSpecies count of defined species
             */
            template<uint32_t numSpecies = bmpl::size<VectorAllSpecies>::type::value>
            struct OfAllSpecies
            {
                /** write meta data for species
                 *
                 * @param threadParams context of the adios plugin
                 * @param fullMeshesPath path to mesh entry
                 */
                void operator()(ThreadParams* threadParams, const std::string& fullMeshesPath) const
                {
                    // assume all boundaries are like the first species for openPMD 1.0.0
                    GetStringProperties<bmpl::at_c<VectorAllSpecies, 0>::type> particleBoundaryProp;
                    std::list<std::string> listParticleBoundary;
                    std::list<std::string> listParticleBoundaryParam;
                    for(uint32_t i = NumberOfExchanges<simDim>::value - 1; i > 0; --i)
                    {
                        if(FRONT % i == 0)
                        {
                            listParticleBoundary.push_back(particleBoundaryProp[ExchangeTypeNames()[i]]["name"].value);
                            listParticleBoundaryParam.push_back(
                                particleBoundaryProp[ExchangeTypeNames()[i]]["param"].value);
                        }
                    }
                    helper::GetADIOSArrayOfString getADIOSArrayOfString;
                    auto arrParticleBoundary = getADIOSArrayOfString(listParticleBoundary);
                    auto arrParticleBoundaryParam = getADIOSArrayOfString(listParticleBoundaryParam);

                    ADIOS_CMD(adios_define_attribute_byvalue(
                        threadParams->adiosGroupHandle,
                        "particleBoundary",
                        fullMeshesPath.c_str(),
                        adios_string_array,
                        listParticleBoundary.size(),
                        &(arrParticleBoundary.starts.at(0))));
                    ADIOS_CMD(adios_define_attribute_byvalue(
                        threadParams->adiosGroupHandle,
                        "particleBoundaryParameters",
                        fullMeshesPath.c_str(),
                        adios_string_array,
                        listParticleBoundaryParam.size(),
                        &(arrParticleBoundaryParam.starts.at(0))));
                }
            };

            /** specialization if no species are defined */
            template<>
            struct OfAllSpecies<0>
            {
                /** write meta data for species
                 *
                 * @param threadParams context of the adios plugin
                 * @param fullMeshesPath path to mesh entry
                 */
                void operator()(
                    ThreadParams* /* threadParams */,
                    const std::string& /* fullMeshesPath */
                ) const
                {
                }
            };

        } // namespace writeMeta

        struct WriteMeta
        {
            void operator()(ThreadParams* threadParams)
            {
                log<picLog::INPUT_OUTPUT>("ADIOS: (begin) write meta attributes.");

                traits::PICToAdios<uint32_t> adiosUInt32Type;
                traits::PICToAdios<float_X> adiosFloatXType;
                traits::PICToAdios<float_64> adiosDoubleType;

                /* openPMD attributes */
                /*   required */
                const std::string openPMDversion("1.0.0");
                ADIOS_CMD(adios_define_attribute_byvalue(
                    threadParams->adiosGroupHandle,
                    "openPMD",
                    "/",
                    adios_string,
                    1,
                    (void*) openPMDversion.c_str()));

                const uint32_t openPMDextension = 1; // ED-PIC ID
                ADIOS_CMD(adios_define_attribute_byvalue(
                    threadParams->adiosGroupHandle,
                    "openPMDextension",
                    "/",
                    adiosUInt32Type.type,
                    1,
                    (void*) &openPMDextension));

                const std::string basePath(ADIOS_PATH_ROOT "%T/");
                ADIOS_CMD(adios_define_attribute_byvalue(
                    threadParams->adiosGroupHandle,
                    "basePath",
                    "/",
                    adios_string,
                    1,
                    (void*) basePath.c_str()));

                const std::string meshesPath(ADIOS_PATH_FIELDS);
                ADIOS_CMD(adios_define_attribute_byvalue(
                    threadParams->adiosGroupHandle,
                    "meshesPath",
                    "/",
                    adios_string,
                    1,
                    (void*) meshesPath.c_str()));

                const std::string particlesPath(ADIOS_PATH_PARTICLES);
                ADIOS_CMD(adios_define_attribute_byvalue(
                    threadParams->adiosGroupHandle,
                    "particlesPath",
                    "/",
                    adios_string,
                    1,
                    (void*) particlesPath.c_str()));

                const std::string iterationEncoding("fileBased");
                ADIOS_CMD(adios_define_attribute_byvalue(
                    threadParams->adiosGroupHandle,
                    "iterationEncoding",
                    "/",
                    adios_string,
                    1,
                    (void*) iterationEncoding.c_str()));

                const std::string iterationFormat(
                    Environment<simDim>::get().Filesystem().basename(threadParams->adiosFilename)
                    + std::string("_%T.bp"));
                ADIOS_CMD(adios_define_attribute_byvalue(
                    threadParams->adiosGroupHandle,
                    "iterationFormat",
                    "/",
                    adios_string,
                    1,
                    (void*) iterationFormat.c_str()));

                /*   recommended */
                const std::string author = Environment<>::get().SimulationDescription().getAuthor();
                if(author.length() > 0)
                {
                    ADIOS_CMD(adios_define_attribute_byvalue(
                        threadParams->adiosGroupHandle,
                        "author",
                        "/",
                        adios_string,
                        1,
                        (void*) author.c_str()));
                }

                const std::string software("PIConGPU");
                ADIOS_CMD(adios_define_attribute_byvalue(
                    threadParams->adiosGroupHandle,
                    "software",
                    "/",
                    adios_string,
                    1,
                    (void*) software.c_str()));

                std::stringstream softwareVersion;
                softwareVersion << PICONGPU_VERSION_MAJOR << "." << PICONGPU_VERSION_MINOR << "."
                                << PICONGPU_VERSION_PATCH;
                if(!std::string(PICONGPU_VERSION_LABEL).empty())
                    softwareVersion << "-" << PICONGPU_VERSION_LABEL;
                ADIOS_CMD(adios_define_attribute_byvalue(
                    threadParams->adiosGroupHandle,
                    "softwareVersion",
                    "/",
                    adios_string,
                    1,
                    (void*) softwareVersion.str().c_str()));

                const std::string date = helper::getDateString("%F %T %z");
                ADIOS_CMD(adios_define_attribute_byvalue(
                    threadParams->adiosGroupHandle,
                    "date",
                    "/",
                    adios_string,
                    1,
                    (void*) date.c_str()));

                /*   ED-PIC */
                const std::string fullMeshesPath(threadParams->adiosBasePath + std::string(ADIOS_PATH_FIELDS));

                GetStringProperties<fields::Solver> fieldSolverProps;
                const std::string fieldSolver(fieldSolverProps["name"].value);
                ADIOS_CMD(adios_define_attribute_byvalue(
                    threadParams->adiosGroupHandle,
                    "fieldSolver",
                    fullMeshesPath.c_str(),
                    adios_string,
                    1,
                    (void*) fieldSolver.c_str()));
                if(fieldSolverProps.find("param") != fieldSolverProps.end())
                {
                    const std::string fieldSolverParam(fieldSolverProps["param"].value);
                    ADIOS_CMD(adios_define_attribute_byvalue(
                        threadParams->adiosGroupHandle,
                        "fieldSolverParameters",
                        fullMeshesPath.c_str(),
                        adios_string,
                        1,
                        (void*) fieldSolverParam.c_str()));
                }

                /* order as in axisLabels:
                 *    3D: z-lower, z-upper, y-lower, y-upper, x-lower, x-upper
                 *    2D: y-lower, y-upper, x-lower, x-upper
                 */
                GetStringProperties<fields::absorber::Absorber> fieldBoundaryProp;
                std::list<std::string> listFieldBoundary;
                std::list<std::string> listFieldBoundaryParam;
                for(uint32_t i = NumberOfExchanges<simDim>::value - 1; i > 0; --i)
                {
                    if(FRONT % i == 0)
                    {
                        listFieldBoundary.push_back(fieldBoundaryProp[ExchangeTypeNames()[i]]["name"].value);
                        listFieldBoundaryParam.push_back(fieldBoundaryProp[ExchangeTypeNames()[i]]["param"].value);
                    }
                }
                helper::GetADIOSArrayOfString getADIOSArrayOfString;
                auto arrFieldBoundary = getADIOSArrayOfString(listFieldBoundary);
                auto arrFieldBoundaryParam = getADIOSArrayOfString(listFieldBoundaryParam);

                ADIOS_CMD(adios_define_attribute_byvalue(
                    threadParams->adiosGroupHandle,
                    "fieldBoundary",
                    fullMeshesPath.c_str(),
                    adios_string_array,
                    listFieldBoundary.size(),
                    &(arrFieldBoundary.starts.at(0))));
                ADIOS_CMD(adios_define_attribute_byvalue(
                    threadParams->adiosGroupHandle,
                    "fieldBoundaryParameters",
                    fullMeshesPath.c_str(),
                    adios_string_array,
                    listFieldBoundaryParam.size(),
                    &(arrFieldBoundaryParam.starts.at(0))));

                writeMeta::OfAllSpecies<>()(threadParams, fullMeshesPath);

                GetStringProperties<fields::currentInterpolation::CurrentInterpolation> currentSmoothingProp;
                const std::string currentSmoothing(currentSmoothingProp["name"].value);
                ADIOS_CMD(adios_define_attribute_byvalue(
                    threadParams->adiosGroupHandle,
                    "currentSmoothing",
                    fullMeshesPath.c_str(),
                    adios_string,
                    1,
                    (void*) currentSmoothing.c_str()));

                if(currentSmoothingProp.find("param") != currentSmoothingProp.end())
                {
                    const std::string currentSmoothingParam(currentSmoothingProp["param"].value);
                    ADIOS_CMD(adios_define_attribute_byvalue(
                        threadParams->adiosGroupHandle,
                        "currentSmoothingParameters",
                        fullMeshesPath.c_str(),
                        adios_string,
                        1,
                        (void*) currentSmoothingParam.c_str()));
                }

                const std::string chargeCorrection("none");
                ADIOS_CMD(adios_define_attribute_byvalue(
                    threadParams->adiosGroupHandle,
                    "chargeCorrection",
                    fullMeshesPath.c_str(),
                    adios_string,
                    1,
                    (void*) chargeCorrection.c_str()));

                /* write current iteration */
                log<picLog::INPUT_OUTPUT>("ADIOS: meta: iteration");
                ADIOS_CMD(adios_define_attribute_byvalue(
                    threadParams->adiosGroupHandle,
                    "iteration",
                    threadParams->adiosBasePath.c_str(),
                    adiosUInt32Type.type,
                    1,
                    (void*) &threadParams->currentStep));

                /* write number of slides */
                log<picLog::INPUT_OUTPUT>("ADIOS: meta: sim_slides");
                uint32_t slides = MovingWindow::getInstance().getSlideCounter(threadParams->currentStep);
                ADIOS_CMD(adios_define_attribute_byvalue(
                    threadParams->adiosGroupHandle,
                    "sim_slides",
                    threadParams->adiosBasePath.c_str(),
                    adiosUInt32Type.type,
                    1,
                    (void*) &slides));

                /* openPMD: required time attributes */
                ADIOS_CMD(adios_define_attribute_byvalue(
                    threadParams->adiosGroupHandle,
                    "dt",
                    threadParams->adiosBasePath.c_str(),
                    adiosFloatXType.type,
                    1,
                    (void*) &DELTA_T));
                const float_X time = float_X(threadParams->currentStep) * DELTA_T;
                ADIOS_CMD(adios_define_attribute_byvalue(
                    threadParams->adiosGroupHandle,
                    "time",
                    threadParams->adiosBasePath.c_str(),
                    adiosFloatXType.type,
                    1,
                    (void*) &time));
                ADIOS_CMD(adios_define_attribute_byvalue(
                    threadParams->adiosGroupHandle,
                    "timeUnitSI",
                    threadParams->adiosBasePath.c_str(),
                    adiosDoubleType.type,
                    1,
                    (void*) &UNIT_TIME));

                /* write normed grid parameters */
                log<picLog::INPUT_OUTPUT>("ADIOS: meta: grid");
                ADIOS_CMD(adios_define_attribute_byvalue(
                    threadParams->adiosGroupHandle,
                    "cell_width",
                    threadParams->adiosBasePath.c_str(),
                    adiosFloatXType.type,
                    1,
                    (void*) &cellSize[0]));
                ADIOS_CMD(adios_define_attribute_byvalue(
                    threadParams->adiosGroupHandle,
                    "cell_height",
                    threadParams->adiosBasePath.c_str(),
                    adiosFloatXType.type,
                    1,
                    (void*) &cellSize[1]));

                ADIOS_CMD(adios_define_attribute_byvalue(
                    threadParams->adiosGroupHandle,
                    "cell_depth",
                    threadParams->adiosBasePath.c_str(),
                    adiosFloatXType.type,
                    1,
                    (void*) &cellSize[2]));


                /* write base units */
                log<picLog::INPUT_OUTPUT>("ADIOS: meta: units");
                ADIOS_CMD(adios_define_attribute_byvalue(
                    threadParams->adiosGroupHandle,
                    "unit_energy",
                    threadParams->adiosBasePath.c_str(),
                    adiosDoubleType.type,
                    1,
                    (void*) &UNIT_ENERGY));
                ADIOS_CMD(adios_define_attribute_byvalue(
                    threadParams->adiosGroupHandle,
                    "unit_length",
                    threadParams->adiosBasePath.c_str(),
                    adiosDoubleType.type,
                    1,
                    (void*) &UNIT_LENGTH));
                ADIOS_CMD(adios_define_attribute_byvalue(
                    threadParams->adiosGroupHandle,
                    "unit_speed",
                    threadParams->adiosBasePath.c_str(),
                    adiosDoubleType.type,
                    1,
                    (void*) &UNIT_SPEED));
                ADIOS_CMD(adios_define_attribute_byvalue(
                    threadParams->adiosGroupHandle,
                    "unit_time",
                    threadParams->adiosBasePath.c_str(),
                    adiosDoubleType.type,
                    1,
                    (void*) &UNIT_TIME));
                ADIOS_CMD(adios_define_attribute_byvalue(
                    threadParams->adiosGroupHandle,
                    "unit_mass",
                    threadParams->adiosBasePath.c_str(),
                    adiosDoubleType.type,
                    1,
                    (void*) &UNIT_MASS));
                ADIOS_CMD(adios_define_attribute_byvalue(
                    threadParams->adiosGroupHandle,
                    "unit_charge",
                    threadParams->adiosBasePath.c_str(),
                    adiosDoubleType.type,
                    1,
                    (void*) &UNIT_CHARGE));
                ADIOS_CMD(adios_define_attribute_byvalue(
                    threadParams->adiosGroupHandle,
                    "unit_efield",
                    threadParams->adiosBasePath.c_str(),
                    adiosDoubleType.type,
                    1,
                    (void*) &UNIT_EFIELD));
                ADIOS_CMD(adios_define_attribute_byvalue(
                    threadParams->adiosGroupHandle,
                    "unit_bfield",
                    threadParams->adiosBasePath.c_str(),
                    adiosDoubleType.type,
                    1,
                    (void*) &UNIT_BFIELD));

                /* write physical constants */
                log<picLog::INPUT_OUTPUT>("ADIOS: meta: mue0/eps0");
                ADIOS_CMD(adios_define_attribute_byvalue(
                    threadParams->adiosGroupHandle,
                    "mue0",
                    threadParams->adiosBasePath.c_str(),
                    adiosFloatXType.type,
                    1,
                    (void*) &MUE0));
                ADIOS_CMD(adios_define_attribute_byvalue(
                    threadParams->adiosGroupHandle,
                    "eps0",
                    threadParams->adiosBasePath.c_str(),
                    adiosFloatXType.type,
                    1,
                    (void*) &EPS0));

                log<picLog::INPUT_OUTPUT>("ADIOS: ( end ) wite meta attributes.");
            }
        };
    } // namespace adios
} // namespace picongpu
