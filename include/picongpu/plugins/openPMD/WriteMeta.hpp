/* Copyright 2013-2021 Axel Huebl, Franz Poeschel
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
#include "picongpu/fields/absorber/ExponentialDamping.hpp"
#include "picongpu/fields/currentInterpolation/CurrentInterpolation.hpp"
#include "picongpu/plugins/common/stringHelpers.hpp"
#include "picongpu/plugins/openPMD/openPMDWriter.def"
#include "picongpu/plugins/openPMD/openPMDVersion.def"
#include "picongpu/simulation_defines.hpp"
#include "picongpu/traits/SIBaseUnits.hpp"

#include <pmacc/Environment.hpp>

#include <openPMD/openPMD.hpp>

#include <list>
#include <sstream>
#include <string>


namespace picongpu
{
    namespace openPMD
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
                 * @param threadParams context of the openPMD plugin
                 * @param fullMeshesPath path to mesh entry
                 */
                void operator()(ThreadParams* threadParams) const
                {
                    /*
                     * @todo set boundary per species
                     */
                    GetStringProperties<bmpl::at_c<VectorAllSpecies, 0>::type> particleBoundaryProp;
                    std::vector<std::string> listParticleBoundary;
                    std::vector<std::string> listParticleBoundaryParam;
                    auto n = NumberOfExchanges<simDim>::value;
                    listParticleBoundary.reserve(n - 1);
                    listParticleBoundaryParam.reserve(n - 1);
                    for(uint32_t i = n - 1; i > 0; --i)
                    {
                        if(FRONT % i == 0)
                        {
                            listParticleBoundary.push_back(particleBoundaryProp[ExchangeTypeNames()[i]]["name"].value);
                            listParticleBoundaryParam.push_back(
                                particleBoundaryProp[ExchangeTypeNames()[i]]["param"].value);
                        }
                    }

                    ::openPMD::Iteration iteration
                        = threadParams->openPMDSeries->WRITE_ITERATIONS[threadParams->currentStep];
                    iteration.setAttribute("particleBoundary", listParticleBoundary);
                    iteration.setAttribute("particleBoundaryParameters", listParticleBoundaryParam);
                }
            };

            /** specialization if no species are defined */
            template<>
            struct OfAllSpecies<0>
            {
                /** write meta data for species
                 *
                 * @param threadParams context of the openPMD plugin
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
                log<picLog::INPUT_OUTPUT>("openPMD: (begin) write meta attributes.");

                ::openPMD::Series& series = *threadParams->openPMDSeries;

                /*
                 * The openPMD API will kindly write the obligatory metadata by
                 * itself, so we don't need to do this manually. We give the
                 * optional metadata:
                 */

                /*   recommended */
                const std::string author = Environment<>::get().SimulationDescription().getAuthor();
                if(author.length() > 0)
                {
                    series.setAuthor(author);
                }

                const std::string software("PIConGPU");

                std::stringstream softwareVersion;
                softwareVersion << PICONGPU_VERSION_MAJOR << "." << PICONGPU_VERSION_MINOR << "."
                                << PICONGPU_VERSION_PATCH;
                if(!std::string(PICONGPU_VERSION_LABEL).empty())
                    softwareVersion << "-" << PICONGPU_VERSION_LABEL;
                series.setSoftware(software, softwareVersion.str());

                const std::string date = helper::getDateString("%F %T %z");
                series.setDate(date);

                ::openPMD::Iteration iteration = series.WRITE_ITERATIONS[threadParams->currentStep];
                ::openPMD::Container<::openPMD::Mesh>& meshes = iteration.meshes;

                // iteration-level attributes
                iteration.setDt<float_X>(DELTA_T);
                iteration.setTime(float_X(threadParams->currentStep) * DELTA_T);
                iteration.setTimeUnitSI(UNIT_TIME);

                GetStringProperties<fields::Solver> fieldSolverProps;
                const std::string fieldSolver(fieldSolverProps["name"].value);
                meshes.setAttribute("fieldSolver", fieldSolver);

                if(fieldSolverProps.find("param") != fieldSolverProps.end())
                {
                    const std::string fieldSolverParam(fieldSolverProps["param"].value);
                    meshes.setAttribute("fieldSolverParameters", fieldSolverParam);
                }

                /* order as in axisLabels:
                 *    3D: z-lower, z-upper, y-lower, y-upper, x-lower, x-upper
                 *    2D: y-lower, y-upper, x-lower, x-upper
                 */
                GetStringProperties<fields::absorber::Absorber> fieldBoundaryProp;
                std::vector<std::string> listFieldBoundary;
                std::vector<std::string> listFieldBoundaryParam;
                auto n = NumberOfExchanges<simDim>::value;
                listFieldBoundary.reserve(n - 1);
                listFieldBoundaryParam.reserve(n - 1);
                for(uint32_t i = n - 1; i > 0; --i)
                {
                    if(FRONT % i == 0)
                    {
                        listFieldBoundary.push_back(fieldBoundaryProp[ExchangeTypeNames()[i]]["name"].value);
                        listFieldBoundaryParam.push_back(fieldBoundaryProp[ExchangeTypeNames()[i]]["param"].value);
                    }
                }

                meshes.setAttribute("fieldBoundary", listFieldBoundary);
                meshes.setAttribute("fieldBoundaryParameters", listFieldBoundaryParam);

                writeMeta::OfAllSpecies<>()(threadParams);

                GetStringProperties<fields::currentInterpolation::CurrentInterpolation> currentSmoothingProp;
                const std::string currentSmoothing(currentSmoothingProp["name"].value);
                meshes.setAttribute("currentSmoothing", currentSmoothing);

                if(currentSmoothingProp.find("param") != currentSmoothingProp.end())
                {
                    const std::string currentSmoothingParam(currentSmoothingProp["param"].value);
                    meshes.setAttribute("currentSmoothingParameters", currentSmoothingParam);
                }

                const std::string chargeCorrection("none");
                meshes.setAttribute("chargeCorrection", chargeCorrection);

                /* write current iteration */
                log<picLog::INPUT_OUTPUT>("openPMD: meta: iteration");
                iteration.setAttribute(
                    "iteration",
                    threadParams->currentStep); // openPMD API will not write this
                                                // automatically

                /* write number of slides */
                log<picLog::INPUT_OUTPUT>("openPMD: meta: sim_slides");
                uint32_t slides = MovingWindow::getInstance().getSlideCounter(threadParams->currentStep);
                iteration.setAttribute("sim_slides", slides);

                /*
                 * Required time attributes are written automatically by openPMD API
                 */


                /* write normed grid parameters */
                log<picLog::INPUT_OUTPUT>("openPMD: meta: grid");
                std::string names[3] = {"cell_width", "cell_height", "cell_depth"};
                for(unsigned i = 0; i < 3; ++i)
                {
                    iteration.setAttribute(names[i], cellSize[i]);
                }


                /* write base units */
                log<picLog::INPUT_OUTPUT>("openPMD: meta: units");
                iteration.setAttribute<double>("unit_energy", UNIT_ENERGY);
                iteration.setAttribute<double>("unit_length", UNIT_LENGTH);
                iteration.setAttribute<double>("unit_speed", UNIT_SPEED);
                iteration.setAttribute<double>("unit_time", UNIT_TIME);
                iteration.setAttribute<double>("unit_mass", UNIT_MASS);
                iteration.setAttribute<double>("unit_charge", UNIT_CHARGE);
                iteration.setAttribute<double>("unit_efield", UNIT_EFIELD);
                iteration.setAttribute<double>("unit_bfield", UNIT_BFIELD);


                /* write physical constants */
                iteration.setAttribute("mue0", MUE0);
                iteration.setAttribute("eps0", EPS0);

                log<picLog::INPUT_OUTPUT>("openPMD: ( end ) wite meta attributes.");
            }
        };
    } // namespace openPMD
} // namespace picongpu
