/* Copyright 2013-2023 Axel Huebl, Franz Poeschel
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once
#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/absorber/absorber.hpp"
#include "picongpu/fields/currentInterpolation/CurrentInterpolation.hpp"
#include "picongpu/plugins/common/openPMDVersion.def"
#include "picongpu/plugins/common/stringHelpers.hpp"
#include "picongpu/plugins/openPMD/openPMDWriter.def"
#include "picongpu/traits/SIBaseUnits.hpp"

#include <pmacc/Environment.hpp>

#include <list>
#include <sstream>
#include <string>

#include <openPMD/openPMD.hpp>


namespace picongpu
{
    namespace openPMD
    {
        using namespace pmacc;

        namespace writeMeta
        {
            /** write meta data for species
             *
             * @param threadParams context of the openPMD plugin
             * @param fullMeshesPath path to mesh entry
             */
            inline void ofAllSpecies(::openPMD::Series& series, uint32_t currentStep)
            {
                if constexpr(!pmacc::mp_empty<VectorAllSpecies>::value)
                {
                    /*
                     * @todo set boundary per species
                     */
                    GetStringProperties<pmacc::mp_front<pmacc::mp_push_back<VectorAllSpecies, void>>>
                        particleBoundaryProp; // add `void` so mp_front will compile with empty VectorAllSpecies
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

                    ::openPMD::Iteration iteration = series.writeIterations()[currentStep];
                    iteration.setAttribute("particleBoundary", listParticleBoundary);
                    iteration.setAttribute("particleBoundaryParameters", listParticleBoundaryParam);
                }
            }
        } // namespace writeMeta

        struct WriteMeta
        {
            void operator()(
                ::openPMD::Series& series,
                /*
                 * Sic! Callers must supply the iteration even though this method
                 * would be able to retrieve it from the series by using currentStep.
                 * But we don't know if callers are using the Streaming API or not,
                 * so let's not fell that decision for them.
                 */
                ::openPMD::Iteration& iteration,
                uint32_t currentStep,
                bool writeFieldMeta = true,
                bool writeParticleMeta = true,
                bool writeToLog = true)
            {
                if(writeToLog)
                    log<picLog::INPUT_OUTPUT>("openPMD: (begin) write meta attributes.");

                /*
                 * The openPMD API will kindly write the obligatory metadata by
                 * itself, so we don't need to do this manually. We give the
                 * optional metadata:
                 */

                // PIConGPU is writing particles and fields based on the openPMD standard 'ED-PIC' extension.
                constexpr uint32_t openPMDExtensionMask = 1u;
                series.setOpenPMDextension(openPMDExtensionMask);

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

                // PIConGPU IO file format version
                series.setAttribute("picongpuIOVersionMajor", picongpuIOVersionMajor);
                series.setAttribute("picongpuIOVersionMinor", picongpuIOVersionMinor);

                // don't write this if a previous run already wrote it
                if(!series.containsAttribute("date"))
                {
                    const std::string date = helper::getDateString("%F %T %z");
                    series.setDate(date);
                }

                ::openPMD::Container<::openPMD::Mesh>& meshes = iteration.meshes;

                // iteration-level attributes
                iteration.setDt<float_X>(sim.pic.getDt());
                iteration.setTime(float_X(currentStep) * sim.pic.getDt());
                iteration.setTimeUnitSI(sim.unit.time());

                if(writeFieldMeta)
                {
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
                    auto& absorber = fields::absorber::Absorber::get();
                    auto fieldBoundaryProp = absorber.getStringProperties();
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
                }

                if(writeParticleMeta)
                {
                    writeMeta::ofAllSpecies(series, currentStep);

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
                }

                /* write current iteration */
                if(writeToLog)
                    log<picLog::INPUT_OUTPUT>("openPMD: meta: iteration");
                iteration.setAttribute("iteration", currentStep); // openPMD API will not write this automatically

                /* write number of slides */
                if(writeToLog)
                    log<picLog::INPUT_OUTPUT>("openPMD: meta: sim_slides");
                uint32_t slides = MovingWindow::getInstance().getSlideCounter(currentStep);
                iteration.setAttribute("sim_slides", slides);

                /*
                 * Required time attributes are written automatically by openPMD API
                 */


                /* write normed grid parameters */
                if(writeToLog)
                    log<picLog::INPUT_OUTPUT>("openPMD: meta: grid");
                std::string names[3] = {"cell_width", "cell_height", "cell_depth"};
                for(unsigned i = 0; i < 3; ++i)
                {
                    iteration.setAttribute(names[i], sim.pic.getCellSize()[i]);
                }


                /* write base units */
                log<picLog::INPUT_OUTPUT>("openPMD: meta: units");
                iteration.setAttribute<double>("unit_energy", UNIT_ENERGY);
                iteration.setAttribute<double>("unit_length", sim.unit.length());
                iteration.setAttribute<double>("unit_speed", UNIT_SPEED);
                iteration.setAttribute<double>("unit_time", sim.unit.time());
                iteration.setAttribute<double>("unit_mass", UNIT_MASS);
                iteration.setAttribute<double>("unit_charge", UNIT_CHARGE);
                iteration.setAttribute<double>("unit_efield", UNIT_EFIELD);
                iteration.setAttribute<double>("unit_bfield", UNIT_BFIELD);


                /* write physical constants */
                iteration.setAttribute("mue0", MUE0);
                iteration.setAttribute("eps0", EPS0);

                if(writeToLog)
                    log<picLog::INPUT_OUTPUT>("openPMD: ( end ) wite meta attributes.");
            }
        };
    } // namespace openPMD
} // namespace picongpu
