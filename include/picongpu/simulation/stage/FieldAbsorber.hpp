/*
 * SPDX-FileCopyrightText: Sergei Bastrakov
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/fields/absorber.hpp"

#include <pmacc/Environment.hpp>

#include <boost/program_options/options_description.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace picongpu
{
    namespace simulation
    {
        namespace stage
        {
            /** Functor for the stage of the PIC loop performing field absorption
             *
             * This stage does not run by itself, but is needed to propagate command-line parameters
             */
            class FieldAbsorber
            {
            public:
                /** Register program options for field absorber
                 *
                 * @param desc program options following boost::program_options::options_description
                 */
                void registerHelp(po::options_description& desc)
                {
                    desc.add_options()(
                        "fieldAbsorber",
                        po::value<std::string>(&kindName),
                        std::string(
                            "Field absorber kind [exponential, pml] default: " + kindName
                            + ".\nWhen changing absorber, adjust parameters in fieldAbsorber.param")
                            .c_str());
                }

                /** Load the stage during loading of the simulation.
                 *
                 * This has to be called before any absorber instance or implementation can be safely used.
                 */
                void load()
                {
                    using namespace fields::absorber;
                    auto kind = Absorber::Kind{};
                    /* For the all-periodic boundaries case, we override the user's choice and use None.
                     * This is done for two reasons:
                     *     - easier compatibility with pre-existing checkpoints with such boundaries;
                     *     - optimization purposes to not have empty PML fields in checkpoints.
                     */
                    if(areAllBoundariesPeriodic())
                        kind = Absorber::Kind::None;
                    else if(kindName == "exponential")
                        kind = Absorber::Kind::Exponential;
                    else if(kindName == "pml")
                        kind = Absorber::Kind::Pml;
                    else
                        throw std::runtime_error("Unsupported field absorber type");
                    auto& absorberFactory = AbsorberFactory::get();
                    absorberFactory.setKind(kind);
                }

            private:
                //! Name set by program option
                std::string kindName = "pml";

                //! Return whether all boudaries are periodic
                bool areAllBoundariesPeriodic() const
                {
                    DataSpace<DIM3> const isPeriodicBoundary
                        = Environment<simDim>::get().EnvironmentController().getCommunicator().getPeriodic();
                    for(uint32_t axis = 0u; axis < simDim; axis++)
                        if(!isPeriodicBoundary[axis])
                            return false;
                    return true;
                }
            };

        } // namespace stage
    } // namespace simulation
} // namespace picongpu
