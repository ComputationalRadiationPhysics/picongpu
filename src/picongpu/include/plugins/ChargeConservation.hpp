/**
 * Copyright 2013-2016 Heiko Burau, Rene Widera
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

#include "plugins/ISimulationPlugin.hpp"
#include <boost/shared_ptr.hpp>
#include "cuSTL/algorithm/mpi/Reduce.hpp"

namespace picongpu
{
using namespace PMacc;

namespace po = boost::program_options;

/**
 * @class ChargeConservation
 * @brief maximum difference between electron charge density and div E
 *
 * WARNING: This plugin assumes a Yee-cell!
 * Do not use it together with other field solvers like `directional splitting` or `Lehe`
 */
class ChargeConservation : public ISimulationPlugin
{
private:
    std::string name;
    std::string prefix;
    uint32_t notifyPeriod;
    const std::string filename;
    MappingDesc* cellDescription;
    std::ofstream output_file;

    typedef boost::shared_ptr<PMacc::algorithm::mpi::Reduce<simDim> > AllGPU_reduce;
    AllGPU_reduce allGPU_reduce;

    void restart(uint32_t restartStep, const std::string restartDirectory);
    void checkpoint(uint32_t currentStep, const std::string checkpointDirectory);

    void pluginLoad();
public:
    ChargeConservation();
    virtual ~ChargeConservation() {}

    void notify(uint32_t currentStep);
    void setMappingDescription(MappingDesc*);
    void pluginRegisterHelp(po::options_description& desc);
    std::string pluginGetName() const;
};

}

#include "ChargeConservation.tpp"
