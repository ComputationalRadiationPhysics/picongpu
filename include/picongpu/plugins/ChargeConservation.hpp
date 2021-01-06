/* Copyright 2013-2021 Heiko Burau, Rene Widera, Axel Huebl
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
#include "picongpu/plugins/ISimulationPlugin.hpp"
#include "picongpu/particles/traits/SpeciesEligibleForSolver.hpp"

#include <pmacc/cuSTL/algorithm/mpi/Reduce.hpp>
#include <pmacc/traits/HasIdentifiers.hpp>
#include <pmacc/traits/HasFlag.hpp>

#include <boost/mpl/and.hpp>
#include <boost/shared_ptr.hpp>


namespace picongpu
{
    using namespace pmacc;

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
        std::string notifyPeriod;
        const std::string filename;
        MappingDesc* cellDescription;
        std::ofstream output_file;

        using AllGPU_reduce = boost::shared_ptr<pmacc::algorithm::mpi::Reduce<simDim>>;
        AllGPU_reduce allGPU_reduce;

        HINLINE void restart(uint32_t restartStep, const std::string restartDirectory);
        HINLINE void checkpoint(uint32_t currentStep, const std::string checkpointDirectory);

        HINLINE void pluginLoad();

    public:
        HINLINE ChargeConservation();
        virtual ~ChargeConservation()
        {
        }

        HINLINE void notify(uint32_t currentStep);
        HINLINE void setMappingDescription(MappingDesc*);
        HINLINE void pluginRegisterHelp(po::options_description& desc);
        HINLINE std::string pluginGetName() const;
    };

    namespace particles
    {
        namespace traits
        {
            template<typename T_Species>
            struct SpeciesEligibleForSolver<T_Species, ChargeConservation>
            {
                using FrameType = typename T_Species::FrameType;

                // this plugin needs at least the weighting particle attribute
                using RequiredIdentifiers = MakeSeq_t<weighting>;

                using SpeciesHasIdentifiers =
                    typename pmacc::traits::HasIdentifiers<FrameType, RequiredIdentifiers>::type;

                // and also a charge ratio for a charge density
                using SpeciesHasFlags = typename pmacc::traits::HasFlag<FrameType, chargeRatio<>>::type;

                using type = typename bmpl::and_<SpeciesHasIdentifiers, SpeciesHasFlags>;
            };

        } // namespace traits
    } // namespace particles
} // namespace picongpu

#include "ChargeConservation.tpp"
