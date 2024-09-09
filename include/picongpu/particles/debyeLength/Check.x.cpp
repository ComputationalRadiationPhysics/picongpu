/* Copyright 2020-2023 Sergei Bastrakov
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

#include "picongpu/particles/debyeLength/Check.hpp"

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/debyeLength/Estimate.hpp"
#include "picongpu/plugins/output/ConstSpeciesAttributes.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/particles/meta/FindByNameOrType.hpp>
#include <pmacc/traits/GetCTName.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>


namespace picongpu::particles::debyeLength
{
    namespace detail
    {
        /** Return if given species is electron-like in charge and mass
         *
         * @tparam T_Species species type
         */
        template<typename T_Species>
        HINLINE bool isElectronLike()
        {
            using FrameType = typename T_Species::FrameType;
            auto const charge = plugins::output::GetChargeOrZero<FrameType>{}();
            auto const chargeRatio = charge / sim.pic.getElectronCharge();
            auto const mass = plugins::output::GetMassOrZero<FrameType>{}();
            auto const massRatio = mass / sim.pic.getElectronMass();
            // Allow slight deviation due to unit conversions
            auto const tolerance = 0.001_X;
            return (std::abs(chargeRatio - 1.0_X) < tolerance) && (std::abs(massRatio - 1.0_X) < tolerance);
        }

        //! Utility class for static storage of counter
        struct Counter
        {
            //! Get the counter value
            static std::uint32_t& value()
            {
                static uint32_t counter = 0u;
                return counter;
            }
        };

        /** Functor to count electron-like species
         *
         * Calling the functor increments Counter::value() by 1 if T_Species is electron-like.
         *
         * @tparam T_Species species type
         */
        template<typename T_Species>
        struct ElectonLikeSpeciesCounter : public Counter
        {
            //! Apply a functor for the species
            void operator()() const
            {
                if(isElectronLike<T_Species>())
                    Counter::value()++;
            }
        };

        /** Return the number of electron-like species in the given sequence of species types
         *
         * @tparam T_SpeciesSeq sequence of species types
         */
        template<typename T_SpeciesSeq>
        HINLINE std::uint32_t countElectronLikeSpecies()
        {
            Counter::value() = 0u;
            meta::ForEach<T_SpeciesSeq, ElectonLikeSpeciesCounter<boost::mpl::_1>> count;
            count();
            return Counter::value();
        }

        /** Functor to check Debye length resolution for the given species in the global domain
         *
         * Accepts any species, but only performs the check for electron-like species.
         *
         * @tparam T_Species species type
         */
        template<typename T_Species>
        struct CheckDebyeLength
        {
            /** Check Debye length resolution for the given species in the global domain
             *
             * Does nothing for not electron-like species.
             * Results are printed to the log.
             * This function must be called from all MPI ranks.
             *
             * @param cellDescription mapping for kernels
             * @param isPrinting whether the current process should print the result
             */
            HINLINE void operator()(MappingDesc const cellDescription, bool isPrinting) const
            {
                if(!isElectronLike<T_Species>())
                    return;
                // Only use supercells with at least this number of macroparticles
                uint32_t const minMacroparticlesPerSupercell = 10u;
                auto estimate = estimateGlobalDebyeLength<T_Species>(cellDescription, minMacroparticlesPerSupercell);
                if(!isPrinting)
                    return;

                auto const name = pmacc::traits::GetCTName_t<T_Species>::str();
                log<picLog::PHYSICS>("Resolving Debye length for species \"%1%\"?") % name;

                if(estimate.numUsedSupercells)
                {
                    auto const temperatureKeV = estimate.sumTemperatureKeV / estimate.sumWeighting;
                    auto const debyeLength = estimate.sumDebyeLength / estimate.sumWeighting;
                    auto maxCellSize = sim.pic.getCellSize()[0];
                    // For 2D do not use grid size along z, as it is always resolved
                    for(uint32_t d = 1; d < simDim; d++)
                        maxCellSize = std::max(maxCellSize, sim.pic.getCellSize()[d]);
                    auto const cellsPerDebyeLength = debyeLength / maxCellSize;
                    auto const debyeLengthSI = debyeLength * sim.unit.length();
                    log<picLog::PHYSICS>("Estimate used momentum variance in %1% supercells with at least %2% "
                                         "macroparticles each")
                        % estimate.numUsedSupercells % minMacroparticlesPerSupercell;
                    auto const ratioFailingSupercells = 100.0f * static_cast<float>(estimate.numFailingSupercells)
                        / static_cast<float>(estimate.numUsedSupercells);
                    log<picLog::PHYSICS>("%1% (%2% %3%) supercells had local Debye length estimate not resolved"
                                         " by a single cell")
                        % estimate.numFailingSupercells % ratioFailingSupercells % "%";
                    log<picLog::PHYSICS>("Estimated weighted average temperature %1% keV and corresponding "
                                         "Debye length %2% m.\n"
                                         "   The grid has %3% cells per average Debye length")
                        % temperatureKeV % debyeLengthSI % cellsPerDebyeLength;
                }
                else
                    log<picLog::PHYSICS>("Check skipped due to no supercells with at least %1% macroparticles")
                        % minMacroparticlesPerSupercell;
            }
        };

    } // namespace detail

    /** Check Debye length resolution
     *
     * Compute and print the weighted average Debye length for the electron species.
     * Print in how many supercells the locally estimated Debye length is not resolved with a single cell.
     *
     * The check is supposed to be called just after the particles are initialized at start of a simulation.
     * The results are output to log<picLog::PHYSICS>.
     *
     * This function must be called from all MPI ranks.
     *
     * @param cellDescription mapping for kernels
     */
    void check(MappingDesc const cellDescription)
    {
        bool isPrinting = (Environment<simDim>::get().GridController().getGlobalRank() == 0);

        // Filter out potential probes having no current flag
        using AllSpeciesWithCurrent =
            typename pmacc::particles::traits::FilterByFlag<VectorAllSpecies, current<>>::type;
        auto const numElectronLikeSpecies = detail::countElectronLikeSpecies<AllSpeciesWithCurrent>();

        // Only perform a check if there is a single electron-like species
        if(numElectronLikeSpecies == 0)
        {
            if(isPrinting)
                log<picLog::PHYSICS>("Debye length resolution check skipped due to no electron species found\n");
        }
        else if(numElectronLikeSpecies > 1)
        {
            if(isPrinting)
                log<picLog::PHYSICS>(
                    "Debye length resolution check skipped due to multiple electron-like species found\n");
        }
        else
        {
            meta::ForEach<AllSpeciesWithCurrent, detail::CheckDebyeLength<boost::mpl::_1>> checkDebyeLength;
            checkDebyeLength(cellDescription, isPrinting);
        }
    }
} // namespace picongpu::particles::debyeLength
