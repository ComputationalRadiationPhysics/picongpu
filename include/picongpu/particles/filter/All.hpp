/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include "picongpu/defines.hpp"

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/dimensions/DataSpace.hpp>

#include <string>

namespace picongpu
{
    namespace particles
    {
        namespace filter
        {
            namespace acc
            {
                //! check the particle handle
                struct All
                {
                    /** check particle handle
                     *
                     * @tparam T_Particle pmacc::Particles, type of the particle
                     * @tparam alpaka accelerator type
                     *
                     * @param worker lockstep worker
                     * @param particle  particle which is checked
                     * @return true if particle handle is valid, else false
                     */
                    template<typename T_Particle, typename T_Worker>
                    HDINLINE bool operator()(T_Worker const&, T_Particle const& particle)
                    {
                        return particle.isHandleValid();
                    }
                };

            } // namespace acc

            struct All
            {
                template<typename T_SpeciesType>
                struct apply
                {
                    using type = All;
                };

                /** create filter for the accelerator
                 *
                 * @tparam T_Worker lockstep::Worker, configuration of the worker
                 * @param offset (in superCells, without any guards) relative
                 *                        to the origin of the local domain
                 * @param configuration of the worker
                 */
                template<typename T_Worker>
                HDINLINE acc::All operator()(T_Worker const& worker, DataSpace<simDim> const&) const
                {
                    return acc::All{};
                }

                HINLINE static std::string getName()
                {
                    return std::string("all");
                }

                /** A filter is deterministic if the filter outcome is equal between evaluations. If so, set this
                 * variable to true, otherwise to false.
                 *
                 * Example: A filter were results depend on a random number generator must return false.
                 */
                static constexpr bool isDeterministic = true;
            };

        } // namespace filter
    } // namespace particles
} // namespace picongpu
