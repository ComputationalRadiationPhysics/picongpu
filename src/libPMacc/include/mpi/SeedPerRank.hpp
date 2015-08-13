/**
 * Copyright 2014 Axel Huebl
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "types.h"
#include "Environment.hpp"

namespace PMacc
{
namespace mpi
{
    /** Calculate a Seed per Rank
     *
     * This functor derives a unqiue seed for each MPI rank (or GPU) from
     * a given global seed in a deterministic manner.
     *
     * \tparam T_DIM Dimensionality of the simulation (1-3 D)
     */
    template <unsigned T_DIM>
    struct SeedPerRank
    {
        /** Functor implementation
         *
         * This method provides a guaranteed unique number per MPI rank
         * (or GPU). When a (only locally unique) localShift parameter is used
         * it is furthermore guaranteed that this number does not collide
         * with an other seed.
         *
         * \param seed initial seed to vary two identical simulations
         * \param localShift e.g. a unique species id
         * \return uint32_t seed
         */
        uint32_t
        operator()( uint32_t seed, uint32_t localShift = 0 )
        {
            PMACC_AUTO(&gc, PMacc::Environment<T_DIM>::get().GridController());

            seed ^= gc.getGlobalSize( ) * localShift +
                    gc.getGlobalRank( );
            return seed;
        }
    };

} /* namespace mpi */
} /* namespace picongpu */
