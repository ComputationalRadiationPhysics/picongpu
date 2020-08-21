/* Copyright 2013-2020 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch,
 *                     Klaus Steiniger, Felix Schmitt, Benjamin Worpitz,
 *                     Juncheng E, Sergei Bastrakov
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

#include "picongpu/plugins/xrayDiffraction/ComputeLocalDomain.hpp"
#include "picongpu/plugins/xrayDiffraction/ReciprocalSpace.hpp"

#include <pmacc/mpi/MPIReduce.hpp>
#include <pmacc/mpi/reduceMethods/Reduce.hpp>
#include <pmacc/nvidia/functors/Add.hpp>

#include <cstdint>
#include <vector>


namespace picongpu
{
namespace plugins
{
namespace xrayDiffraction
{
namespace detail
{

    /** Compute X-ray diffraction results aggregated for the global domain
     *
     * Contains the structure factor value for each scattering vector computed
     * for the local domain, equation (5) in J.C. E, L. Wang, S. Chen,
     * Y.Y. Zhang, S.N. Luo. GAPD: a GPU-accelerated atom-based polychromatic
     * diffraction simulation code // Journal of Synchrotron Radiation.
     * 25, 604-611 (2018).
     *
     * Is intended to be instantiated once for each local domain, calling
     * operator() aggregates the results on the master rank.
     */
    struct ComputeGlobalDomain
    {
        //! Reciprocal space
        ReciprocalSpace reciprocalSpace;

        //! Structure factor (5) for each scattering vector on the global domain
        std::vector< Complex > structureFactor;

        /** Diffraction intensity (4) for each scattering vector
         *  on the global domain
         */
        std::vector< float_X > diffractionIntensity;

        //! Number of real particles (combined weighting of macroparticles)
        float_64 totalWeighting;

        //! Reducer for distributed memory
        mpi::MPIReduce reduce;

        /** Create a global domain computation functor
         *
         * @param reciprocalSpace reciprocal space
         */
        ComputeGlobalDomain( ReciprocalSpace const & reciprocalSpace );

        /** Aggregate local domain results into the global result
         *  on the master rank
         *
         * @param localDomainResult local domain result
         */
        void operator()( ComputeLocalDomain const & localDomainResult );

    private:

        /** Perform reduction of the local domain results on the master rank
         *
         * @param localDomainResult local domain result
         */
        void reduceResults( ComputeLocalDomain const & localDomainResult );

        //! Compute diffraction intensity from the structure factor
        void computeIntensity();

    };

    ComputeGlobalDomain::ComputeGlobalDomain(
        ReciprocalSpace const & reciprocalSpace
    ):
        reciprocalSpace( reciprocalSpace )
    {
        auto const size = reciprocalSpace.size.productOfComponents();
        structureFactor.resize( size );
        diffractionIntensity.resize( size );
    }

    void ComputeGlobalDomain::operator()(
        ComputeLocalDomain const & localDomainResult
    )
    {
        reduceResults( localDomainResult );
        computeIntensity( );
    }

    void ComputeGlobalDomain::reduceResults(
        ComputeLocalDomain const & localDomainResult
    )
    {
        reduce(
            nvidia::functors::Add( ),
            structureFactor.data( ),
            localDomainResult.structureFactor.getHostBuffer( ).getBasePointer( ),
            structureFactor.size(),
            mpi::reduceMethods::Reduce( )
        );
        totalWeighting = 0._X;
        reduce(
            nvidia::functors::Add( ),
            &totalWeighting,
            localDomainResult.totalWeighting.getHostBuffer( ).getBasePointer( ),
            1,
            mpi::reduceMethods::Reduce( )
        );
        computeIntensity();
    }

    void ComputeGlobalDomain::computeIntensity()
    {
        auto const size = diffractionIntensity.size();
        for( size_t i = 0; i < size; i++ )
        {
            auto const factor = structureFactor[ i ];
            diffractionIntensity[ i ] = (
                factor.get_real() * factor.get_real() +
                factor.get_imag() * factor.get_imag()
            ) / totalWeighting;
        }
    }

} // namespace detail
} // namespace xrayDiffraction
} // namespace plugins
} // namespace picongpu
