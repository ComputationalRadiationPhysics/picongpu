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

#include "picongpu/plugins/xrayDiffraction/ComputeLocalDomain.kernel"
#include "picongpu/plugins/xrayDiffraction/ReciprocalSpace.hpp"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/dimensions/DataSpaceOperations.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/traits/GetNumWorkers.hpp>

#include <cstdint>


namespace picongpu
{
namespace plugins
{
namespace xrayDiffraction
{
namespace detail
{

    /** Compute X-ray diffraction results for the local domain
     *
     * Contains the structure factor value for each scattering vector computed
     * for the local domain, equation (11) in J.C. E, L. Wang, S. Chen,
     * Y.Y. Zhang, S.N. Luo. GAPD: a GPU-accelerated atom-based polychromatic
     * diffraction simulation code // Journal of Synchrotron Radiation.
     * 25, 604-611 (2018).
     *
     * Is intended to be instantiated once for each local domain, calling
     * operator() fills the buffers on host with the current results.
     */
    struct ComputeLocalDomain
    {

        //! Reciprocal space of scattering vectors
        ReciprocalSpace reciprocalSpace;

        //! Structure factor values for each scattering vector
        GridBuffer<
            Complex,
            DIM1
        > structureFactor;

        //! Number of real particles (combined weighting of macroparticles)
        GridBuffer<
            float_64,
            DIM1
        > totalWeighting;

        /** Create a local domain computation functor
         *
         * @param reciprocalSpace reciprocal space
         */
        ComputeLocalDomain( ReciprocalSpace const & reciprocalSpace );

        /** Compute local domain results
         *
         * @tparam T_Species species type
         *
         * @param cellDescription mapping description
         */
        template< typename T_Species >
        void operator()( MappingDesc const & cellDescription );

    };

    ComputeLocalDomain::ComputeLocalDomain( ReciprocalSpace const & reciprocalSpace ):
        reciprocalSpace( reciprocalSpace ),
        structureFactor( DataSpace< DIM1 >( reciprocalSpace.size.productOfComponents() ) ),
        totalWeighting( DataSpace< DIM1 >( 1 ) )
    {
    }

    template< typename T_Species >
    void ComputeLocalDomain::operator()( MappingDesc const & cellDescription )
    {
        structureFactor.getDeviceBuffer( ).setValue( Complex( 0.0_X, 0.0_X ) );
        totalWeighting.getDeviceBuffer( ).setValue( 0.0 );

        auto const & subGrid = Environment< simDim >::get().SubGrid();
        auto const localDomainOffset = subGrid.getLocalDomain().offset;

        DataConnector &dc = Environment<>::get().DataConnector();
        auto particles = dc.get< T_Species >(
            T_Species::FrameType::getName(),
            true
        );

        // Run a thread per scattering vector
        constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
            pmacc::math::CT::volume< MappingDesc::SuperCellSize >::type::value
        >::value;
        auto const totalNumVectors = reciprocalSpace.size.productOfComponents( );
        auto const numBlocks = ( totalNumVectors + numWorkers - 1 ) / numWorkers;

        PMACC_KERNEL(
            KernelXrayDiffraction< numWorkers >{ }
        )(
            numBlocks,
            numWorkers
        )(
            particles->getDeviceParticlesBox( ),
            structureFactor.getDeviceBuffer( ).getDataBox( ),
            totalWeighting.getDeviceBuffer( ).getDataBox( ),
            localDomainOffset,
            cellDescription,
            reciprocalSpace
        );

        dc.releaseData( T_Species::FrameType::getName( ) );

        structureFactor.deviceToHost( );
        totalWeighting.deviceToHost( );
        __getTransactionEvent().waitForFinished( );
    }

} // namespace detail
} // namespace xrayDiffraction
} // namespace plugins
} // namespace picongpu
