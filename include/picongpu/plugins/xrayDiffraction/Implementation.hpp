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

#include "picongpu/plugins/xrayDiffraction/ComputeGlobalDomain.hpp"
#include "picongpu/plugins/xrayDiffraction/ComputeLocalDomain.hpp"
#include "picongpu/plugins/xrayDiffraction/ReciprocalSpace.hpp"
#include "picongpu/plugins/xrayDiffraction/Writer.hpp"

#include <pmacc/memory/MakeUnique.hpp>
#include <pmacc/mpi/MPIReduce.hpp>
#include <pmacc/mpi/reduceMethods/Reduce.hpp>

#include <cstdint>
#include <memory>
#include <string>


namespace picongpu
{
namespace plugins
{
namespace xrayDiffraction
{
namespace detail
{

    /** Implementation of X-ray diffraction computation and output
     *
     * @tparam T_Species species type
     */
    template< typename T_Species >
    class Implementation
    {
    public:

        /** Create an implementation of X-ray diffraction computation and output
         *
         * @param reciprocalSpace reciprocal space
         * @param prefix prefix for output
         */
        Implementation(
            ReciprocalSpace const & reciprocalSpace,
            std::string const & prefix
        );

        /** Compute and output diffraction
         *
         * @param currentStep current time iteration
         * @param cellDescription mapping for kernels
         */
        void operator()(
            uint32_t const currentStep,
            MappingDesc const & cellDescription
        );

    private:

        //! Reciprocal space
        ReciprocalSpace reciprocalSpace;

        //! Global domain computation functor
        ComputeGlobalDomain computeGlobalDomain;

        //! Local domain computation functor
        ComputeLocalDomain computeLocalDomain;

        //! Writer to output the results
        std::unique_ptr< Writer > writer;

        //! If the current rank is master
        bool isMasterRank;

    };

    template< typename T_Species >
    Implementation< T_Species >::Implementation(
        ReciprocalSpace const & reciprocalSpace,
        std::string const & prefix
    ):
        reciprocalSpace( reciprocalSpace ),
        computeLocalDomain( reciprocalSpace ),
        computeGlobalDomain( reciprocalSpace )
    {
        mpi::MPIReduce reduce;
        isMasterRank = reduce.hasResult( mpi::reduceMethods::Reduce() );
        if( isMasterRank )
            writer = memory::makeUnique< Writer >( prefix );
    }

    template< typename T_Species >
    void Implementation< T_Species >::operator()(
        uint32_t const currentStep,
        MappingDesc const & cellDescription
    )
    {
        computeLocalDomain.operator()< T_Species >( cellDescription );
        computeGlobalDomain( computeLocalDomain );
        if( isMasterRank )
        {
            writer->write(
                computeGlobalDomain,
                currentStep
            );
        }
    }

} // namespace detail
} // namespace xrayDiffraction
} // namespace plugins
} // namespace picongpu
