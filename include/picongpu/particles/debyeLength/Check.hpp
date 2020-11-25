/* Copyright 2020 Sergei Bastrakov
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

#include "picongpu/particles/debyeLength/Check.kernel"

#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/Environment.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/memory/buffers/HostDeviceBuffer.hpp>
#include <pmacc/mpi/MPIReduce.hpp>
#include <pmacc/mpi/reduceMethods/AllReduce.hpp>
#include <pmacc/particles/meta/FindByNameOrType.hpp>
#include <pmacc/traits/GetNumWorkers.hpp>

#include <cstdint>


namespace picongpu
{
namespace particles
{
namespace debyeLength
{
namespace detail
{

    /** Check Debye length resolution for the given electron species
     *  in the local simulation volume
     *
     * @tparam T_ElectronSpecies electron species type
     *
     * @param cellDescription mapping for kernels
     */
    template< typename T_ElectronSpecies >
    HINLINE Result checkLocalDebyeLength( MappingDesc const cellDescription )
    {
        using Frame = typename T_ElectronSpecies::FrameType;
        DataConnector & dc = Environment< >::get( ).DataConnector( );
        auto & electrons = *( dc.get< T_ElectronSpecies >( Frame::getName(), true ) );

        pmacc::AreaMapping<
            CORE + BORDER,
            MappingDesc
        > mapper( cellDescription );
        constexpr uint32_t numWorkers = pmacc::traits::GetNumWorkers<
            pmacc::math::CT::volume< MappingDesc::SuperCellSize >::type::value
        >::value;

        auto hostDeviceBuffer = pmacc::HostDeviceBuffer< Result, 1 >{ 1u };
        auto hostBox = hostDeviceBuffer.getHostBuffer( ).getDataBox( );
        hostDeviceBuffer.hostToDevice( );
        auto kernel = DebyeLengthCheckKernel<
            numWorkers
        >{ };
        PMACC_KERNEL( kernel )(
            mapper.getGridDim( ),
            numWorkers
        )(
            electrons.getDeviceParticlesBox( ),
            mapper,
            hostDeviceBuffer.getDeviceBuffer( ).getDataBox( )
        );
        hostDeviceBuffer.deviceToHost( );

        // Copy is asynchronous, need to wait for it to finish
        __getTransactionEvent().waitForFinished();
        dc.releaseData( Frame::getName() );
        return hostBox( 0 );
    }

    //! Result of the Debye length check
    enum class Status
    {
        Passed,
        Failed,
        Skipped
    };

    /** Check Debye length resolution for the given electron species
     *  in the global simulation volume
     *
     * This function must be called from all MPI ranks.
     *
     * @tparam T_ElectronSpecies electron species type
     *
     * @param cellDescription mapping for kernels
     */
    template< typename T_ElectronSpecies >
    HINLINE Status checkDebyeLength( MappingDesc const cellDescription )
    {
        auto localResult = checkLocalDebyeLength< T_ElectronSpecies >(
            cellDescription
        );
        auto globalResult = Result{};
        pmacc::mpi::MPIReduce reduce;
        reduce(
            pmacc::nvidia::functors::Add(),
            &globalResult.numViolatingSupercells,
            &localResult.numViolatingSupercells,
            1,
            pmacc::mpi::reduceMethods::AllReduce()
        );
        reduce(
            pmacc::nvidia::functors::Add(),
            &globalResult.numActiveSupercells,
            &localResult.numActiveSupercells,
            1,
            pmacc::mpi::reduceMethods::AllReduce()
        );
        if( globalResult.numActiveSupercells )
        {
            auto const ratioFailing =
                static_cast< float_64 >( globalResult.numViolatingSupercells ) /
                static_cast< float_64 >( globalResult.numActiveSupercells );
            /* Due to random nature of temperature and estimates used internally,
             * we allow a relatively small number of supercells to fail the check
             */
            auto const thresholdFailing = 0.1;
            if( ratioFailing <= thresholdFailing )
                return Status::Passed;
            else
                return Status::Failed;
        }
        else
            return Status::Skipped;
    }

} // namespace detail

    /** Check Debye length resolution
     *
     * The check is currently done only for species called "e" and is supposed
     * to be called just after the particles are initialized as start of a
     * simulation. The results are output to log< picLog::PHYSICS >.
     *
     * This function must be called from all MPI ranks.
     *
     * @param cellDescription mapping for kernels
     */
    HINLINE void check( MappingDesc const cellDescription )
    {
        bool isPrinting = ( Environment<simDim>::get().GridController().getGlobalRank() == 0 );
        using ElectronSpecies = pmacc::particles::meta::FindByNameOrType_t<
            VectorAllSpecies,
            PMACC_CSTRING( "e" ),
            pmacc::errorHandlerPolicies::ReturnType< void >
        >;
        bool isElectronsFound = !std::is_same< ElectronSpecies, void >::value;
        if( isElectronsFound )
        {
            auto status = detail::checkDebyeLength< ElectronSpecies >( cellDescription );
            if( isPrinting )
                switch( status )
                {
                    case detail::Status::Passed:
                        log< picLog::PHYSICS >(
                            "Estimated Debye length for species \"e\" is resolved by the grid"
                        );
                        break;
                    case detail::Status::Failed:
                         log< picLog::PHYSICS >(
                            "Warning: estimated Debye length for species \"e\" is not resolved by the grid\n"
                            "Estimates are based on initial momentums of electrons\n"
                            "   (see: speciesInitialization.param)"
                        );
                        break;
                    case detail::Status::Skipped:
                        log< picLog::PHYSICS >(
                            "Debye length resolution check skipped, "
                            "as there are no particles of species \"e\"\n"
                        );
                }
        }
        else
            if( isPrinting )
                log< picLog::PHYSICS >("Debye length resolution check skipped, "
                    "as there are no species \"e\"\n");
    }

} // namespace debyeLength
} // namespace particles
} // namespace picongpu
