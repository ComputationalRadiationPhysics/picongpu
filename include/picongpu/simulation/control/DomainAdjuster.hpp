/* Copyright 2018-2021 Rene Widera
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

#include "picongpu/fields/absorber/Absorber.hpp"

#include <pmacc/dimensions/DataSpace.hpp>

#include <pmacc/Environment.hpp>
#include <pmacc/mpi/reduceMethods/Reduce.hpp>
#include <pmacc/mpi/MPIReduce.hpp>
#include <pmacc/nvidia/functors/Add.hpp>
#include <pmacc/nvidia/functors/Max.hpp>
#include <pmacc/nvidia/functors/Min.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/mpi/GetMPI_StructAsArray.hpp>

#include <stdexcept>
#include <array>


namespace picongpu
{
    /** adjust domain sizes
     *
     * Extend the local offset, the local and global domain size to fulfill all
     * PIConGPU/PMacc conditions.
     */
    class DomainAdjuster
    {
    public:
        /** constructor
         *
         * @param numDevices number of devices [per dimension]
         * @param mpiPosition the position of the local device [per dimension]
         * @param isPeriodic if the outer simulation boundaries are periodic [per dimension]
         *                   1 is meaning periodic else 0
         * @param movingWindowEnabled if moving window is enabled
         */
        DomainAdjuster(
            DataSpace<simDim> const& numDevices,
            DataSpace<simDim> const& mpiPosition,
            DataSpace<simDim> const& isPeriodic,
            bool const movingWindowEnabled)
            : m_numDevices(numDevices)
            , m_mpiPosition(mpiPosition)
            , m_isPeriodic(isPeriodic)
            , m_movingWindowEnabled(movingWindowEnabled)
            , m_isMaster(mpiPosition == DataSpace<simDim>::create(0))
        {
        }

        /** adjust the domain size
         *
         * This method is a MPI collective operation and must be called from all MPI ranks.
         *
         * @param[in,out] globalDomainSize size of the global volume [in cells]
         * @param[in,out] localDomainSize size of the local volume [in cells]
         * @param[out] localDomainOffset local offset [in cells] relative to the origin of the global domain
         */
        void operator()(
            DataSpace<simDim>& globalDomainSize,
            DataSpace<simDim>& localDomainSize,
            DataSpace<simDim>& localDomainOffset)
        {
            m_globalDomainSize = globalDomainSize;
            m_localDomainSize = localDomainSize;

            for(uint32_t d = 0; d < simDim; ++d)
            {
                multipleOfSuperCell(d);
                minThreeSuperCells(d);
                greaterEqualThanAbsorber(d);
                deriveGlobalDomainSize(d);
                updateLocalDomainOffset(d);
            }

            if(globalDomainSize != m_globalDomainSize || localDomainSize != m_localDomainSize)
            {
                std::cout << " new grid size (global|local|offset): " << m_globalDomainSize.toString() << "|"
                          << m_localDomainSize.toString() << "|" << m_localDomainOffset.toString() << std::endl;
            }

            // write results back
            globalDomainSize = m_globalDomainSize;
            localDomainSize = m_localDomainSize;
            localDomainOffset = m_localDomainOffset;
        }

        /** only validate conditions
         *
         * Disable domain sizes auto adjustment.
         * The domain condition will be still checked.
         */
        void validateOnly()
        {
            m_validateOnly = true;
        }

    private:
        /** update local domain offset
         *
         * Share the local domain size with all MPI ranks and calculate the offset of the
         * local domain [in cells] relative to the origin of the global domain.
         *
         * @param dim dimension to update
         */
        void updateLocalDomainOffset(size_t const dim)
        {
            pmacc::GridController<simDim>& gc = pmacc::Environment<simDim>::get().GridController();

            int mpiPos(gc.getPosition()[dim]);
            int numMpiRanks = gc.getGlobalSize();

            // gather mpi position in the direction we are checking
            std::vector<int> mpiPositions(numMpiRanks);
            MPI_CHECK(MPI_Allgather(
                &mpiPos,
                1,
                MPI_INT,
                mpiPositions.data(),
                1,
                MPI_INT,
                gc.getCommunicator().getMPIComm()));

            // gather local sizes in the direction we are checking
            std::vector<uint64_t> allLocalSizes(numMpiRanks);
            uint64_t lSize = static_cast<uint64_t>(m_localDomainSize[dim]);
            MPI_CHECK(MPI_Allgather(
                &lSize,
                1,
                MPI_UINT64_T,
                allLocalSizes.data(),
                1,
                MPI_UINT64_T,
                gc.getCommunicator().getMPIComm()));

            uint64_t offset = 0u;
            for(size_t i = 0u; i < mpiPositions.size(); ++i)
            {
                if(mpiPositions[i] < mpiPos)
                    offset += allLocalSizes[i];
            }

            /* since we are not doing independent reduces per slice we need
             * to adjust the offset result by dividing with the number of
             * MPI ranks in all other dimensions.
             */
            offset /= static_cast<uint64_t>(m_numDevices.productOfComponents() / m_numDevices[dim]);
            m_localDomainOffset[dim] = static_cast<int>(offset);
        }

        /** ensure that the local size is a multiple of the supercell size
         *
         * @param dim dimension to update
         */
        void multipleOfSuperCell(size_t const dim)
        {
            int const sCellSize = SuperCellSize::toRT()[dim];
            // round up to full supercells
            int const validLocalSize = ((m_localDomainSize[dim] + sCellSize - 1) / sCellSize) * sCellSize;

            if(validLocalSize != m_localDomainSize[dim])
            {
                showMessage(
                    dim,
                    "Local grid size is not a multiple of supercell size.",
                    m_localDomainSize[dim],
                    validLocalSize);

                m_localDomainSize[dim] = validLocalSize;
            }
        }

        /** ensure that we have a CORE and BORDER region
         *
         * Each region must have the size of at least one supercell.
         *
         * @param dim dimension to update
         */
        void minThreeSuperCells(size_t const dim)
        {
            int numSuperCells = m_localDomainSize[dim] / SuperCellSize::toRT()[dim];

            if(numSuperCells < 3)
            {
                int newLocalDomainSize = 3 * SuperCellSize::toRT()[dim];
                showMessage(
                    dim,
                    "Local grid size is not containing at least 3 supercells.",
                    m_localDomainSize[dim],
                    newLocalDomainSize);

                m_localDomainSize[dim] = newLocalDomainSize;
            }
        }

        /** ensure that the absorber fits into the local domain
         *
         * The methods checks the local domain size only if the absorber for the
         * given dimension is enabled.
         *
         * @param dim dimension to update
         */
        void greaterEqualThanAbsorber(size_t const dim)
        {
            int validLocalSize = m_localDomainSize[dim];

            bool const isAbsorberEnabled = !m_isPeriodic[dim];
            bool const isBoundaryDevice = (m_mpiPosition[dim] == 0 || m_mpiPosition[dim] == m_numDevices[dim] - 1);
            if(isAbsorberEnabled && isBoundaryDevice)
            {
                size_t boundary = m_mpiPosition[dim] == 0u ? 0u : 1u;
                int maxAbsorberCells = fields::absorber::numCells[dim][boundary];

                if(m_movingWindowEnabled && dim == 1u)
                {
                    /* since the device changes their position during the simulation
                     * the negative and positive absorber cells must fit into the domain
                     */
                    maxAbsorberCells = static_cast<int>(
                        std::max(fields::absorber::numCells[dim][0], fields::absorber::numCells[dim][1]));
                }

                if(m_localDomainSize[dim] < maxAbsorberCells)
                {
                    int const sCellSize = SuperCellSize::toRT()[dim];
                    // round up to full supercells
                    validLocalSize = ((maxAbsorberCells + sCellSize - 1) / sCellSize) * sCellSize;
                }

                if(validLocalSize != m_localDomainSize[dim])
                {
                    showMessage(
                        dim,
                        "Local grid size must be greater or equal than the largest absorber.",
                        m_localDomainSize[dim],
                        validLocalSize);

                    m_localDomainSize[dim] = validLocalSize;
                }
            }
        }

        /** derive the local domain size
         *
         * Calculate the local domain size.
         * This function takes into account that the local domain size must be
         * equal for all domains if moving window is activated.
         *
         * @param dim dimension to update
         */
        void deriveLocalDomainSize(size_t const dim)
        {
            if(m_movingWindowEnabled && dim == 1u)
            {
                pmacc::mpi::MPIReduce mpiReduce;

                int globalMax;
                mpiReduce(
                    pmacc::nvidia::functors::Max(),
                    &globalMax,
                    &m_localDomainSize[dim],
                    1,
                    pmacc::mpi::reduceMethods::AllReduce());

                int globalMin;
                mpiReduce(
                    pmacc::nvidia::functors::Min(),
                    &globalMin,
                    &m_localDomainSize[dim],
                    1,
                    pmacc::mpi::reduceMethods::AllReduce());

                // local size must be equal for all devices in y direction
                if(m_isMaster && globalMax != globalMin)
                {
                    showMessage(
                        dim,
                        "Local grid size must be equal for all devices because moving window is enabled.",
                        m_localDomainSize[dim],
                        globalMax);
                }

                m_localDomainSize[dim] = globalMax;
            }
        }

        /** derive the global domain size
         *
         * Calculate the global domain size.
         *
         * @param dim dimension to update
         */
        void deriveGlobalDomainSize(size_t const dim)
        {
            uint64_t validGlobalGridSize = 0u;

            deriveLocalDomainSize(dim);

            if(m_movingWindowEnabled && dim == 1u)
            {
                // the local sizes in slide direction must be equal sized
                validGlobalGridSize = static_cast<uint64_t>(m_localDomainSize[dim] * m_numDevices[dim]);
            }
            else
            {
                uint64_t localDomainSize = static_cast<uint64_t>(m_localDomainSize[dim]);
                pmacc::mpi::MPIReduce mpiReduce;
                mpiReduce(
                    pmacc::nvidia::functors::Add(),
                    &validGlobalGridSize,
                    &localDomainSize,
                    1,
                    pmacc::mpi::reduceMethods::AllReduce());
                /* since we are not doing independent reduces per slice we need
                 * to adjust the reduce result by dividing the sizes of all other dimensions
                 * we are not check within the method call
                 */
                validGlobalGridSize /= static_cast<uint64_t>(m_numDevices.productOfComponents() / m_numDevices[dim]);
            }

            if(m_isMaster && validGlobalGridSize != static_cast<uint64_t>(m_globalDomainSize[dim]))
            {
                showMessage(dim, "Invalid global grid size.", m_globalDomainSize[dim], validGlobalGridSize);
            }

            m_globalDomainSize[dim] = static_cast<int>(validGlobalGridSize);
        }

        /** print a message to the user
         *
         * Throw an error with the message if is validateOnly was called.
         *
         * @param dim dimension index which was checked
         * @param msg problem description
         * @param currentSize current domain size in the given direction
         * @param updatedSize updated/corrected domain size for the given dimension
         */
        void showMessage(size_t const dim, std::string const& msg, int const currentSize, int const updatedSize) const
        {
            /**! lookup table to translate a dimension index into a name
             *
             * \warning `= { { ... } }` is not required by the c++11 standard but
             * is necessary for g++ 4.9
             */
            std::array<char, 3> const dimNames = {{'x', 'y', 'z'}};

            if(m_validateOnly)
                throw std::runtime_error(
                    std::string("Dimension ") + dimNames[dim] + ": " + msg + " Suggestion: set "
                    + std::to_string(currentSize) + " to " + std::to_string(updatedSize));
            else
                std::cout << "Dimension " << dimNames[dim] << ": " << msg << " Auto adjust from " << currentSize
                          << " to " << updatedSize << std::endl;
        }

        DataSpace<simDim> m_globalDomainSize;
        DataSpace<simDim> m_localDomainSize;
        DataSpace<simDim> m_localDomainOffset;
        DataSpace<simDim> const m_numDevices;
        DataSpace<simDim> const m_mpiPosition;
        DataSpace<simDim> const m_isPeriodic;
        bool const m_movingWindowEnabled;
        bool const m_isMaster;

        //! if true it will only validate the conditions
        bool m_validateOnly = false;
    };

} // namespace picongpu
