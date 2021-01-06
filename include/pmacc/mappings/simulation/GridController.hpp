/* Copyright 2013-2021 Axel Huebl, Felix Schmitt, Rene Widera,
 *                     Wolfgang Hoenig, Benjamin Worpitz
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/dimensions/DataSpaceOperations.hpp"
#include "pmacc/mappings/simulation/EnvironmentController.hpp"
#include "pmacc/communication/CommunicatorMPI.hpp"
#include "pmacc/mappings/simulation/SubGrid.hpp"

namespace pmacc
{
    /**
     * GridController manages grid information.
     *
     * GridController provides information for a DIM-dimensional grid
     * such as the number of GPU nodes and the current node's position in the grid
     * and manages sliding window.
     * GridController is a singleton.
     *
     * @tparam DIM dimension of the controlled grid
     */
    template<unsigned DIM>
    class GridController
    {
    public:
        /**
         * Initialisation of the controller.
         *
         * This methode must be called before any subgrids or buffers are used.
         *
         * @param nodes number of GPU nodes in each dimension
         * @param periodic specifying whether the grid is periodic (1) or not (0) in each dimension
         */
        void init(DataSpace<DIM> nodes, DataSpace<DIM> periodic = DataSpace<DIM>())
        {
            static bool commIsInit = false;
            if(!commIsInit)
            {
                gpuNodes = nodes;

                DataSpace<DIM3> tmp;
                DataSpace<DIM3> periodicTmp;
                tmp[0] = nodes[0];
                periodicTmp[0] = periodic[0];
                if(DIM < DIM2)
                {
                    tmp[1] = 1;
                    periodicTmp[1] = 1;
                }
                else
                {
                    tmp[1] = nodes[1];
                    periodicTmp[1] = periodic[1];
                }

                if(DIM < DIM3)
                {
                    tmp[2] = 1;
                    periodicTmp[2] = 1;
                }
                else
                {
                    tmp[2] = nodes[2];
                    periodicTmp[2] = periodic[2];
                }

                comm.init(tmp, periodicTmp);
                commIsInit = true;

                Environment<DIM>::get().EnvironmentController().setCommunicator(comm);
            }
        }

        /**
         * Returns the number of GPU nodes in each dimension.
         *
         * @return number of nodes
         */
        const DataSpace<DIM> getGpuNodes() const
        {
            return gpuNodes;
        }

        /**
         * Returns the position of the calling process' GPU in the grid.
         *
         * @return current GPU position
         * */
        const DataSpace<DIM> getPosition() const
        {
            return comm.getCoordinates();
        }

        /**
         * Returns the scalar position (rank) of this GPU,
         * depending on its current grid position
         *
         * @return current grid position as scalar value
         */
        uint32_t getScalarPosition() const
        {
            return DataSpaceOperations<DIM>::map(getGpuNodes(), getPosition());
        }

        /**
         * Returns the local rank of the caller on the current host.
         *
         * return local rank on host
         */
        uint32_t getHostRank() const
        {
            return comm.getHostRank();
        }

        /**
         * Returns the global MPI rank of the caller among all hosts.
         *
         * @return global MPI rank
         */
        uint32_t getGlobalRank() const
        {
            return comm.getRank();
        }

        /**
         * Returns the global MPI size.
         *
         * @return global number of MPI ranks
         */
        uint32_t getGlobalSize() const
        {
            return comm.getSize();
        }

        /**
         * Initialises a slide of the simulation area.
         *
         * Starts a slide of the simulation area. In the process, GPU nodes are
         * reassigned to new grid positions to enable large simulation areas
         * to be computed.
         * All nodes in the simulation must call this function at the same iteration.
         *
         * @return true if the position of the calling GPU is switched to the end, false otherwise
         */
        bool slide()
        {
            /* wait that all tasks are finished */
            Environment<DIM>::get().Manager().waitForAllTasks(); //

            bool result = comm.slide();

            updateDomainOffset();

            return result;
        }

        /**
         * Slides multiple times.
         *
         * Restores the state of the communicator and the domain offsets as
         * if the simulation has been slided for numSlides times.
         *
         * \warning you are not allowed to call this method if moving
         *          the simulation does not use a moving window,
         *          else static load balancing will break in y-direction
         *
         * @param[in] numSlides number of slides to slide
         * @return true if the position of gpu is switched to the end, else false
         */
        bool setStateAfterSlides(size_t numSlides)
        {
            // nothing to do, nothing to change
            // note: prevents destroying static load balancing in y for
            //       non-moving window simulations
            if(numSlides == 0)
                return false;

            bool result = comm.setStateAfterSlides(numSlides);
            updateDomainOffset(numSlides);
            return result;
        }

        /**
         * Returns a Mask which describes all neighbouring GPU nodes.
         *
         * @return Mask with all neighbors
         */
        const Mask& getCommunicationMask() const
        {
            return Environment<DIM>::get().EnvironmentController().getCommunicationMask();
        }

        /**
         * Returns the MPI communicator class
         *
         * @return current CommunicatorMPI
         */
        CommunicatorMPI<DIM>& getCommunicator()
        {
            return comm;
        }

    private:
        friend class Environment<DIM>;
        /**
         * Constructor
         */
        GridController() : gpuNodes(DataSpace<DIM>())
        {
        }

        /**
         * Constructor
         */
        GridController(const GridController& gc)
        {
        }

        /**
         * Sets globalDomain.offset & localDomain.offset using the current position.
         *
         * (This function is idempotent)
         *
         * @param[in] numSlides number of slides to slide
         *
         * \warning the implementation of this method is not compatible with
         *          static load balancing in y-direction
         */
        void updateDomainOffset(size_t numSlides = 1)
        {
            /* if we slide we must change our localDomain.offset of the simulation
             * (only change slide direction Y)
             */
            int gpuOffset_y = this->getPosition().y();
            const SubGrid<DIM>& subGrid = Environment<DIM>::get().SubGrid();
            DataSpace<DIM> localDomainOffset(subGrid.getLocalDomain().offset);
            DataSpace<DIM> globalDomainOffset(subGrid.getGlobalDomain().offset);
            /* this is allowed in the case that we use sliding window
             * because size in Y direction is the same for all gpus domains
             */
            localDomainOffset.y() = gpuOffset_y * subGrid.getLocalDomain().size.y();
            globalDomainOffset.y() += numSlides * subGrid.getLocalDomain().size.y();

            Environment<DIM>::get().SubGrid().setLocalDomainOffset(localDomainOffset);
            Environment<DIM>::get().SubGrid().setGlobalDomainOffset(globalDomainOffset);
        }

        /**
         * Returns the instance of the controller.
         *
         * This class is a singleton class.
         *
         * @return a controller instance
         */
        static GridController<DIM>& getInstance()
        {
            static GridController<DIM> instance;
            return instance;
        }

        /**
         * Communicator for MPI
         */
        static CommunicatorMPI<DIM> comm;

        /**
         * number of GPU nodes for each direction
         */
        DataSpace<DIM> gpuNodes;
    };

    template<unsigned DIM>
    CommunicatorMPI<DIM> GridController<DIM>::comm;

} // namespace pmacc
