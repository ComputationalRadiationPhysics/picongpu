/* Copyright 2013-2021 Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt, Alexander Debus
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

#include <pmacc/types.hpp>
#include "picongpu/simulation_defines.hpp"

#include "picongpu/simulation/control/Window.hpp"

namespace picongpu
{
    using namespace pmacc;

    /**
     * Singleton class managing the moving window, slides.
     * Can be used to create window views on the grid.
     */
    class MovingWindow
    {
    private:
        MovingWindow() = default;

        MovingWindow(MovingWindow& cc);

        void getCurrentSlideInfo(uint32_t currentStep, bool* doSlide, float_64* offsetFirstGPU)
        {
            if(doSlide)
                *doSlide = false;

            if(offsetFirstGPU)
                *offsetFirstGPU = 0.0;

            if(slidingWindowEnabled)
            {
                /* Sliding stayed enabled but if we reach the end step where we should stop sliding
                 * the moving window is freezed.
                 * All offsets will stay constant until the end of the simulation.
                 */
                if(currentStep >= endSlidingOnStep)
                    currentStep = endSlidingOnStep;

                const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();

                /* speed of the moving window */
                const float_64 windowMovingSpeed = float_64(SPEED_OF_LIGHT);

                /* defines in which direction the window moves
                 *
                 * 0 == x,  1 == y , 2 == z direction
                 *
                 * note: currently only y direction is supported
                 */
                const uint32_t moveDirection = 1;

                /* the moving window is smaller than the global domain by exactly one
                 * GPU (local domain size)
                 * \todo calculation of the globalWindowSizeInMoveDirection is constant should be
                 * only done once in it's own central object/api
                 */
                const uint32_t globalWindowSizeInMoveDirection
                    = subGrid.getGlobalDomain().size[moveDirection] - subGrid.getLocalDomain().size[moveDirection];

                const uint32_t gpuNumberOfCellsInMoveDirection = subGrid.getLocalDomain().size[moveDirection];

                /* unit PIConGPU length */
                const float_64 cellSizeInMoveDirection = float_64(cellSize[moveDirection]);

                const float_64 deltaWayPerStep = (windowMovingSpeed * float_64(DELTA_T));

                /* How many cells the virtual particle with speed of light is pushed forward
                 * at the begin of the simulation.
                 * The number of cells is round up thus we avoid window moves and slides
                 * depends on half cells.
                 */
                const uint32_t virtualParticleInitialStartCell
                    = math::ceil(float_64(globalWindowSizeInMoveDirection) * (float_64(1.0) - movePoint));

                /* Is the time step when the virtual particle **passed** the GPU next to the last
                 * in the current to the next step
                 */
                const uint32_t firstSlideStep
                    = math::ceil(
                          float_64(subGrid.getGlobalDomain().size[moveDirection] - virtualParticleInitialStartCell)
                          * cellSizeInMoveDirection / deltaWayPerStep)
                    - 1;

                /* way which the virtual particle must move before the window begins
                 * to move the first time [in pic length] */
                const float_64 wayToFirstMove
                    = float_64(globalWindowSizeInMoveDirection - virtualParticleInitialStartCell)
                    * cellSizeInMoveDirection;
                /* Is the time step when the virtual particle **passed** the moving window
                 * in the current to the next step
                 * Signed type of firstMoveStep to allow for edge case movePoint = 0.0
                 * for a moving window right from the start of the simulation.
                 */
                const int32_t firstMoveStep = math::ceil(wayToFirstMove / deltaWayPerStep) - 1;

                if(firstMoveStep <= int32_t(currentStep))
                {
                    /* calculate the current position of the virtual particle */
                    const float_64 virtualParticleWayPassed = deltaWayPerStep * float_64(currentStep);
                    const uint32_t virtualParticleWayPassedInCells
                        = uint32_t(math::floor(virtualParticleWayPassed / cellSizeInMoveDirection));
                    const uint32_t virtualParticlePositionInCells
                        = virtualParticleWayPassedInCells + virtualParticleInitialStartCell;

                    /* calculate the position of the virtual particle after the current step is calculated */
                    const float_64 nextVirtualParticleWayPassed = deltaWayPerStep * float_64(currentStep + 1);
                    const uint32_t nextVirtualParticleWayPassedInCells
                        = uint32_t(math::floor(nextVirtualParticleWayPassed / cellSizeInMoveDirection));
                    /* This position is used to detect the point in time where the virtual particle
                     * moves over a GPU border.
                     */
                    const uint32_t nextVirtualParticlePositionInCells
                        = nextVirtualParticleWayPassedInCells + virtualParticleInitialStartCell;

                    /* within the to be simulated time step (currentStep -> currentStep+1)
                     * the virtual particle will have reached at least the position
                     * of the cell behind the end of the initial global domain
                     * (also true for all later time steps)
                     */
                    const bool endOfInitialGlobalDomain = firstSlideStep <= currentStep;

                    /* virtual particle will pass a GPU border during the current
                     * (to be simulated) time step
                     */
                    const bool virtualParticlePassesGPUBorder
                        = (nextVirtualParticlePositionInCells % gpuNumberOfCellsInMoveDirection)
                        < (virtualParticlePositionInCells % gpuNumberOfCellsInMoveDirection);

                    if(endOfInitialGlobalDomain && virtualParticlePassesGPUBorder)
                    {
                        incrementSlideCounter(currentStep);
                        if(doSlide)
                            *doSlide = true;
                    }

                    /* valid range for the offset is [0;GPU number of cells in move direction) */
                    if(offsetFirstGPU)
                    {
                        /* since the moving window in PIConGPU always starts on the
                         * first plane (3D) / row (2D) of GPUs in move direction, this
                         * calculation is equal to the globalWindow.offset in move direction
                         *
                         * note: also works with windowMovingSpeed > c
                         */
                        *offsetFirstGPU = nextVirtualParticlePositionInCells % gpuNumberOfCellsInMoveDirection;
                    }
                }
            }
        }

        /** increment slide counter
         *
         * It is allowed to call this function more than once per time step
         * The function takes care that the counter is only incremented once
         * per simulation step
         *
         * @param current simulation step
         */
        void incrementSlideCounter(const uint32_t currentStep)
        {
            // do not slide twice in one simulation step
            if(isSlidingWindowActive(currentStep) && lastSlideStep < currentStep)
            {
                slideCounter++;
                lastSlideStep = currentStep;
            }
        }

        /** true is sliding window is activated
         *
         * How long the window is sliding is defined with endSlidingOnStep.
         */
        bool slidingWindowEnabled = false;

        /** Defines when to start sliding the window
         *
         * A virtual photon starts at t=0 at the lower end (min y) of the global
         * simulation box in the positive y direction. The window sliding starts at
         * the moment of time when the particle covers the movePoint ratio of the
         * global moving window size in the y direction.
         *
         * Note that with the moving window enabled, there is an additional "hidden"
         * row of local domains (and devices simulating them) at the y-front.
         * Therefore, the global moving window size in the y direction is the global
         * domain size minus a local domain size (which is required to be the same
         * for all domains).
         *
         * So, in short, the window starts sliding in time required to pass the
         * distance of movePoint * (global window size in y) when moving with
         * the speed of light.
         *
         * Setting movePoint to 0.0 makes the window start sliding at the start
         * of a simulation, and setting it to 1.0 makes it start sliding when the
         * virtual photon reaches the start of the "hidden" row of local domains.
         * It is permitted to use values outside of the [0.0, 1.0] interval to
         * achieve the effects of "pre-movement" and "delayed movement", however
         * this might complicate the setup and so not recommended unless essential.
         */
        float_64 movePoint;

        /** current number of slides since start of simulation */
        uint32_t slideCounter = 0u;

        /**
         * last simulation step with slide
         * used to prevent multiple slides per simulation step
         */
        uint32_t lastSlideStep = 0u;

        //! time step where the sliding window is stopped
        uint32_t endSlidingOnStep = 0u;

    public:
        /** Set window move point which defines when to start sliding the window
         *
         * See declaration of movePoint for a detailed explanation.
         *
         * @param point ratio of the global window size
         */
        void setMovePoint(float_64 const point)
        {
            movePoint = point;
        }

        /**
         * Set step where the simulation stops the moving window
         *
         * @param step 0 means no sliding window, else sliding is enabled until step is reached.
         */
        void setEndSlideOnStep(int32_t step)
        {
            // maybe we have a underflow in the cast, this is fine because it results in a very large number
            const uint32_t maxSlideStep = static_cast<uint32_t>(step);
            if(maxSlideStep < lastSlideStep)
                throw std::runtime_error("It is not allowed to stop the moving window in the past.");

            endSlidingOnStep = maxSlideStep;

            static bool firstCall = true;
            /* Disable or enable sliding window only in the first call.
             * Later changes of step will not influence if the sliding window is activated.
             */
            if(firstCall && endSlidingOnStep != 0u)
                slidingWindowEnabled = true;

            firstCall = false;
        }

        /**
         * Set the number of already performed moving window slides
         *
         * @param slides number of slides
         * @param currentStep current simulation timestep
         */
        void setSlideCounter(uint32_t slides, uint32_t currentStep)
        {
            slideCounter = slides;
            /* ensure that we will not change the slide counter with `incrementSlideCounter()`
             * in the same time step again
             */
            lastSlideStep = currentStep;
        }

        /**
         * Return the number of slides since start of simulation.
         * If slide occurs in \p currentStep, it is included in the result.
         *
         * @param currentStep current simulation step
         * @return number of slides
         */
        uint32_t getSlideCounter(uint32_t currentStep)
        {
            getCurrentSlideInfo(currentStep, nullptr, nullptr);
            return slideCounter;
        }

        /**
         * Returns if sliding window is enabled
         *
         * @return true if enabled, false otherwise
         */
        bool isEnabled() const
        {
            return slidingWindowEnabled;
        }

        /**
         * Returns if the window can move in the current step
         *
         * @return false, if Moving window is activated (isEnabled() == true) but already stopped.
         *         true if moving windows is enabled and simulation step is smaller than
         */
        bool isSlidingWindowActive(const uint32_t currenStep) const
        {
            return isEnabled() && currenStep < endSlidingOnStep;
        }

        /**
         * Return if a slide occurs in the current simulation step.
         *
         * @param currentStep current simulation step
         * @return true if slide in current step, false otherwise
         */
        bool slideInCurrentStep(uint32_t currentStep)
        {
            bool doSlide = false;

            if(slidingWindowEnabled)
            {
                getCurrentSlideInfo(currentStep, &doSlide, nullptr);
            }

            return doSlide;
        }

        /**
         * Return true if this is a 'bottom' GPU (y position is y_size - 1), false otherwise
         * only set if sliding window is active
         */
        bool isBottomGPU(void) const
        {
            Mask comm_mask = Environment<simDim>::get().GridController().getCommunicationMask();
            return !comm_mask.isSet(BOTTOM);
        }

        /**
         * Returns an instance of MovingWindow
         *
         * @return an instance
         */
        static MovingWindow& getInstance()
        {
            static MovingWindow instance;
            return instance;
        }

        /**
         * Return a window which describes the global and local moving window
         *
         * @param currentStep current simulation step
         * @return moving window
         */
        Window getWindow(uint32_t currentStep)
        {
            const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();

            /* Without moving window, the selected window spans the whole global domain.
             * \see https://github.com/ComputationalRadiationPhysics/picongpu/wiki/PIConGPU-domain-definitions
             *
             * The window's global offset is therefore zero inside the global domain.
             * The window's global and local size are equal to the SubGrid quantities.
             * The local window offset is the offset within the global window which
             * is equal to the local domain offset of the GPU.
             */
            Window window;
            window.localDimensions = subGrid.getLocalDomain();
            window.globalDimensions = Selection<simDim>(subGrid.getGlobalDomain().size);

            /* moving window can only slide in y direction */
            if(slidingWindowEnabled)
            {
                /* the moving window is smaller than the global domain by exactly one
                 * GPU (local domain size) in moving (y) direction
                 */
                window.globalDimensions.size.y() -= subGrid.getLocalDomain().size.y();

                float_64 offsetFirstGPU = 0.0;
                getCurrentSlideInfo(currentStep, nullptr, &offsetFirstGPU);

                /* while moving, the windows global offset within the global domain is between 0
                 * and smaller than the local domain's size in y.
                 */
                window.globalDimensions.offset.y() = offsetFirstGPU;

                /* set top/bottom if there are no communication partners
                 * for this GPU in the respective direction */
                const Mask comm_mask = Environment<simDim>::get().GridController().getCommunicationMask();
                const bool isTopGpu = !comm_mask.isSet(TOP);
                const bool isBottomGpu = !comm_mask.isSet(BOTTOM);

                if(isTopGpu)
                {
                    /* the windows local offset within the global window is reduced
                     * by the global window offset within the global domain
                     */
                    window.localDimensions.size.y() -= offsetFirstGPU;
                }
                else
                {
                    window.localDimensions.offset.y() = subGrid.getLocalDomain().offset.y() - offsetFirstGPU;
                    if(isBottomGpu)
                    {
                        window.localDimensions.size.y() = offsetFirstGPU;
                    }
                }
            }

            return window;
        }

        /**
         * Return a window which describes the global and local domain
         *
         * @param currentStep current simulation step
         * @return window over global/local domain
         */
        Window getDomainAsWindow(uint32_t currentStep) const
        {
            const SubGrid<simDim>& subGrid = Environment<simDim>::get().SubGrid();
            Window window;

            window.localDimensions = subGrid.getLocalDomain();
            window.globalDimensions = Selection<simDim>(subGrid.getGlobalDomain().size);

            return window;
        }
    };

} // namespace picongpu
