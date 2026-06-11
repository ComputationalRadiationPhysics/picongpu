/*
 * SPDX-FileCopyrightText: Axel Huebl, Heiko Burau, Rene Widera, Felix Schmitt, Alexander Debus
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/simulation/control/Window.hpp"
#include "picongpu/unitless/precision.unitless"

#include <pmacc/types.hpp>

#include <cstdint>

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

        MovingWindow(MovingWindow& cc) = delete;

        struct ComputedConstants
        {
            uint32_t gpuNumberOfCellsInMoveDirection;
            float_64 cellSizeInMoveDirection;
            float_64 deltaWayPerStep;
            uint32_t virtualParticleInitialStartCell;
            uint32_t firstSlideStep;
            int32_t firstMoveStep;

            ComputedConstants(float_64 movePoint)
            {
                SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();

                /* speed of the moving window */
                auto const windowMovingSpeed = static_cast<float_64>(sim.pic.getSpeedOfLight());

                gpuNumberOfCellsInMoveDirection = subGrid.getLocalDomain().size[moveDirection];

                /* the moving window is smaller than the global domain by exactly one
                 * GPU (local domain size)
                 */
                uint32_t const globalWindowSizeInMoveDirection
                    = subGrid.getGlobalDomain().size[moveDirection] - gpuNumberOfCellsInMoveDirection;

                /* unit PIConGPU length */
                cellSizeInMoveDirection = static_cast<float_64>(sim.pic.getCellSize()[moveDirection]);

                deltaWayPerStep = windowMovingSpeed * static_cast<float_64>(sim.pic.getDt());

                /* How many cells the virtual particle with speed of light is pushed forward
                 * at the begin of the simulation.
                 * The number of cells is round up thus we avoid window moves and slides
                 * depends on half cells.
                 */
                virtualParticleInitialStartCell = math::ceil(
                    static_cast<float_64>(globalWindowSizeInMoveDirection) * (static_cast<float_64>(1.0) - movePoint));

                /* Is the time step when the virtual particle **passed** the GPU next to the last
                 * in the current to the next step
                 */
                firstSlideStep
                    = static_cast<uint32_t>(math::ceil(
                          static_cast<float_64>(
                              subGrid.getGlobalDomain().size[moveDirection] - virtualParticleInitialStartCell)
                          * cellSizeInMoveDirection / deltaWayPerStep))
                      - 1;

                /* way which the virtual particle must move before the window begins
                 * to move the first time [in pic length] */
                auto const wayToFirstMove
                    = static_cast<float_64>(globalWindowSizeInMoveDirection - virtualParticleInitialStartCell)
                      * cellSizeInMoveDirection;
                /* Is the time step when the virtual particle **passed** the moving window
                 * in the current to the next step
                 * Signed type of firstMoveStep to allow for edge case movePoint = 0.0
                 * for a moving window right from the start of the simulation.
                 */
                firstMoveStep = static_cast<int32_t>(math::ceil(wayToFirstMove / deltaWayPerStep)) - 1;
            }
        };

        ComputedConstants const& getOrComputeConstants()
        {
            if(!computedConstants)
            {
                computedConstants = ComputedConstants(movePoint);
            }
            return *computedConstants;
        }

        // Has a side effect. May increment the slide counter
        // It is assumed that this is only called after movePoint and endSlideOnStep are set
        void getCurrentSlideInfo(uint32_t currentStep, bool* doSlide, float_64* offsetFirstGPU)
        {
            if(doSlide)
                *doSlide = false;
            // Default offset before movement starts
            if(offsetFirstGPU)
                *offsetFirstGPU = 0; // Assume 0 offset if window hasn't started moving relative to particle frame

            if(slidingWindowEnabled)
            {
                auto const& constants = getOrComputeConstants();

                auto const realCurStep = currentStep;
                // Clamp currentStep if sliding has ended
                if(currentStep > endSlidingOnStep)
                    currentStep = endSlidingOnStep;

                // Check if the window should be moving yet based on the precomputed step
                if(constants.firstMoveStep <= static_cast<int32_t>(currentStep))
                {
                    /* calculate the current position of the virtual particle */
                    float_64 const virtualParticleWayPassed
                        = constants.deltaWayPerStep * static_cast<float_64>(currentStep);
                    uint32_t const virtualParticleWayPassedInCells
                        = uint32_t(math::floor(virtualParticleWayPassed / constants.cellSizeInMoveDirection));
                    uint32_t const virtualParticlePositionInCells
                        = virtualParticleWayPassedInCells + constants.virtualParticleInitialStartCell;

                    /* calculate the position of the virtual particle after the current step is calculated */
                    float_64 const nextVirtualParticleWayPassed
                        = constants.deltaWayPerStep * static_cast<float_64>(currentStep + 1);
                    uint32_t const nextVirtualParticleWayPassedInCells
                        = uint32_t(math::floor(nextVirtualParticleWayPassed / constants.cellSizeInMoveDirection));
                    /* This position is used to detect the point in time where the virtual particle
                     * moves over a GPU border.
                     */
                    uint32_t const nextVirtualParticlePositionInCells
                        = nextVirtualParticleWayPassedInCells + constants.virtualParticleInitialStartCell;

                    /* within the to be simulated time step (currentStep -> currentStep+1)
                     * the virtual particle will have reached at least the position
                     * of the cell behind the end of the initial global domain
                     * (also true for all later time steps)
                     */
                    bool const endOfInitialGlobalDomain = constants.firstSlideStep <= currentStep;

                    uint32_t const currentCell
                        = virtualParticlePositionInCells % constants.gpuNumberOfCellsInMoveDirection;
                    uint32_t const nextCell
                        = nextVirtualParticlePositionInCells % constants.gpuNumberOfCellsInMoveDirection;
                    /* virtual particle will pass a GPU border during the current
                     * (to be simulated) time step
                     */
                    bool const virtualParticlePassesGPUBorder = (nextCell < currentCell);

                    if(endOfInitialGlobalDomain && virtualParticlePassesGPUBorder && realCurStep <= endSlidingOnStep)
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
                        *offsetFirstGPU = nextCell;
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
        void incrementSlideCounter(uint32_t const currentStep)
        {
            // do not slide twice in one simulation step
            if(isSlidingWindowActive(currentStep) && lastSlideStep < currentStep)
            {
                slideCounter++;
                lastSlideStep = currentStep;
            }
        }

        /** tracks if endSlideOnStep has been called */
        bool firstCallEndSlideOnStep = true;

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

        // Not RAII because these are only required if moving window is enabled
        std::optional<ComputedConstants> computedConstants;

    public:
        /* defines in which direction the window moves
         *
         * 0 == x,  1 == y , 2 == z direction
         *
         * note: currently only y direction is supported
         */
        static constexpr uint32_t moveDirection = 1;

        /** Set window move point which defines when to start sliding the window
         *
         * See declaration of movePoint for a detailed explanation.
         *
         * @param point ratio of the global window size
         */
        void setMovePoint(float_64 const point)
        {
            movePoint = point;
            computedConstants.reset();
        }

        void resetForTesting()
        {
            slidingWindowEnabled = false;
            movePoint = 0.0;
            slideCounter = 0u;
            lastSlideStep = 0u;
            endSlidingOnStep = 0u;
            computedConstants.reset();
            firstCallEndSlideOnStep = true;
        }

        /**
         * Set step where the simulation stops the moving window
         *
         * @param step 0 means no sliding window, else sliding is enabled until step is reached.
         * Negative numbers are the way to use this if you don't want the sliding to end (practically) -1 sets it to
         * the largest uint32 - 4,294,967,295
         */
        void setEndSlideOnStep(int32_t step)
        {
            // maybe we have a underflow in the cast, this is fine because it results in a very large number
            auto const maxSlideStep = static_cast<uint32_t>(step);
            if(maxSlideStep < lastSlideStep)
                throw std::runtime_error("It is not allowed to stop the moving window in the past.");

            endSlidingOnStep = maxSlideStep;

            /* Disable or enable sliding window only in the first call.
             * Later changes of step will not influence if the sliding window is activated.
             */
            if(firstCallEndSlideOnStep && endSlidingOnStep != 0u)
            {
                slidingWindowEnabled = true;
                firstCallEndSlideOnStep = false;
            }
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
         * If a slide occurs in \p currentStep, it is included in the result.
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
        bool isSlidingWindowActive(uint32_t const currentStep) const
        {
            return isEnabled() && currentStep < endSlidingOnStep;
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
         * Get the moving window origin position in cell, not discretized to the cell grid
         * This position is the total origin (0,0,0) until we get to the first move step
         * On the first move step the position is equal to the moving window move location plus the distance moved with
         * c in one timestep. This ensures that this comoving particle is always inside the region of the moving window
         * After this, the particle continues moving with c, until we reach the step where the window stops
         * moving, Then, the particle stops where ever it is (maybe in the middle of a cell)
         *
         * @param currentStep current simulation step
         * @return moving window origin position relative to total origin in cells
         */
        pmacc::math::Vector<float_64, simDim> getMovingWindowOriginPositionCells(uint32_t currentStep)
        {
            auto offsets = pmacc::math::Vector<float_64, simDim>::create(0);

            if(!slidingWindowEnabled)
                return offsets;

            auto const& constants = getOrComputeConstants();

            SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();

            uint32_t const globalWindowSizeInMoveDirection
                = subGrid.getGlobalDomain().size[moveDirection] - constants.gpuNumberOfCellsInMoveDirection;

            // Clamp currentStep if sliding has ended
            if(currentStep > endSlidingOnStep)
                currentStep = endSlidingOnStep;

            auto cellsPerStep = constants.deltaWayPerStep / constants.cellSizeInMoveDirection;

            auto firstStep = constants.firstMoveStep > 0 ? constants.firstMoveStep : 0;

            if(static_cast<int32_t>(currentStep) >= firstStep)
            {
                offsets[moveDirection] = constants.virtualParticleInitialStartCell
                                         + cellsPerStep * static_cast<float_64>(currentStep + 1)
                                         - globalWindowSizeInMoveDirection;
            }

            return offsets;
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
            SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();

            /* Without moving window, the selected window spans the whole global domain.
             * @see https://github.com/ComputationalRadiationPhysics/picongpu/wiki/PIConGPU-domain-definitions
             *
             * The window's global offset is therefore zero inside the global domain.
             * The window's global and local size are equal to the SubGrid quantities.
             * The local window offset is the offset within the global window which
             * is equal to the local domain offset of the GPU.
             */
            Window window;
            window.localDimensions = subGrid.getLocalDomain();
            window.globalDimensions = Selection<simDim>(subGrid.getGlobalDomain().size);
            auto const& constants = getOrComputeConstants();

            /* moving window can only slide in y direction */
            if(slidingWindowEnabled)
            {
                /* the moving window is smaller than the global domain by exactly one
                 * GPU (local domain size) in moving (y) direction
                 */
                window.globalDimensions.size[moveDirection] -= constants.gpuNumberOfCellsInMoveDirection;

                float_64 offsetFirstGPU = 0.0;
                getCurrentSlideInfo(currentStep, nullptr, &offsetFirstGPU);

                /* while moving, the windows global offset within the global domain is between 0
                 * and smaller than the local domain's size in y.
                 */
                window.globalDimensions.offset[moveDirection] = offsetFirstGPU;

                /* set top/bottom if there are no communication partners
                 * for this GPU in the respective direction */
                Mask const comm_mask = Environment<simDim>::get().GridController().getCommunicationMask();
                bool const isTopGpu = !comm_mask.isSet(TOP);
                bool const isBottomGpu = !comm_mask.isSet(BOTTOM);

                if(isTopGpu)
                {
                    /* the windows local offset within the global window is reduced
                     * by the global window offset within the global domain
                     */
                    window.localDimensions.size[moveDirection] -= offsetFirstGPU;
                }
                else
                {
                    window.localDimensions.offset[moveDirection]
                        = subGrid.getLocalDomain().offset[moveDirection] - offsetFirstGPU;
                    if(isBottomGpu)
                    {
                        window.localDimensions.size[moveDirection] = offsetFirstGPU;
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
            SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();
            Window window;

            window.localDimensions = subGrid.getLocalDomain();
            window.globalDimensions = Selection<simDim>(subGrid.getGlobalDomain().size);

            return window;
        }
    };

} // namespace picongpu
