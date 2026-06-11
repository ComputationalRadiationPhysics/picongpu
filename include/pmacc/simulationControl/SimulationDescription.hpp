/*
 * SPDX-FileCopyrightText: Axel Huebl
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/Environment.def"
#include "pmacc/types.hpp"

#include <string>

namespace pmacc
{
    namespace simulationControl
    {
        /**
         * Provides convenience methods for querying general simulation information.
         * Singleton class.
         */
        class SimulationDescription
        {
        public:
            /** Return author of the simulation setup.
             *
             * The author that runs the simulation and is responsible for created
             * output files.
             *
             * @return std::string with author name, can be empty
             */
            std::string getAuthor()
            {
                return author;
            }

            /** Set author
             *
             * @see getAuthor
             *
             * @param[in] std::string setAuthor
             */
            void setAuthor(std::string const setAuthor)
            {
                this->author = setAuthor;
            }

            /** Return last time step of simulation
             *
             * @return uint32_t last step of the simulation to run to
             */
            uint32_t getRunSteps()
            {
                return runSteps;
            }

            /** Set last time step of simulation
             *
             * @see getRunSteps
             *
             * @param[in] uint32_t setRunSteps
             */
            void setRunSteps(uint32_t const setRunSteps)
            {
                runSteps = setRunSteps;
            }

            /** Returns the current time step of the simulation
             *
             * @return uint32_t current time step
             */
            uint32_t getCurrentStep()
            {
                return currentStep;
            }

            /** Set the current time step
             *
             * @see getCurrentStep
             *
             * @param[in] uint32_t setCurrentStep
             */
            void setCurrentStep(uint32_t const setCurrentStep)
            {
                currentStep = setCurrentStep;
            }

        protected:
            /** author that runs the simulation */
            std::string author;

            /** maximum step to run this simulation to */
            uint32_t runSteps{0};

            /** current time step of simulation */
            uint32_t currentStep{0};

        private:
            friend struct detail::Environment;

            static SimulationDescription& getInstance()
            {
                static SimulationDescription instance;
                return instance;
            }

            SimulationDescription() : author("")
            {
            }
        };

    } // namespace simulationControl
} // namespace pmacc
