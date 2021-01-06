/* Copyright 2015-2021 Axel Huebl
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
            void setAuthor(const std::string setAuthor)
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
            void setRunSteps(const uint32_t setRunSteps)
            {
                runSteps = setRunSteps;
            }

            /** Returns the current time step of the simulation
             *
             * \return uint32_t current time step
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
            void setCurrentStep(const uint32_t setCurrentStep)
            {
                currentStep = setCurrentStep;
            }

        protected:
            /** author that runs the simulation */
            std::string author;

            /** maximum step to run this simulation to */
            uint32_t runSteps;

            /** current time step of simulation */
            uint32_t currentStep;

        private:
            friend struct detail::Environment;

            static SimulationDescription& getInstance()
            {
                static SimulationDescription instance;
                return instance;
            }

            SimulationDescription() : author(""), runSteps(0), currentStep(0)
            {
            }
        };

    } // namespace simulationControl
} // namespace pmacc
