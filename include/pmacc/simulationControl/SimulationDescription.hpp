/* Copyright 2015-2024 Axel Huebl
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
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

            /** Whether checkpoint creation has been configured by the user
             *
             * This allows plugins (e.g. the checkpoint IO-backends) to skip
             * initialization when checkpointing is not requested.
             */
            bool isCheckpointingConfigured() const
            {
                return checkpointingConfigured;
            }

            /** Set whether checkpoint creation has been configured
             *
             * @see isCheckpointingConfigured
             */
            void setCheckpointingConfigured(bool const value)
            {
                checkpointingConfigured = value;
            }

            /** Return the common directory for checkpoints
             *
             * @return std::string checkpoint directory
             */
            std::string const& getCheckpointDirectory() const
            {
                return checkpointDirectory;
            }

            /** Set the common directory for checkpoints
             *
             * @see getCheckpointDirectory
             */
            void setCheckpointDirectory(std::string const& value)
            {
                checkpointDirectory = value;
            }

            /** Return the common directory for restarts
             *
             * @return std::string restart directory
             */
            std::string const& getRestartDirectory() const
            {
                return restartDirectory;
            }

            /** Set the common directory for restarts
             *
             * @see getRestartDirectory
             */
            void setRestartDirectory(std::string const& value)
            {
                restartDirectory = value;
            }

            /** Whether a restart from a checkpoint has been requested
             *
             * This allows plugins (e.g. the checkpoint IO-backends) to skip
             * initialization when no restart is requested.
             */
            bool isRestartConfigured() const
            {
                return restartConfigured;
            }

            /** Set whether a restart from a checkpoint has been requested
             *
             * @see isRestartConfigured
             */
            void setRestartConfigured(bool const value)
            {
                restartConfigured = value;
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

            /** whether checkpoint creation has been configured */
            bool checkpointingConfigured{false};

            /** whether a restart from a checkpoint has been requested */
            bool restartConfigured{false};

            /** common directory for checkpoints */
            std::string checkpointDirectory{"checkpoints"};

            /** common directory for restarts */
            std::string restartDirectory{"checkpoints"};

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
