/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/plugins/multi/IInstance.hpp"

#include <memory>
#include <string>

namespace picongpu
{
    //! Interface for IO-backends with restart capability
    class IIOBackend : public plugins::multi::IInstance
    {
    public:
        IIOBackend() = default;

        ~IIOBackend() override = default;

        //! create a checkpoint
        virtual void dumpCheckpoint(
            uint32_t currentStep,
            std::string const& checkpointDirectory,
            std::string const& checkpointFilename)
            = 0;

        //! restart from a checkpoint
        virtual void doRestart(
            uint32_t restartStep,
            std::string const& restartDirectory,
            std::string const& restartFilename,
            uint32_t restartChunkSize)
            = 0;
    };

} // namespace picongpu
