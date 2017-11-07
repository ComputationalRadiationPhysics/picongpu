/* Copyright 2013-2017 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Alexander Grund
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/plugins/ILightweightPlugin.hpp"

#include <string>
#include <memory>

namespace picongpu
{

//! Interface for IO-backends with restart capability
class IIOBackend : public ILightweightPlugin
{
public:


    IIOBackend()
    {

    }

    template< typename T_IOBackend >
    static std::shared_ptr<IIOBackend> create( bool const isIndependent = true )
    {
        return std::shared_ptr<IIOBackend>( new T_IOBackend( isIndependent ) );
    }

    virtual ~IIOBackend()
    {

    }

    //! create a checkpoint
    virtual void dumpCheckpoint(
        uint32_t currentStep,
        std::string const & checkpointDirectory,
        std::string const & checkpointFilename
    ) = 0;

    //! restart from a checkpoint
    virtual void doRestart(
        uint32_t restartStep,
        std::string const & restartDirectory,
        std::string const & restartFilename,
        uint32_t restartChunkSize
    ) = 0;

    virtual void expandHelp(
        std::string const & prefix,
        boost::program_options::options_description & desc
    ) = 0;

};

} //namespace picongpu
