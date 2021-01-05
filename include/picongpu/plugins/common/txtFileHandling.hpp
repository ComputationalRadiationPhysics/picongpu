/* Copyright 2015-2021 Axel Huebl, Richard Pausch
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

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

#include <boost/filesystem.hpp>

namespace picongpu
{
    using namespace boost::filesystem;

    /** Restore a txt file from the checkpoint dir
     *
     * Restores a txt file from the checkpoint dir and starts appending to it.
     * Opened files in \see outFile are closed and a valid handle is opened again
     * if a restart file is found. Otherwise new output file stays untouched.
     *
     * \param outFile std::ofstream file handle to regular file that shall be restored
     * \param filename the file's name
     * \param restartStep the file's version in time to restore
     * \param restartDirectory path to the checkpoint directory
     *
     * \return operation was successful or not
     */
    HINLINE bool restoreTxtFile(
        std::ofstream& outFile,
        std::string filename,
        uint32_t restartStep,
        const std::string restartDirectory)
    {
        /* get restart time step as string */
        std::stringstream sStep;
        sStep << restartStep;

        /* set location of restart file and output file */
        path src(restartDirectory + std::string("/") + filename + std::string(".") + sStep.str());
        path dst(filename);

        /* check whether restart file exists */
        if(!boost::filesystem::exists(src))
        {
            /* restart file does not exists */
            log<picLog::INPUT_OUTPUT>("Plugin restart file: %1% was not found. \
                                       --> Starting plugin from current time step.")
                % src;
            return true;
        }
        else
        {
            /* restart file found - fix output file created at restart */
            if(outFile.is_open())
                outFile.close();

            copy_file(src, dst, copy_option::overwrite_if_exists);

            outFile.open(filename.c_str(), std::ofstream::out | std::ostream::app);
            if(!outFile)
            {
                std::cerr << "[Plugin] Can't open file '" << filename << "', output disabled" << std::endl;
                return false;
            }
            return true;
        }
    }

    /** Checkpoints a txt file
     *
     * The file is flushed, copied to the checkpoint dir with extension fileName.step
     *
     * \param outFile std::ofstream file handle to regular file that shall be checkpointed
     * \param filename the file's name
     * \param currentStep the current time step
     * \param checkpointDirectory path to the checkpoint directory
     */
    HINLINE void checkpointTxtFile(
        std::ofstream& outFile,
        std::string filename,
        uint32_t currentStep,
        const std::string checkpointDirectory)
    {
        outFile.flush();

        std::stringstream sStep;
        sStep << currentStep;

        path src(filename);
        path dst(checkpointDirectory + std::string("/") + filename + std::string(".") + sStep.str());

        copy_file(src, dst, copy_option::overwrite_if_exists);
    }

} /* namespace picongpu */
