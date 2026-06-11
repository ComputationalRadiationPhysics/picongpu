/*
 * SPDX-FileCopyrightText: Axel Huebl, Richard Pausch
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/logging.hpp"

#include <pmacc/filesystem.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace picongpu
{
    /** Restore a txt file from the checkpoint dir
     *
     * Restores a txt file from the checkpoint dir and starts appending to it.
     * Opened files in @see outFile are closed and a valid handle is opened again
     * if a restart file is found. Otherwise new output file stays untouched.
     *
     * @param outFile std::ofstream file handle to regular file that shall be restored
     * @param filename the file's name
     * @param restartStep the file's version in time to restore
     * @param restartDirectory path to the checkpoint directory
     *
     * @return operation was successful or not
     */
    inline bool restoreTxtFile(
        std::ofstream& outFile,
        std::string filename,
        uint32_t restartStep,
        std::string const restartDirectory)
    {
        /* get restart time step as string */
        std::stringstream sStep;
        sStep << restartStep;

        /* set location of restart file and output file */
        stdfs::path src(restartDirectory + std::string("/") + filename + std::string(".") + sStep.str());
        stdfs::path dst(filename);

        /* check whether restart file exists */
        if(!stdfs::exists(src))
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

            stdfs::copy_file(src, dst, stdfs::copy_options::overwrite_existing);

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
     * @param outFile std::ofstream file handle to regular file that shall be checkpointed
     * @param filename the file's name
     * @param currentStep the current time step
     * @param checkpointDirectory path to the checkpoint directory
     */
    inline void checkpointTxtFile(
        std::ofstream& outFile,
        std::string filename,
        uint32_t currentStep,
        std::string const checkpointDirectory)
    {
        outFile.flush();

        std::stringstream sStep;
        sStep << currentStep;

        stdfs::path src(filename);
        stdfs::path dst(checkpointDirectory + std::string("/") + filename + std::string(".") + sStep.str());

        stdfs::copy_file(src, dst, stdfs::copy_options::overwrite_existing);
    }

} /* namespace picongpu */
