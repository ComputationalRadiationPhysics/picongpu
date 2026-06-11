/*
 * SPDX-FileCopyrightText: Felix Schmitt
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#include "pmacc/mappings/simulation/Filesystem.hpp"

#include "pmacc/Environment.hpp"
#include "pmacc/filesystem.hpp"
#include "pmacc/mappings/simulation/GridController.hpp"

namespace pmacc
{
    void Filesystem::createDirectory(std::string const dir) const
    {
        /* using `create_directories` instead of `create_directory` because the former does not throw if the directory
         * exists or has been created */
        stdfs::create_directories(dir);
    }

    void Filesystem::setDirectoryPermissions(std::string const dir) const
    {
        using namespace stdfs;
        /* set permissions */
        permissions(
            dir,
            perms::owner_all | perms::group_read | perms::group_exec | perms::others_read | perms::others_exec);
    }

    void Filesystem::createDirectoryWithPermissions(std::string const dir) const
    {
        createDirectory(dir);
        /* must be set by only one process to avoid races */
        setDirectoryPermissions(dir);
    }

    std::string Filesystem::basename(std::string const pathFilename) const
    {
        return stdfs::path(pathFilename).filename().string();
    }
} // namespace pmacc
