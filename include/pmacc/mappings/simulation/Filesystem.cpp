/* Copyright 2014-2023 Felix Schmitt
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


#include "pmacc/mappings/simulation/Filesystem.hpp"

#include "pmacc/Environment.hpp"
#include "pmacc/filesystem.hpp"
#include "pmacc/mappings/simulation/GridController.hpp"

namespace pmacc
{
    void Filesystem::createDirectory(const std::string dir) const
    {
        /* using `create_directories` instead of `create_directory` because the former does not throw if the directory
         * exists or has been created */
        stdfs::create_directories(dir);
    }

    void Filesystem::setDirectoryPermissions(const std::string dir) const
    {
        using namespace stdfs;
        /* set permissions */
        permissions(
            dir,
            perms::owner_all | perms::group_read | perms::group_exec | perms::others_read | perms::others_exec);
    }

    void Filesystem::createDirectoryWithPermissions(const std::string dir) const
    {
        auto const mpiRank = Environment<>::get().EnvironmentController().getCommunicator().getRank();
        bool const isRootRank = mpiRank == 0;

        if(isRootRank)
        {
            createDirectory(dir);
            /* must be set by only one process to avoid races */
            setDirectoryPermissions(dir);
        }
    }

    std::string Filesystem::basename(const std::string pathFilename) const
    {
        return stdfs::path(pathFilename).filename().string();
    }
} // namespace pmacc
