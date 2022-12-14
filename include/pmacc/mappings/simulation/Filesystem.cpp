/* Copyright 2014-2022 Felix Schmitt
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

#include "pmacc/mappings/simulation/Filesystem.hpp"

#include <pmacc/boost_workaround.hpp>

#include "pmacc/Environment.hpp"
#include "pmacc/mappings/simulation/GridController.hpp"

#include <boost/filesystem.hpp>

namespace pmacc
{
    template<unsigned DIM>
    void Filesystem<DIM>::createDirectory(const std::string dir) const
    {
        /* does not throw if the directory exists or has been created */
        bfs::create_directories(dir);
    }

    template<unsigned DIM>
    void Filesystem<DIM>::setDirectoryPermissions(const std::string dir) const
    {
        /* set permissions */
        bfs::permissions(dir, bfs::owner_all | bfs::group_read | bfs::group_exe | bfs::others_read | bfs::others_exe);
    }

    template<unsigned DIM>
    void Filesystem<DIM>::createDirectoryWithPermissions(const std::string dir) const
    {
        GridController<DIM>& gc = Environment<DIM>::get().GridController();

        createDirectory(dir);

        if(gc.getGlobalRank() == 0)
        {
            /* must be set by only one process to avoid races */
            setDirectoryPermissions(dir);
        }
    }


    template<unsigned DIM>
    std::string Filesystem<DIM>::basename(const std::string pathFilename) const
    {
        return bfs::path(pathFilename).filename().string();
    }

    // Explicit template instantiation to provide symbols for usage together with PMacc
    template class Filesystem<DIM2>;
    template class Filesystem<DIM3>;

} // namespace pmacc
