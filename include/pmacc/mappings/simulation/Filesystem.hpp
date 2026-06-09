/*
 * SPDX-FileCopyrightText: PIConGPU contributors
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

/* Copyright 2014-2024 Felix Schmitt
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
    /**
     * Singleton class providing common filesystem operations.
     */
    class Filesystem
    {
    public:
        /** Create directory with default permissions
         *
         * @attention Only one MPI rank is allowed to call this method.
         *
         * @param dir name of directory
         */
        void createDirectory(std::string const dir) const;
        /** Set 755 permissions for a directory
         *
         * @attention Only one MPI rank is allowed to call this method.
         *
         * @param dir name of directory
         */
        void setDirectoryPermissions(std::string const dir) const;

        /** Create directory and set 755 permissions by root rank.
         *
         * @attention Only one MPI rank is allowed to call this method.
         *
         * @param dir name of directory
         */
        void createDirectoryWithPermissions(std::string const dir) const;

        /** Strip path from absolute or relative paths to filenames
         *
         * @param path and filename
         */
        std::string basename(std::string const pathFilename) const;

        /** Returns the instance of the filesystem class.
         *
         * This class is a singleton class.
         *
         * @return a filesystem instance
         */
        static Filesystem& get()
        {
            static Filesystem instance;
            return instance;
        }

    private:
        /**
         * Constructor
         */
        Filesystem() = default;

        /**
         * Constructor
         */
        Filesystem(Filesystem const& fs) = default;
    };

} // namespace pmacc
