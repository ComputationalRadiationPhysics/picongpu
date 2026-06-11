/*
 * SPDX-FileCopyrightText: Felix Schmitt
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
