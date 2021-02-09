/* Copyright 2015-2021 Axel Huebl
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

#include "picongpu/version.hpp"

#include <ostream>
#include <string>
#include <list>


namespace picongpu
{
    /** Collect software dependencies of PIConGPU
     *
     * Collect the versions of dependent software in PIConGPU
     * for output and reproducibility.
     *
     * @param[out] cliText formatted table for output to a command line
     * @return a list of strings in the form software/version
     */
    std::list<std::string> getSoftwareVersions(std::ostream& cliText);
} // namespace picongpu
