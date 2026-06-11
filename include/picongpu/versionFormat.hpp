/*
 * SPDX-FileCopyrightText: Axel Huebl
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#pragma once

#include "picongpu/version.hpp"

#include <list>
#include <ostream>
#include <string>

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
