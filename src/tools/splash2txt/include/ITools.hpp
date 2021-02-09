/* Copyright 2013-2021 Felix Schmitt
 *
 * This file is part of splash2txt.
 *
 * splash2txt is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * splash2txt is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with splash2txt.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef ITOOLS_HPP
#define ITOOLS_HPP

#include "splash2txt.hpp"

class ITools
{
public:

    ITools(ProgramOptions &options, Dims &mpiTopology, std::ostream &outStream) :
    m_options(options),
    m_mpiTopology(mpiTopology),
    m_outStream(outStream)
    {

    }

    virtual ~ITools()
    {

    }

    virtual void convertToText() = 0;

    virtual void listAvailableDatasets() = 0;

protected:
    ProgramOptions &m_options;
    Dims &m_mpiTopology;
    std::ostream &m_outStream;
};

#endif    /* ITOOLS_HPP */

