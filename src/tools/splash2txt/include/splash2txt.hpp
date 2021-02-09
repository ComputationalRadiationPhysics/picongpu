/* Copyright 2013-2021 Felix Schmitt, Axel Huebl, Rene Widera
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

#ifndef SPLASH2TXT_HPP
#define SPLASH2TXT_HPP

#include <mpi.h>

#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <stdint.h>

#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/variables_map.hpp>
#include <iomanip>

class Dims
{
public:

    Dims()
    {
        set(1, 1, 1);
    }

    Dims(size_t x, size_t y, size_t z)
    {
        set(x, y, z);
    }

    void set(size_t x, size_t y, size_t z)
    {
        s[0] = x;
        s[1] = y;
        s[2] = z;
    }

    size_t *getPointer()
    {
        return s;
    }

    size_t & operator[](const size_t t) {
        return s[t];
    }

    const size_t & operator[](const size_t t) const {
        return s[t];
    }

private:
    size_t s[3];
};

enum FileMode { FM_SPLASH = 0
#if (ENABLE_ADIOS == 1)
               ,FM_ADIOS = 1
#endif
};

typedef struct
{
    FileMode fileMode; // type of input file
    std::string inputFile; // input file, common part
    std::string outputFile; // output file
    bool toFile; // use output file
    std::string delimiter;
    uint32_t step; // simulation iteration
    std::vector<std::string> data; // names of datasets
    Dims fieldDims; // for field data, dimensions of slice, e.g. xy -> (1, 1, 0)
    size_t sliceOffset; // offset of slice (fieldDims) in dataset
    bool isReverseSlice; // if one of allowedReverseSlices is used
    bool verbose; // verbose output on stdout
    bool listDatasets; // list available datasets
    bool applyUnits; // apply the unit stored in HDF5 to the output data
} ProgramOptions;

#endif    /* SPLASH2TXT_HPP */

