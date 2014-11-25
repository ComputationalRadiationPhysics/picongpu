/**
 * Copyright 2013 Axel Huebl, Heiko Burau, Rene Widera
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



#ifndef DENSITYTOBINARY_HPP
#define	DENSITYTOBINARY_HPP

#include "types.h"
#include "simulation_defines.hpp"
#include "simulation_types.hpp"

#include <string>
#include <mpi.h>
#include "mappings/simulation/GridController.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>

//c includes
#include "sys/stat.h"

namespace picongpu
{
using namespace PMacc;

struct DensityToBinary
{
    typedef float_64 ValueType;

    DensityToBinary(std::string name, std::string folder) : name(folder + "/" + name), folder(folder), createFolder(true)
    {
    }

    ~DensityToBinary()
    {
    }

    template<class Box>
    void operator()(
                    const Box data,
                    const ValueType unit,
                    const Size2D size,
                    const MessageHeader & header)
    {

        if (createFolder)
        {
            Environment<simDim>::get().Filesystem().createDirectoryWithPermissions(folder);
            createFolder = false;
        }

        std::stringstream step;
        step << std::setw(6) << std::setfill('0') << header.sim.step;
        //std::string filename(name + "_" + step.str() + ".bin");
        std::string filename(name + "_" + step.str() + ".dat");

        double x_cell = header.sim.cellSizeArr[0];
        double y_cell = header.sim.cellSizeArr[1];

        double x_simOff = header.sim.simOffsetToNull[0];
        double y_simOff = header.sim.simOffsetToNull[1];

        DataSpace<DIM2> gOffset = header.window.offset;

        std::ofstream file(filename.c_str(), std::ofstream::out); //| std::ofstream::binary);

        typedef std::numeric_limits< ValueType > dbl;
        file.precision(dbl::digits10);
        file << std::scientific;

        ValueType sizex = (int) size.x();
        //file.write((char*) (&sizex), sizeof (ValueType));
        file << sizex << " ";

        //first line with y header information
        for (int x = 0; x < size.x(); ++x)
        {
            ValueType cellPos = (ValueType) ((x + x_simOff + gOffset.x()) * x_cell * UNIT_LENGTH);
            //file.write((char*) &(cellPos), sizeof (ValueType));
            file << cellPos << " ";
        }
        file << std::endl;

        //the first column is for x header information
        for (int y = 0; y < size.y(); ++y)
        {
            const ValueType cellPos = (ValueType) ((y + y_simOff + gOffset.y()) * y_cell * UNIT_LENGTH);
            file << cellPos;
            for (int x = 0; x < size.x(); ++x)
            {
                const ValueType value = precisionCast<ValueType>(data[y][x]) * unit;

                /** \info take care, that gnuplots binary matrix does
                 *        not support float64 (IEEE float32 only)
                 *  \see http://stackoverflow.com/questions/8751154/looking-at-binary-output-from-fortran-on-gnuplot
                 *       http://gnuplot.sourceforge.net/docs_4.2/node101.html
                 */
                //file.write((char*) &(value), sizeof (ValueType));
                file << " " << value;
            }
            file << std::endl;
        }

        file.close();
    }

private:

    std::string name;
    std::string folder;
    bool createFolder;

};

}//namespace

#endif	/* DENSITYTOBINARY_HPP */

