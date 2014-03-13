/**
 * Copyright 2014 Felix Schmitt
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

#include <iostream>
#include <string>
#include <vector>
#include <pngwriter.h>
#include <splash/splash.h>
#include <mpi.h>
#include <boost/program_options.hpp>

using namespace splash;
namespace po = boost::program_options;

typedef struct
{
    std::string filename;
    std::string densityFilename;
    std::string densityDataset;
    int iteration;
    Dimensions dataSize;
} Options;

bool parseCmdLine(int argc, char **argv, Options &options)
{
    try
    {
        std::vector<size_t> sizes;
        options.filename = "";
        options.densityFilename = "gas";
        options.densityDataset = "fields/Density_e";
        options.iteration = 0;

        std::stringstream desc_stream;
        desc_stream << "Usage " << argv[0] << " <png-file> -g width height depth [options]" << std::endl;

        // add possible options
        po::options_description desc(desc_stream.str());
        desc.add_options()
                ("help,h", "print help message")
                ("grid,g", po::value<std::vector<size_t> > (&sizes)->multitoken(), "3D Grid dimensions")
                ("png", po::value<std::string > (&options.filename), "Input PNG file")
                ("iteration,i", po::value<int > (&options.iteration)->default_value(options.iteration),
                "Iteration (timestep) for density data")
                ("output,o", po::value<std::string > (&options.densityFilename)->default_value(options.densityFilename),
                "Output filename (basepart)")
                ("dataset,d", po::value<std::string > (&options.densityDataset)->default_value(options.densityDataset),
                "Fully qualified density HDF5 dataset name")
                ;

        po::positional_options_description pos_options_descr;
        pos_options_descr.add("png", 1);

        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(desc).positional(pos_options_descr).run(), vm);
        po::notify(vm);

        // print help message and return 
        if (vm.count("help"))
        {
            std::cout << desc << std::endl;
            return false;
        }

        if (vm.count("png") != 1)
        {
            std::cerr << "Error: Please specify exactly one input PNG file." << std::endl;
            std::cerr << std::endl << desc << std::endl;
            return false;
        }

        if (!vm.count("grid") || sizes.size() != 3)
        {
            std::cerr << "Error: Please specify 3D dimensions." << std::endl;
            std::cerr << std::endl << desc << std::endl;
            return false;
        }

        options.dataSize.set(sizes[0], sizes[1], sizes[2]);

    } catch (boost::program_options::error& e)
    {
        std::cerr << e.what() << std::endl;
        return false;
    }

    return true;
}

int main(int argc, char **argv)
{
    Options options;
    if (!parseCmdLine(argc, argv, options))
        return -1;

    MPI_Init(NULL, NULL);

    Dimensions data_size(options.dataSize);
    std::cout << "Creating density data with size " << data_size.toString() << std::endl;

    float *data = new float[data_size.getScalarSize()];
    memset(data, 0, data_size.getScalarSize() * sizeof (float));

    std::cout << " Reading PNG file '" << options.filename << "'" << std::endl;
    pngwriter image(data_size[1], data_size[0], 0, (options.filename + std::string(".tmp")).c_str());
    image.readfromfile(options.filename.c_str());

    if (image.getwidth() != (int)data_size[1] || image.getheight() != (int)data_size[0])
    {
        image.close();
        std::cerr << "Invalid image size (" << image.getwidth() << "," <<
                image.getheight() << ") for data size" << std::endl;
        MPI_Finalize();
        return -1;
    }

    for (size_t x = 0; x < data_size[0]; ++x)
        for (size_t y = 0; y < data_size[1]; ++y)
        {
            /* pngwriter coordinates start at (1,1) and the y direction is inverted */
            double px_value = image.dreadHSV(1 + y, 1 + (data_size[0] - x - 1), 3);
            float color = 1.0 - px_value;

            image.plot(1 + y, 1 + (data_size[0] - x - 1), color, color, color);

            for (size_t z = 0; z < data_size[2]; ++z)
            {
                size_t index = z * (data_size[0] * data_size[1]) + y * data_size[0] + x;
                data[index] = color;
            }
        }

    /* write and close the output png */
    image.close();

    std::cout << " Creating density HDF5 file '" << options.densityFilename << "_" <<
            options.iteration << ".h5'" << std::endl;

    /* write density information to HDF5 */
    ParallelDataCollector *pdc = new
            ParallelDataCollector(MPI_COMM_WORLD, MPI_INFO_NULL, Dimensions(1, 1, 1), 1);
    DataCollector::FileCreationAttr attr;
    DataCollector::initFileCreationAttr(attr);
    pdc->open(options.densityFilename.c_str(), attr);

    ColTypeFloat ctFloat;
    pdc->write(options.iteration, ctFloat, data_size.getDims(),
            data_size, options.densityDataset.c_str(), data);

    pdc->close();
    delete pdc;

    delete[] data;

    MPI_Finalize();

    return 0;
}
