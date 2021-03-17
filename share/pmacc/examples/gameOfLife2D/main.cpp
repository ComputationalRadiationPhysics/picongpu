/* Copyright 2013-2021 Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
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

#include "types.hpp"
#include <pmacc/Environment.hpp>
#include "Simulation.hpp"

#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/variables_map.hpp>

#include <iostream>


namespace po = boost::program_options;

/*! start of PIConGPU
 *
 * @param argc count of arguments in argv
 * @param argv arguments of program start
 */
int main(int argc, char** argv)
{
    typedef ::gol::Space Space;

    std::vector<uint32_t> devices; /* will be set by boost program argument option "-d 3 3" */
    std::vector<uint32_t> gridSize; /* same but with -g */
    std::vector<uint32_t> periodic;
    uint32_t steps;
    std::string rule; /* Game of Life Simulation Rules like 23/3 */

    po::options_description desc("Allowed options");
    desc.add_options()(
        "help,h",
        "produce help message")("steps,s", po::value<uint32_t>(&steps)->default_value(100), "simulation steps")(
        "rule,r",
        po::value<std::string>(&rule)->default_value("23/3"),
        "simulation rule as stay_alive/born")(
        "devices,d",
        po::value<std::vector<uint32_t>>(&devices)->multitoken(),
        "number of devices in each dimension (only 1D or 2D). If you use more than "
        "one device in total, you will need to run mpirun with \"mpirun -n "
        "<DeviceCount.x*DeviceCount.y> ./gameOfLife\"")(
        "grid,g",
        po::value<std::vector<uint32_t>>(&gridSize)->multitoken(),
        "size of the simulation grid (must be 2D, e.g.: -g 128 128). Because of the border, which is one supercell = "
        "16 cells wide, "
        "the size in each direction should be greater or equal than 3*16=48 per device, so that the core will be "
        "non-empty")(
        "periodic,p",
        po::value<std::vector<uint32_t>>(&periodic)->multitoken(),
        "specifying whether the grid is periodic (1) or not (0) in each dimension, default: no periodic dimensions");

    /* parse command line options and config file and store values in vm */
    po::variables_map vm;
    po::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    /* print help message and quit simulation */
    if(vm.count("help"))
    {
        std::cerr << desc << "\n";
        return false;
    }


    /* fill periodic with 0 */
    while(periodic.size() < DIM2)
        periodic.push_back(0);

    /* check on correct number of devices. fill with default value 1 for missing dimensions */
    if(devices.size() > DIM2)
    {
        std::cerr << "Invalid number of devices." << std::endl;
        std::cerr << "use [-d dx dy] with dx, dy being number of devices in each dimension" << std::endl;
    }
    else
        while(devices.size() < DIM2)
            devices.push_back(1);

    /* check on correct grid size. fill with default grid size value 1 for missing 3. dimension */
    if(gridSize.size() != DIM2)
    {
        std::cerr << "Invalid or missing grid size.\nuse -g width height" << std::endl;
        MPI_CHECK(MPI_Finalize());
        return 0;
    }


    /* after checking all input values, copy into DataSpace Datatype */
    Space gpus(devices[0], devices[1]);
    Space grid(gridSize[0], gridSize[1]);
    Space endless(periodic[0], periodic[1]);

    /* extract stay alive and born rule from rule string */
    uint32_t ruleMask = 0;
    size_t strLen = rule.length();
    size_t gPoint = rule.find('/');
    std::string stayAliveIf = rule.substr(0, gPoint);
    std::string newBornIf = rule.substr(gPoint + 1, strLen - gPoint - 1);


    for(unsigned int i = 0; i < newBornIf.length(); ++i)
    {
        std::stringstream ss; /* used for converting const char* "123" to int 123 */
        ss << newBornIf[i]; /* extract every integer separately */
        int shift;
        ss >> shift;
        /* ruleMask has 32 bits - bit 10 to 18 are reserved for born encoding
           10th bit: born if 1 neighbor exists, 11th bit: born if 2 neighbors exist, ... */
        ruleMask = ruleMask | 1 << (shift + 9);
    }
    for(unsigned int i = 0; i < stayAliveIf.length(); ++i)
    {
        std::stringstream ss;
        ss << stayAliveIf[i]; /* extract every integer separately */
        int shift;
        ss >> shift;
        /* ruleMask has 32 bits - bit 1 to 9 are reserved for stay alive encoding
           1st bit: stay alive if 1 neighbor exists, 2nd bit: stay alive if 2 neighbors exist, ... */
        ruleMask = ruleMask | 1 << (shift);
    }
    std::cout << "newborn if=" << newBornIf << " stay alive if=" << stayAliveIf << " mask=" << ruleMask << std::endl;

    /* start game of life simulation */
    gol::Simulation sim(ruleMask, steps, grid, gpus, endless);
    sim.init();
    sim.start();
    sim.finalize();

    /* finalize the pmacc context */
    pmacc::Environment<>::get().finalize();

    return 0;
}
