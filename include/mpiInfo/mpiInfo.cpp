/* Copyright 2013-2021  Rene Widera
 *
 * This file is part of mpiInfo.
 *
 * mpiInfo is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * mpiInfo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with mpiInfo.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <mpi.h>
#include <cstdlib>
#include <iostream> // std::cerr

#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/variables_map.hpp>


#define MPI_CHECK(cmd)                                                                                                \
    {                                                                                                                 \
        int error = cmd;                                                                                              \
        if(error != MPI_SUCCESS)                                                                                      \
        {                                                                                                             \
            printf("<%s>:%i ", __FILE__, __LINE__);                                                                   \
            throw std::runtime_error(std::string("[MPI] Error"));                                                     \
        }                                                                                                             \
    }

namespace po = boost::program_options;

enum
{
    gridInitTag = 1,
    gridHostnameTag = 2,
    gridHostRankTag = 3,
    gridExitTag = 4,
    gridExchangeTag = 5
};

/* Set the first found non charactor or number to 0 (nullptr)
 * name like p1223(Pid=1233) is than p1223
 * in some MPI implementation /mpich) the hostname is unique
 */
void cleanHostname(char* name)
{
    for(int i = 0; i < MPI_MAX_PROCESSOR_NAME; ++i)
    {
        if(!(name[i] >= 'A' && name[i] <= 'Z') && !(name[i] >= 'a' && name[i] <= 'z')
           && !(name[i] >= '0' && name[i] <= '9') && !(name[i] == '_') && !(name[i] == '-'))
        {
            name[i] = 0;
            return;
        }
    }
}

/*! gets hostRank
 *
 * process with MPI-rank 0 is the master and builds a map with hostname
 * and number of already known processes on this host.
 * Each rank will provide its hostname via MPISend and gets its HostRank
 * from the master.
 *
 */
int getHostRank()
{
    char hostname[MPI_MAX_PROCESSOR_NAME];
    int length;
    int hostRank;

    int totalnodes;
    int myrank;

    MPI_CHECK(MPI_Get_processor_name(hostname, &length));
    cleanHostname(hostname);
    hostname[length++] = '\0';

    // int totalnodes;

    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &totalnodes));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myrank));

    if(myrank == 0)
    {
        std::map<std::string, int> hosts;
        hosts[hostname] = 0;
        hostRank = 0;
        for(int rank = 1; rank < totalnodes; ++rank)
        {
            MPI_CHECK(MPI_Recv(
                hostname,
                MPI_MAX_PROCESSOR_NAME,
                MPI_CHAR,
                rank,
                gridHostnameTag,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE));

            // printf("Hostname: %s\n", hostname);
            int hostrank = 0;
            if(hosts.count(hostname) > 0)
                hostrank = hosts[hostname] + 1;

            MPI_CHECK(MPI_Send(&hostrank, 1, MPI_INT, rank, gridHostRankTag, MPI_COMM_WORLD));

            hosts[hostname] = hostrank;
        }
    }
    else
    {
        MPI_CHECK(MPI_Send(hostname, length, MPI_CHAR, 0, gridHostnameTag, MPI_COMM_WORLD));

        MPI_CHECK(MPI_Recv(&hostRank, 1, MPI_INT, 0, gridHostRankTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        // if(hostRank!=0) hostRank--; //!\todo fix mpi hostrank start with 1
    }

    return hostRank;
}

int getMyRank()
{
    int myrank;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myrank));
    return myrank;
}

int getTotalRanks()
{
    int totalnodes;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &totalnodes));
    return totalnodes;
}

int main(int argc, char** argv)
{
    bool localRank = false;
    bool myRank = false;
    bool totalRank = false;

    po::options_description desc("Allowed options");
    desc.add_options()(
        "help,h",
        "produce help message")("mpi_host_rank", po::value<bool>(&localRank)->zero_tokens(), "get local mpi rank")(
        "mpi_rank",
        po::value<bool>(&myRank)->zero_tokens(),
        "get mpi rank")("mpi_size", po::value<bool>(&totalRank)->zero_tokens(), "get count of mpi ranks");

    // parse command line options and config file and store values in vm
    po::variables_map vm;
    po::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    // print help message and quit simulation
    if(vm.count("help"))
    {
        std::cerr << desc << "\n";
        return 0;
    }

    MPI_CHECK(MPI_Init(&argc, &argv));
    if(localRank)
        std::cout << "mpi_host_rank: " << getHostRank() << std::endl;
    if(myRank)
        std::cout << "mpi_rank: " << getMyRank() << std::endl;
    if(totalRank)
        std::cout << "mpi_size: " << getTotalRanks() << std::endl;


    MPI_CHECK(MPI_Finalize());

    return 0;
}
