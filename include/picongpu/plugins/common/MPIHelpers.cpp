/*
 * SPDX-FileCopyrightText: Franz Poeschel
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "picongpu/plugins/common/MPIHelpers.hpp"

#include <pmacc/communication/manager_common.hpp>

#include <algorithm>
#include <fstream>
#include <numeric>
#include <sstream>
#include <vector>

namespace picongpu
{
    /**
     * @brief Read a file in MPI-collective manner.
     *
     * The file is read on rank 0 and its contents subsequently distributed
     * to all other ranks.
     *
     * @param path Path for the file to read.
     * @param comm MPI communicator.
     * @return std::string Full file content.
     */
    std::string collective_file_read(std::string const& path, MPI_Comm comm)
    {
        int rank, size;
        MPI_CHECK(MPI_Comm_rank(comm, &rank));
        MPI_CHECK(MPI_Comm_size(comm, &size));

        std::string res;
        size_t stringLength = 0;
        if(rank == 0)
        {
            std::fstream handle;
            handle.open(path, std::ios_base::in);
            std::stringstream stream;
            stream << handle.rdbuf();
            res = stream.str();
            if(!handle.good())
            {
                throw std::runtime_error("Failed reading JSON config from file " + path + ".");
            }
            stringLength = res.size() + 1;
        }
        MPI_Datatype datatype = MPI_Types<size_t>{}.value;
        int err = MPI_Bcast(&stringLength, 1, datatype, 0, comm);
        if(err != MPI_SUCCESS)
        {
            throw std::runtime_error("[collective_file_read] MPI_Bcast stringLength failure.");
        }
        std::vector<char> recvbuf(stringLength, 0);
        if(rank == 0)
        {
            std::copy_n(res.c_str(), stringLength, recvbuf.data());
        }
        err = MPI_Bcast(recvbuf.data(), stringLength, MPI_CHAR, 0, comm);
        if(err != MPI_SUCCESS)
        {
            throw std::runtime_error("[collective_file_read] MPI_Bcast file content failure.");
        }
        if(rank != 0)
        {
            res = recvbuf.data();
        }
        return res;
    }
} // namespace picongpu
