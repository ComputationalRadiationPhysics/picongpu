/* Copyright 2020-2024 Pawel Ordyna
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

// PMacc Includes
#include <pmacc/lockstep.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/math/Vector.hpp>
#include <pmacc/math/operation.hpp>
#include <pmacc/memory/shared/Allocate.hpp>
#include <pmacc/mpi/MPIReduce.hpp>
#include <pmacc/mpi/reduceMethods/Reduce.hpp>
#include <pmacc/particles/algorithm/ForEach.hpp>
#include <pmacc/random/RNGProvider.hpp>
#include <pmacc/random/distributions/Uniform.hpp>

// Standard Library Includes
#include <cmath>
#include <cstddef>
#include <limits>

#pragma once

namespace picongpu::particles::fusion::detail
{
    using namespace pmacc;

    template<typename T_worker, typename T_arr>
    DINLINE void zeroArray(T_worker const& worker, T_arr* arr, uint32_t const& size)
    {
        for(int i = worker.workerIdx(); i < size; i += worker.numWorkers())
        {
            arr[i] = 0;
        }
        worker.sync();
    }

    template<bool debug = false, typename T_worker, typename T_arr>
    DINLINE void maxArrayDestroy(T_worker const& worker, T_arr& arr, int const& size)
    {
        uint32_t pow = 1;
        while(pow < size)
        {
            for(uint32_t i = worker.workerIdx(); pow * (2 * i + 1) < size; i += 2 * pow * worker.numWorkers())
            {
                arr[2 * i * pow] = std::max(arr[2 * i * pow], arr[pow * (2 * i + 1)]);
            }
            pow <<= 1; //*2
            worker.sync();
            if constexpr(debug)
            {
                if(worker.workerIdx() == 0)
                    printArray(arr);
                worker.sync();
            }
        }
        // max is now at arr[0];
    }

    template<std::size_t... Is, std::size_t N>
    DINLINE void printArrayImpl(memory::Array<uint32_t, N>& arr, std::index_sequence<Is...>)
    {
        printf("array: ");
        ((printf("%u, ", arr[Is])), ...);
        printf("\n");
    }

    template<std::size_t N>
    DINLINE void printArray(memory::Array<uint32_t, N>& arr)
    {
        printArrayImpl(arr, std::make_index_sequence<N>{});
    }

} // namespace picongpu::particles::fusion::detail
