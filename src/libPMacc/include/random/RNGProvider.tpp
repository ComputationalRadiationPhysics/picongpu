/**
 * Copyright 2015-2016 Alexander Grund
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "random/RNGProvider.hpp"
#include "dimensions/DataSpaceOperations.hpp"

namespace PMacc
{
namespace random
{

    namespace kernel {

        template<class T_RNGMethod, class T_RNGBox, class T_Space>
        __global__ void
        initRNGProvider(T_RNGBox rngBox, uint32_t seed, const T_Space size)
        {
            const uint32_t linearTid = blockIdx.x * blockDim.x + threadIdx.x;
            if(linearTid >= size.productOfComponents())
                return;

            const T_Space cellIdx = DataSpaceOperations<T_Space::dim>::map(size, linearTid);
            T_RNGMethod().init(rngBox(cellIdx), seed, linearTid);
        }

    }  // namespace kernel

    template<uint32_t T_dim, class T_RNGMethod>
    RNGProvider<T_dim, T_RNGMethod>::RNGProvider(const Space& size, const std::string& uniqueId):
    		m_size(size), m_uniqueId(uniqueId.empty() ? getName() : uniqueId),
    		buffer(new Buffer(size))
    {
        if(m_size.productOfComponents() == 0)
            throw std::invalid_argument("Cannot create RNGProvider with zero size");
    }

    template<uint32_t T_dim, class T_RNGMethod>
    void RNGProvider<T_dim, T_RNGMethod>::init(uint32_t seed)
    {

        const uint32_t blockSize = 256;
        const uint32_t gridSize = (m_size.productOfComponents() + blockSize - 1u) / blockSize; // Round up

        PMACC_AUTO(bufferBox, buffer->getDeviceBuffer().getDataBox());

        __cudaKernel(kernel::initRNGProvider<RNGMethod>)
        (gridSize, blockSize)
        (bufferBox, seed, m_size);

        Environment<dim>::get().DataConnector().registerData(*this);
    }

    template<uint32_t T_dim, class T_RNGMethod>
    typename RNGProvider<T_dim, T_RNGMethod>::Handle
    RNGProvider<T_dim, T_RNGMethod>::createHandle(const std::string& id)
    {
        RNGProvider& provider = Environment<>::get().DataConnector().getData<RNGProvider>(id, true);
        Handle result(provider.getDeviceDataBox());
        Environment<>::get().DataConnector().releaseData(id);
        return result;
    }

    template<uint32_t T_dim, class T_RNGMethod>
    template<class T_Distribution>
    typename RNGProvider<T_dim, T_RNGMethod>::template GetRandomType<T_Distribution>::type
    RNGProvider<T_dim, T_RNGMethod>::createRandom(const std::string& id)
    {
        typedef typename GetRandomType<T_Distribution>::type ResultType;
        return ResultType(createHandle());
    }

    template<uint32_t T_dim, class T_RNGMethod>
    RNGProvider<T_dim, T_RNGMethod>::DataBoxType
    RNGProvider<T_dim, T_RNGMethod>::getDeviceDataBox()
    {
        return buffer->getDeviceBuffer().getDataBox();
    }

    template<uint32_t T_dim, class T_RNGMethod>
    std::string
    RNGProvider<T_dim, T_RNGMethod>::getName()
    {
        /* generate a unique name (for this type!) to use as a default ID */
        return std::string("RNGProvider")
                + boost::lexical_cast<std::string>(dim+0) /* +0 to create a rvalue of the class const(expr) */
                + RNGMethod::getName();
    }

    template<uint32_t T_dim, class T_RNGMethod>
    SimulationDataId
    RNGProvider<T_dim, T_RNGMethod>::getUniqueId()
    {
        return m_uniqueId;
    }

    template<uint32_t T_dim, class T_RNGMethod>
    void
    RNGProvider<T_dim, T_RNGMethod>::synchronize()
    {
        buffer->deviceToHost();
    }

}  // namespace random
}  // namespace PMacc
