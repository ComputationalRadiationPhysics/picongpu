/**
 * Copyright 2016 Erik Zenker
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

// PMacc
#include "Environment.hpp"
#include <particles/operations/CountParticles.hpp>
#include "pmacc_types.hpp"
#include "forward.hpp"
#include "dimensions/DataSpace.hpp"
#include "algorithms/ForEach.hpp"
#include "dataManagement/DataConnector.hpp"
#include "mappings/simulation/ResourceMonitor.hpp"

namespace PMacc
{
    template<typename T_DIM, typename T_Species>
    struct MyCountParticles
    {
        template <typename T_Vector, typename T_MappingDesc>
        void operator()(T_Vector & particleCounts, T_MappingDesc & cellDescription)
        {
            DataConnector & dc = Environment<>::get().DataConnector();

            const SubGrid<T_DIM::value> & subGrid = Environment<T_DIM::value>::get().SubGrid();
            const DataSpace<T_DIM::value> localSize(subGrid.getLocalDomain().size);

            uint64_cu totalNumParticles = 0;
            totalNumParticles = PMacc::CountParticles::countOnDevice < CORE + BORDER > (
                    dc.getData<T_Species >(T_Species::FrameType::getName(), true),
                    cellDescription,
                    DataSpace<T_DIM::value>(),
                    localSize);
            particleCounts.push_back(totalNumParticles);
        }
    };

    template<unsigned T_DIM>
    ResourceMonitor<T_DIM>::ResourceMonitor()
    {

    }

    template<unsigned T_DIM>
    size_t ResourceMonitor<T_DIM>::getCellCount()
    {
        return Environment<T_DIM>::get().SubGrid().getLocalDomain().size.productOfComponents();
    }

    template<unsigned T_DIM>
    template <typename T_Species, typename T_MappingDesc>
    std::vector<size_t> ResourceMonitor<T_DIM>::getParticleCounts(T_MappingDesc &cellDescription)
    {
        typedef bmpl::integral_c<unsigned, T_DIM> dim;
        std::vector<size_t> particleCounts;
        algorithms::forEach::ForEach<T_Species, MyCountParticles<dim, bmpl::_1> > countParticles;
        countParticles(forward(particleCounts), forward(cellDescription));
        return particleCounts;
    }

} //namespace PMacc
