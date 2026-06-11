/*
 * SPDX-FileCopyrightText: Erik Zenker
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

// pmacc
#include "pmacc/Environment.hpp"
#include "pmacc/dataManagement/DataConnector.hpp"
#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/mappings/simulation/ResourceMonitor.hpp"
#include "pmacc/meta/ForEach.hpp"
#include "pmacc/particles/operations/CountParticles.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    template<typename T_DIM, typename T_Species>
    struct MyCountParticles
    {
        template<typename T_Vector, typename T_MappingDesc, typename T_ParticleFilter>
        void operator()(T_Vector& particleCounts, T_MappingDesc& cellDescription, T_ParticleFilter& parFilter)
        {
            DataConnector& dc = Environment<>::get().DataConnector();

            SubGrid<T_DIM::value> const& subGrid = Environment<T_DIM::value>::get().SubGrid();
            DataSpace<T_DIM::value> const localSize(subGrid.getLocalDomain().size);

            uint64_cu totalNumParticles = 0;
            totalNumParticles = pmacc::CountParticles::countOnDevice<CORE + BORDER>(
                *dc.get<T_Species>(T_Species::FrameType::getName()),
                cellDescription,
                DataSpace<T_DIM::value>(),
                localSize,
                parFilter);
            particleCounts.push_back(totalNumParticles);
        }
    };

    template<unsigned T_DIM>
    ResourceMonitor<T_DIM>::ResourceMonitor() = default;

    template<unsigned T_DIM>
    size_t ResourceMonitor<T_DIM>::getCellCount()
    {
        return Environment<T_DIM>::get().SubGrid().getLocalDomain().size.productOfComponents();
    }

    template<unsigned T_DIM>
    template<typename T_Species, typename T_MappingDesc, typename T_ParticleFilter>
    std::vector<size_t> ResourceMonitor<T_DIM>::getParticleCounts(
        T_MappingDesc& cellDescription,
        T_ParticleFilter& parFilter)
    {
        using dim = std::integral_constant<unsigned int, T_DIM>;
        std::vector<size_t> particleCounts;
        meta::ForEach<T_Species, MyCountParticles<dim, boost::mpl::_1>> countParticles;
        countParticles(particleCounts, cellDescription, parFilter);
        return particleCounts;
    }

} // namespace pmacc
