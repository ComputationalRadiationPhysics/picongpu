/* Copyright 2021-2023 Franz Poeschel
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

#pragma once

#include "picongpu/simulation_defines.hpp"

#include <string>
#include <vector>

#include <openPMD/openPMD.hpp>

namespace picongpu
{
    namespace openPMD
    {
        class SetMeshAttributes
        {
        private:
            static std::vector<float_64> initGridGlobalOffset(uint32_t currentStep);
            static std::vector<float_X> initGridSpacing();
            static std::vector<std::string> initAxisLabels();

        public:
            static constexpr ::openPMD::UnitDimension openPMDUnitDimensions[7]
                = {::openPMD::UnitDimension::L,
                   ::openPMD::UnitDimension::M,
                   ::openPMD::UnitDimension::T,
                   ::openPMD::UnitDimension::I,
                   ::openPMD::UnitDimension::theta,
                   ::openPMD::UnitDimension::N,
                   ::openPMD::UnitDimension::J};

            // Attributes per Mesh
            std::vector<std::string> m_axisLabels = initAxisLabels();
            ::openPMD::Mesh::DataOrder m_dataOrder = ::openPMD::Mesh::DataOrder::C;
            ::openPMD::Mesh::Geometry m_geometry = ::openPMD::Mesh::Geometry::cartesian;
            std::vector<float_64> m_gridGlobalOffset; // no default since it depends on time step
            std::vector<float_X> m_gridSpacing = initGridSpacing();
            double m_gridUnitSI = UNIT_LENGTH;
            float_X m_timeOffset = 0.;
            std::map<::openPMD::UnitDimension, double> m_unitDimension{};

            // Attributes per Mesh Component
            // my compiler thinks I want to define a function when I use the shorthand
            std::vector<float_X> m_position = std::vector<float_X>(simDim, 0.5);
            double m_unitSI = 1.0;

            SetMeshAttributes(uint32_t currentStep);

            SetMeshAttributes const& operator()(::openPMD::Mesh& mesh) const;

            SetMeshAttributes const& operator()(::openPMD::MeshRecordComponent& dataset) const;
        };


        /******************
         * IMPLEMENTATION *
         ******************/

        inline std::vector<std::string> SetMeshAttributes::initAxisLabels()
        {
            using vec_t = std::vector<std::string>;
            return simDim == DIM2 ? vec_t{"y", "x"} : vec_t{"z", "y", "x"};
        }

        inline std::vector<float_X> SetMeshAttributes::initGridSpacing()
        {
            std::vector<float_X> gridSpacing(simDim, 0.0);
            for(uint32_t d = 0; d < simDim; ++d)
            {
                gridSpacing.at(simDim - 1 - d) = cellSize[d];
            }
            return gridSpacing;
        }


        inline std::vector<float_64> SetMeshAttributes::initGridGlobalOffset(uint32_t currentStep)
        {
            std::vector<float_64> gridGlobalOffset;
            /* globalSlideOffset due to gpu slides between origin at time step 0
             * and origin at current time step
             * ATTENTION: openPMD offset are globalSlideOffset + picongpu offsets
             */
            auto movingWindow = MovingWindow::getInstance().getWindow(currentStep);
            gridGlobalOffset = std::vector<float_64>(simDim);
            DataSpace<simDim> globalSlideOffset;
            const pmacc::Selection<simDim> localDomain = Environment<simDim>::get().SubGrid().getLocalDomain();
            const uint32_t numSlides = MovingWindow::getInstance().getSlideCounter(currentStep);
            globalSlideOffset.y() += numSlides * localDomain.size.y();
            for(uint32_t d = 0; d < simDim; ++d)
            {
                gridGlobalOffset.at(simDim - 1 - d)
                    = float_64(cellSize[d]) * float_64(movingWindow.globalDimensions.offset[d] + globalSlideOffset[d]);
            }
            return gridGlobalOffset;
        }

        inline SetMeshAttributes::SetMeshAttributes(uint32_t currentStep)
            : m_gridGlobalOffset{initGridGlobalOffset(currentStep)}
        {
        }

        inline SetMeshAttributes const& SetMeshAttributes::operator()(::openPMD::Mesh& mesh) const
        {
            mesh.setAxisLabels(m_axisLabels);
            mesh.setDataOrder(m_dataOrder);
            mesh.setGeometry(m_geometry);
            mesh.setGridGlobalOffset(m_gridGlobalOffset);
            mesh.setGridSpacing(m_gridSpacing);
            mesh.setGridUnitSI(m_gridUnitSI);
            mesh.setTimeOffset(m_timeOffset);
            mesh.setUnitDimension(m_unitDimension);
            return *this;
        }

        inline SetMeshAttributes const& SetMeshAttributes::operator()(::openPMD::MeshRecordComponent& dataset) const
        {
            dataset.setPosition(m_position);
            dataset.setUnitSI(m_unitSI);
            return *this;
        }
    } // namespace openPMD
} // namespace picongpu
