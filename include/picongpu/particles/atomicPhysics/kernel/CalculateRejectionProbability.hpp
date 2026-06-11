/*
 * SPDX-FileCopyrightText: Brian Marre
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "picongpu/defines.hpp"
#include "picongpu/particles/atomicPhysics/FieldEnergy.hpp"

#include <cstdint>

namespace picongpu::particles::atomicPhysics::kernel
{
    //! methods for calculating the rejection probability
    struct CalculateRejectionProbability
    {
        using VectorIdx = DataSpace<picongpu::simDim>;

        /** store histogram bin rejection probability for specified bin in the passed rejectionProbabilityCacheBin
         *
         * @param binIndex
         * @param histogram
         * @param rejectionProbabilityCacheBin entry for bin will be set to -1 if no rejection necessary or rejection
         *  probability, >= 0, otherwise
         * @param sharedResourcesOverSubscribed previous state of sharedResourcesOverSubscribed
         *
         * @return bin is over subscribed
         */
        template<typename T_Histogram, typename T_RejectionProbabilityCache_Bin>
        HDINLINE static bool ofHistogramBin(
            uint32_t const binIndex,
            T_Histogram const& histogram,
            T_RejectionProbabilityCache_Bin& rejectionProbabilityCacheBin)
        {
            float_X const weight0 = histogram.getBinWeight0(binIndex);
            float_X const deltaWeight = histogram.getBinDeltaWeight(binIndex);

            float_X rejectionProbability = -1._X;
            bool sharedResourcesOverSubscribed = false;
            if(weight0 < deltaWeight)
            {
                // bin is oversubscribed by suggested changes

                // calculate rejection probability
                rejectionProbability = math::max(
                    // proportion of weight we want to reject
                    (deltaWeight - weight0) / deltaWeight,
                    // but at least one average one macro ion should be rejected
                    picongpu::sim.unit.typicalNumParticlesPerMacroParticle() / deltaWeight);

                // set flag that we found at least one over subscribed resource
                sharedResourcesOverSubscribed = true;
            }

            rejectionProbabilityCacheBin.setBin(binIndex, rejectionProbability);
            return sharedResourcesOverSubscribed;
        }

        /** store cell rejection probability for specified linearCellIndex in the passed rejectionProbabilityCacheCell
         *
         * @param linearCellIndex 1D index of the cell
         * @param superCellCellOffset offset of the superCell in cells
         * @param eFieldBox dataBox giving access to the eField Values of all local cells
         * @param eFieldEnergyUseCacheCell cache of the EField energy use for each cell
         * @param rejectionProbabilityCacheBin entry for bin will be set to -1 if no rejection necessary or rejection
         *  probability, >= 0, otherwise
         * @param sharedResourcesOverSubscribed previous state of sharedResourcesOverSubscribed
         *
         * @return cell is oversubscribed
         */
        template<typename T_EFieldBox, typename T_EFieldEnergyUseCacheCell, typename T_RejectionProbabilityCache_Cell>
        HDINLINE static bool ofCell(
            uint32_t const linearCellIndex,
            VectorIdx const& superCellCellOffset,
            T_EFieldBox const eFieldBox,
            T_EFieldEnergyUseCacheCell const& eFieldEnergyUseCacheCell,
            T_RejectionProbabilityCache_Cell& rejectionProbabilityCacheCell)
        {
            VectorIdx const localCellIndex
                = pmacc::math::mapToND(picongpu::SuperCellSize::toRT(), static_cast<int>(linearCellIndex));
            VectorIdx const cellIndex = localCellIndex + superCellCellOffset;

            // unit_energy
            float_X const eFieldEnergy = FieldEnergy::getEFieldEnergy(pmacc::math::l2norm2(eFieldBox(cellIndex)));

            // unit: eV * 1 = eV * unit_energy/unit_energy = (ev / unit_energy) * unit_energy = unit_energy
            float_X const eFieldEnergyUse
                = picongpu::sim.pic.get_eV() * eFieldEnergyUseCacheCell.energyUsed(linearCellIndex);

            float_X rejectionProbability = -1._X;
            bool sharedResourcesOverSubscribed = false;
            if(eFieldEnergyUse > eFieldEnergy)
            {
                // cell is oversubscribed by suggested changes

                // calculate rejection probability
                rejectionProbability = pmacc::math::max(
                    // proportion of weight we want to reject
                    (eFieldEnergyUse - eFieldEnergy) / eFieldEnergyUse,
                    // but approximately at least one average one macro ion per cell should be rejected
                    1._X / static_cast<float_X>(sim.getTypicalNumParticlesPerCell()));

                // set flag that we found at least one over subscribed resource
                sharedResourcesOverSubscribed = true;
            }

            rejectionProbabilityCacheCell.setCell(linearCellIndex, rejectionProbability);
            return sharedResourcesOverSubscribed;
        }
    };
} // namespace picongpu::particles::atomicPhysics::kernel
