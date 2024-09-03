/* Copyright 2022-2023 Brian Marre
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
// need: picongpu/param/atomicPhysics_Debug.param and picongpu/param/atomicPhysics.param

#include "picongpu/particles/atomicPhysics/ConvertEnum.hpp"
#include "picongpu/particles/atomicPhysics/atomicData/AtomicTuples.def"
#include "picongpu/particles/atomicPhysics/atomicData/DataBox.hpp"
#include "picongpu/particles/atomicPhysics/atomicData/DataBuffer.hpp"

#include <cstdint>
#include <memory>

/** @file implements the storage of charge state property data
 *
 * The charge state data consists of the following data sets:
 *
 * - list of ionization states (sorted in ascending order of ionization):
 *      [ (ionization energy, [eV]
 *         screenedCharge, [eV] )]
 */

namespace picongpu::particles::atomicPhysics::atomicData
{
    /** data box storing charge state property data
     *
     * for use on device.
     *
     * @tparam T_DataBoxType dataBox type used for storage
     * @tparam T_Number dataType used for number storage, typically uint32_t
     * @tparam T_Value dataType used for value storage, typically float_X
     * @tparam T_atomicNumber atomic number of element this data corresponds to, eg. Cu -> 29
     */
    template<
        typename T_Number,
        typename T_Value,
        uint8_t T_atomicNumber> // element
    class ChargeStateDataBox : public DataBox<T_Number, T_Value>
    {
    public:
        using S_ChargeStateTuple = ChargeStateTuple<T_Value>;
        using S_DataBox = DataBox<T_Number, T_Value>;
        static constexpr uint8_t atomicNumber = T_atomicNumber;

    private:
        //! unit: eV
        typename S_DataBox::BoxValue m_boxIonizationEnergy;
        //! unit: elementary charge
        typename S_DataBox::BoxValue m_boxScreenedCharge;

    public:
        /** constructor
         *
         * @attention charge state data must be sorted by ascending charge and
         * @attention the completely ionized state must be left out.
         *
         * @param ionizationEnergy ionization energy[eV] of charge states
         * @param screenedCharge screenedCharge[e] of charge states
         */
        ChargeStateDataBox(typename S_DataBox::BoxValue ionizationEnergy, typename S_DataBox::BoxValue screenedCharge)
            : m_boxIonizationEnergy(ionizationEnergy)
            , m_boxScreenedCharge(screenedCharge)
        {
        }

        /** store charge state in data box
         *
         * @attention do not forget to call syncToDevice() on the
         *  corresponding buffer, or the state is only added on the host side.
         * @attention needs to fulfill all ordering and content assumptions of constructor!
         *
         * @param collectionIndex index of data box entry to rewrite
         * @param tuple chargeState tuple read from input file
         */
        HINLINE void store(uint32_t const collectionIndex, S_ChargeStateTuple& tuple)
        {
            uint8_t chargeState = u8(std::get<0>(tuple));

            if constexpr(picongpu::atomicPhysics::debug::atomicData::RANGE_CHECKS_IN_DATA_LOAD)
            {
                if(collectionIndex >= u32(T_atomicNumber))
                {
                    throw std::runtime_error(
                        "atomicPhysics ERROR: out of range call store() chargeState property data");
                    return;
                }
                if(collectionIndex != chargeState)
                {
                    throw std::runtime_error(
                        "atomicPhysics ERROR: chargeState and collectionIndex of tuple added inconsistent");
                    return;
                }
                if(collectionIndex == T_atomicNumber)
                {
                    throw std::runtime_error(
                        "atomicPhysics ERROR: completely ionized charge state property data must not be stored");
                    return;
                }
                if(collectionIndex > T_atomicNumber)
                {
                    throw std::runtime_error("atomicPhysics ERROR: unphysical charge state may not be stored");
                    return;
                }
            }

            m_boxIonizationEnergy[collectionIndex] = std::get<1>(tuple);
            m_boxScreenedCharge[collectionIndex] = std::get<2>(tuple);
        }

        /** get ionization energy of the ground state of the charge state
         *
         * @attention NEVER call with chargeState >= T_atomicNumber, otherwise invalid memory access
         *
         * @return unit: eV
         */
        HDINLINE T_Value ionizationEnergy(uint8_t const chargeState) const
        {
            if constexpr(picongpu::atomicPhysics::debug::atomicData::RANGE_CHECKS_IN_DATA_QUERIES)
                if(chargeState >= u32(T_atomicNumber))
                {
                    printf("atomicPhysics ERROR: out of range ionizationEnergy() call\n");
                    return static_cast<T_Value>(0.);
                }

            return m_boxIonizationEnergy[chargeState];
        }

        //! @attention NEVER call with chargeState >= T_atomicNumber, otherwise invalid memory access
        HDINLINE T_Value screenedCharge(uint8_t const chargeState) const
        {
            if constexpr(picongpu::atomicPhysics::debug::atomicData::RANGE_CHECKS_IN_DATA_QUERIES)
                if(chargeState >= u32(T_atomicNumber))
                {
                    printf("atomicPhysics ERROR: out of range ionizationEnergy() call\n");
                    return static_cast<T_Value>(0.);
                }

            return m_boxScreenedCharge[chargeState];
        }
    };

    /** complementing buffer class
     *
     * @tparam T_Number dataType used for number storage, typically uint32_t
     * @tparam T_Value dataType used for value storage, typically float_X
     * @tparam T_atomicNumber atomic number of element this data corresponds to, eg. Cu -> 29
     */
    template<typename T_Number, typename T_Value, uint8_t T_atomicNumber>
    class ChargeStateDataBuffer : public DataBuffer<T_Number, T_Value>
    {
    public:
        using S_DataBuffer = DataBuffer<T_Number, T_Value>;
        using S_ChargeStateDataBox = ChargeStateDataBox<T_Number, T_Value, T_atomicNumber>;

    private:
        std::unique_ptr<typename S_DataBuffer::BufferValue> bufferIonizationEnergy;
        std::unique_ptr<typename S_DataBuffer::BufferValue> bufferScreenedCharge;

    public:
        HINLINE ChargeStateDataBuffer()
        {
            auto const guardSize = pmacc::DataSpace<1>::create(0);
            auto const layoutChargeStates = pmacc::GridLayout<1>(T_atomicNumber, guardSize).sizeWithoutGuardND();

            bufferIonizationEnergy.reset(new typename S_DataBuffer::BufferValue(layoutChargeStates, false));
            bufferScreenedCharge.reset(new typename S_DataBuffer::BufferValue(layoutChargeStates, false));
        }

        HINLINE S_ChargeStateDataBox getHostDataBox()
        {
            return ChargeStateDataBox<T_Number, T_Value, T_atomicNumber>(
                bufferIonizationEnergy->getHostBuffer().getDataBox(),
                bufferScreenedCharge->getHostBuffer().getDataBox());
        }

        HINLINE S_ChargeStateDataBox getDeviceDataBox()
        {
            return ChargeStateDataBox<T_Number, T_Value, T_atomicNumber>(
                bufferIonizationEnergy->getDeviceBuffer().getDataBox(),
                bufferScreenedCharge->getDeviceBuffer().getDataBox());
        }

        HINLINE void hostToDevice()
        {
            bufferIonizationEnergy->hostToDevice();
            bufferScreenedCharge->hostToDevice();
        }

        HINLINE void deviceToHost()
        {
            bufferIonizationEnergy->deviceToHost();
            bufferScreenedCharge->deviceToHost();
        }
    };
} // namespace picongpu::particles::atomicPhysics::atomicData
