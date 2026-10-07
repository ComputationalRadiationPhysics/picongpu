/* Copyright 2013-2024 Axel Huebl, Felix Schmitt, Rene Widera, Franz Poeschel
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

#if (ENABLE_OPENPMD == 1)

#    include "picongpu/defines.hpp"
#    include "picongpu/plugins/openPMD/GetComponentsType.hpp"
#    include "picongpu/plugins/openPMD/openPMDWriter.def"
#    include "picongpu/traits/PICToOpenPMD.hpp"

#    include <pmacc/assert.hpp>
#    include <pmacc/traits/GetComponentsType.hpp>
#    include <pmacc/traits/GetNComponents.hpp>
#    include <pmacc/traits/Resolve.hpp>

#    include <cstdint>
#    include <memory>
#    include <stdexcept>
#    include <type_traits>

#    include <openPMD/openPMD.hpp>

namespace picongpu
{
    namespace openPMD
    {
        using namespace pmacc;

        namespace detail
        {
            //! tag carrying a C++ type for datatype dispatch
            template<typename T>
            struct TypeTag
            {
                using type = T;
            };

            /** Call @p functor with a tag of the C++ type corresponding to the
             * openPMD datatype @p dt.
             *
             * Written as a plain switch (instead of openPMD's `switchType`) so
             * that it works with all supported openPMD-api versions (0.15+).
             */
            template<typename T_Functor>
            void dispatchDatatype(::openPMD::Datatype dt, T_Functor&& functor)
            {
                using DT = ::openPMD::Datatype;
                switch(dt)
                {
                case DT::CHAR:
                    functor(TypeTag<char>{});
                    break;
                case DT::SCHAR:
                    functor(TypeTag<signed char>{});
                    break;
                case DT::UCHAR:
                    functor(TypeTag<unsigned char>{});
                    break;
                case DT::SHORT:
                    functor(TypeTag<short>{});
                    break;
                case DT::INT:
                    functor(TypeTag<int>{});
                    break;
                case DT::LONG:
                    functor(TypeTag<long>{});
                    break;
                case DT::LONGLONG:
                    functor(TypeTag<long long>{});
                    break;
                case DT::USHORT:
                    functor(TypeTag<unsigned short>{});
                    break;
                case DT::UINT:
                    functor(TypeTag<unsigned int>{});
                    break;
                case DT::ULONG:
                    functor(TypeTag<unsigned long>{});
                    break;
                case DT::ULONGLONG:
                    functor(TypeTag<unsigned long long>{});
                    break;
                case DT::FLOAT:
                    functor(TypeTag<float>{});
                    break;
                case DT::DOUBLE:
                    functor(TypeTag<double>{});
                    break;
                case DT::LONG_DOUBLE:
                    functor(TypeTag<long double>{});
                    break;
                default:
                    throw std::runtime_error(
                        "openPMD: unsupported datatype for a particle attribute (only arithmetic types are "
                        "supported).");
                }
            }
        } // namespace detail

        /** Load attribute of a species from openPMD checkpoint storage
         *
         * @tparam T_Identifier identifier of species attribute
         */
        template<typename T_Identifier>
        struct LoadParticleAttributesFromOpenPMD
        {
            /** read attributes from openPMD file
             *
             * @param params thread params
             * @param frame frame with all particles
             * @param particleSpecies the openpmd representation of the species
             * @param particlesOffset read offset in the attribute array
             * @param elements number of elements which should be read the attribute
             * array
             */
            template<typename FrameType>
            HINLINE void operator()(
                ThreadParams* params,
                FrameType& frame,
                ::openPMD::ParticleSpecies particleSpecies,
                uint64_t const particlesOffset,
                uint64_t const elements)
            {
                using Identifier = T_Identifier;
                using ValueType = typename pmacc::traits::Resolve<Identifier>::type::type;
                uint32_t const components = GetNComponents<ValueType>::value;
                using ComponentType = typename GetComponentsType<ValueType>::type;
                picongpu::traits::OpenPMDName<Identifier> openPMDName;
                /* SI unit of the attribute in the simulation, per component;
                 * used to convert values read from the file (which carry their
                 * own `unitSI`) into the simulation's internal representation. */
                picongpu::traits::OpenPMDUnit<Identifier> openPMDUnit;
                std::vector<double> const unitSISim = openPMDUnit();
                log<picLog::INPUT_OUTPUT>("openPMD: ( begin ) load species attribute: %1%") % openPMDName();

                std::string const name_lookup[] = {"x", "y", "z"};

                for(uint32_t n = 0; n < components; ++n)
                {
                    ::openPMD::Record record = particleSpecies[openPMDName()];
                    ::openPMD::RecordComponent rc = components > 1 ? record[name_lookup[n]] : record;

                    // conversion factor from the file's unit to the simulation's unit
                    double const unitFactor
                        = unitSISim.size() > n && unitSISim[n] != 0.0 ? rc.unitSI() / unitSISim[n] : 1.0;

                    ValueType* dataPtr = frame.getIdentifier(Identifier()).getPointer();

                    uint64_t globalNumElements = 1;
                    for(auto ext : rc.getExtent())
                    {
                        globalNumElements *= ext;
                    }

                    log<picLog::INPUT_OUTPUT>("openPMD:  Did read %1% local of %2% global elements for "
                                              "%3%")
                        % elements % globalNumElements % openPMDName();

                    if(elements == 0)
                    {
                        params->openPMDSeries->flush();
                        continue;
                    }

                    // avoid deadlock between not finished pmacc tasks and mpi
                    // calls in openPMD
                    eventSystem::getTransactionEvent().waitForFinished();

                    /* Read in the file's native datatype and convert explicitly.
                     * This supports files whose datatype differs from the
                     * simulation precision (e.g. double-precision input into a
                     * single-precision simulation).
                     */
                    detail::dispatchDatatype(
                        rc.getDatatype(),
                        [&](auto tag)
                        {
                            using LoadType = typename std::decay_t<decltype(tag)>::type;
                            auto loadBfr = std::shared_ptr<LoadType>{
                                new LoadType[elements],
                                [](LoadType* ptr) { delete[] ptr; }};
                            rc.loadChunkRaw(
                                loadBfr.get(),
                                ::openPMD::Offset{particlesOffset},
                                ::openPMD::Extent{elements});

                            /** start a blocking read of all scheduled variables
                             *  (this is collective call in many methods of openPMD
                             * backends)
                             */
                            params->openPMDSeries->flush();

                        /* copy component from temporary array to array of
                         * structs, converting to the simulation precision and unit */
#    pragma omp parallel for simd
                            for(size_t i = 0; i < elements; ++i)
                            {
                                ComponentType* ref = &reinterpret_cast<ComponentType*>(dataPtr)[i * components + n];
                                *ref = static_cast<ComponentType>(static_cast<double>(loadBfr.get()[i]) * unitFactor);
                            }
                        });
                }

                log<picLog::INPUT_OUTPUT>("openPMD:  ( end ) load species attribute: %1%") % openPMDName();
            }
        };

    } /* namespace openPMD */
} /* namespace picongpu */

#endif
