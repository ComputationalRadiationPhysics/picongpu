/* Copyright 2013-2021 Axel Huebl, Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Benjamin Worpitz, Richard Pausch
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "picongpu/simulation_defines.hpp"

#include "picongpu/fields/FieldB.hpp"
#include "picongpu/fields/FieldE.hpp"

#include "picongpu/plugins/ISimulationPlugin.hpp"

#include <pmacc/mpi/reduceMethods/Reduce.hpp>
#include <pmacc/mpi/MPIReduce.hpp>
#include <pmacc/nvidia/functors/Add.hpp>
#include <pmacc/nvidia/reduce/Reduce.hpp>
#include <pmacc/memory/boxes/DataBoxDim1Access.hpp>
#include <pmacc/memory/boxes/DataBoxUnaryTransform.hpp>
#include <pmacc/dataManagement/DataConnector.hpp>
#include <pmacc/dimensions/DataSpaceOperations.hpp>

#include "common/txtFileHandling.hpp"

#include <iostream>
#include <fstream>
#include <memory>


namespace picongpu
{
    using namespace pmacc;

    namespace po = boost::program_options;

    namespace energyFields
    {
        template<typename T_Type>
        struct cast64Bit
        {
            typedef typename TypeCast<float_64, T_Type>::result result;

            HDINLINE result operator()(const T_Type& value) const
            {
                return precisionCast<float_64>(value);
            }
        };

        template<typename T_Type>
        struct squareComponentWise
        {
            using result = T_Type;

            HDINLINE result operator()(const T_Type& value) const
            {
                return value * value;
            }
        };

    } // namespace energyFields

    class EnergyFields : public ISimulationPlugin
    {
    private:
        MappingDesc* cellDescription;
        std::string notifyPeriod;

        std::string pluginName;
        std::string pluginPrefix;
        std::string filename;
        std::ofstream outFile;
        /*only rank 0 create a file*/
        bool writeToFile;

        mpi::MPIReduce mpiReduce;

        nvidia::reduce::Reduce* localReduce;

        typedef promoteType<float_64, FieldB::ValueType>::type EneVectorType;

    public:
        EnergyFields()
            : cellDescription(nullptr)
            , pluginName("EnergyFields: calculate the energy of the fields")
            , pluginPrefix(std::string("fields_energy"))
            , filename(pluginPrefix + ".dat")
            , writeToFile(false)
            , localReduce(nullptr)
        {
            Environment<>::get().PluginConnector().registerPlugin(this);
        }

        virtual ~EnergyFields()
        {
        }

        void notify(uint32_t currentStep)
        {
            getEnergyFields(currentStep);
        }

        void pluginRegisterHelp(po::options_description& desc)
        {
            desc.add_options()(
                (pluginPrefix + ".period").c_str(),
                po::value<std::string>(&notifyPeriod),
                "enable plugin [for each n-th step]");
        }

        std::string pluginGetName() const
        {
            return pluginName;
        }

        void setMappingDescription(MappingDesc* cellDescription)
        {
            this->cellDescription = cellDescription;
        }

    private:
        void pluginLoad()
        {
            if(!notifyPeriod.empty())
            {
                localReduce = new nvidia::reduce::Reduce(1024);
                writeToFile = mpiReduce.hasResult(mpi::reduceMethods::Reduce());

                if(writeToFile)
                {
                    outFile.open(filename.c_str(), std::ofstream::out | std::ostream::trunc);
                    if(!outFile)
                    {
                        std::cerr << "Can't open file [" << filename << "] for output, disable plugin output. "
                                  << std::endl;
                        writeToFile = false;
                    }
                    // create header of the file
                    outFile << "#step total[Joule] Bx[Joule] By[Joule] Bz[Joule] Ex[Joule] Ey[Joule] Ez[Joule]"
                            << " \n";
                }
                Environment<>::get().PluginConnector().setNotificationPeriod(this, notifyPeriod);
            }
        }

        void pluginUnload()
        {
            if(!notifyPeriod.empty())
            {
                if(writeToFile)
                {
                    outFile.flush();
                    outFile << std::endl; // now all data are written to file
                    if(outFile.fail())
                        std::cerr << "Error on flushing file [" << filename << "]. " << std::endl;
                    outFile.close();
                }
                __delete(localReduce);
            }
        }

        void restart(uint32_t restartStep, const std::string restartDirectory)
        {
            if(!writeToFile)
                return;

            writeToFile = restoreTxtFile(outFile, filename, restartStep, restartDirectory);
        }

        void checkpoint(uint32_t currentStep, const std::string checkpointDirectory)
        {
            if(!writeToFile)
                return;

            checkpointTxtFile(outFile, filename, currentStep, checkpointDirectory);
        }

        void getEnergyFields(uint32_t currentStep)
        {
            DataConnector& dc = Environment<>::get().DataConnector();

            auto fieldE = dc.get<FieldE>(FieldE::getName(), true);
            auto fieldB = dc.get<FieldB>(FieldB::getName(), true);

            /* idx == 0 -> fieldB
             * idx == 1 -> fieldE
             */
            EneVectorType globalFieldEnergy[2];
            globalFieldEnergy[0] = EneVectorType::create(0.0);
            globalFieldEnergy[1] = EneVectorType::create(0.0);

            EneVectorType localReducedFieldEnergy[2];
            localReducedFieldEnergy[0] = reduceField(fieldB);
            localReducedFieldEnergy[1] = reduceField(fieldE);

            mpiReduce(
                nvidia::functors::Add(),
                globalFieldEnergy,
                localReducedFieldEnergy,
                2,
                mpi::reduceMethods::Reduce());

            float_64 energyFieldBReduced = 0.0;
            float_64 energyFieldEReduced = 0.0;

            for(int d = 0; d < FieldB::numComponents; ++d)
            {
                /* B field convert */
                globalFieldEnergy[0][d] *= float_64(0.5 / MUE0 * CELL_VOLUME);
                /* E field convert */
                globalFieldEnergy[1][d] *= float_64(EPS0 * CELL_VOLUME * 0.5);

                /* add all to one */
                energyFieldBReduced += globalFieldEnergy[0][d];
                energyFieldEReduced += globalFieldEnergy[1][d];
            }

            float_64 globalEnergy = energyFieldEReduced + energyFieldBReduced;


            if(writeToFile)
            {
                using dbl = std::numeric_limits<float_64>;

                outFile.precision(dbl::digits10);
                outFile << currentStep << " " << std::scientific << globalEnergy * UNIT_ENERGY << " "
                        << (globalFieldEnergy[0] * UNIT_ENERGY).toString(" ", "") << " "
                        << (globalFieldEnergy[1] * UNIT_ENERGY).toString(" ", "") << std::endl;
            }
        }

    private:
        template<typename T_Field>
        EneVectorType reduceField(std::shared_ptr<T_Field> field)
        {
            /*define stacked DataBox's for reduce algorithm*/
            typedef DataBoxUnaryTransform<typename T_Field::DataBoxType, energyFields::squareComponentWise>
                TransformedBox;
            typedef DataBoxUnaryTransform<TransformedBox, energyFields::cast64Bit> Box64bit;
            using D1Box = DataBoxDim1Access<Box64bit>;

            /* reduce field E*/
            DataSpace<simDim> fieldSize = field->getGridLayout().getDataSpaceWithoutGuarding();
            DataSpace<simDim> fieldGuard = field->getGridLayout().getGuard();

            TransformedBox fieldTransform(field->getDeviceDataBox().shift(fieldGuard));
            Box64bit field64bit(fieldTransform);
            D1Box d1Access(field64bit, fieldSize);

            EneVectorType fieldEnergyReduced
                = (*localReduce)(nvidia::functors::Add(), d1Access, fieldSize.productOfComponents());

            return fieldEnergyReduced;
        }
    };

} // namespace picongpu
