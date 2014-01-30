/* 
 * File:   Environment.hpp
 * Author: conrad
 *
 * Created on 20. Januar 2014, 16:18
 */

#ifndef ENVIRONMENT_HPP
#define	ENVIRONMENT_HPP

// include Ausklammerungen für GameOfLife

#include "dataManagement/DataConnector.hpp"
#include "mappings/simulation/GridController.hpp"
#include "mappings/simulation/SubGrid.hpp"
#include "mappings/simulation/EnvironmentController.hpp"

#include "moduleSystem/ModuleConnector.hpp"

#include "particles/tasks/ParticleFactory.hpp"

#include "eventSystem/tasks/Factory.hpp"
#include "eventSystem/Manager.hpp"
#include "eventSystem/transactions/TransactionManager.hpp"
#include "eventSystem/streams/StreamController.hpp"

#include "memory/buffers/GridBuffer.hpp"
#include "nvidia/memory/MemoryInfo.hpp"


namespace PMacc {
    //using namespace gol;

    template <unsigned DIM = DIM1>
    class Environment {
    public:

        GridController<DIM>& getGridController() {
            return GridController<DIM>::getInstance();
        };

        StreamController& getStreamController() {
            return StreamController::getInstance();
        };

        Manager& getManager() {
            return Manager::getInstance();
        };

        TransactionManager& getTransactionManager() {
            return TransactionManager::getInstance();
        };

        SubGrid<DIM>& getSubGrid() {
            return SubGrid<DIM>::getInstance();
        };

        EnvironmentController& getEnvironmentController() {
            return EnvironmentController::getInstance();
        };

        Factory& getFactory() {
            return Factory::getInstance();
        }

        static Environment<DIM>& getInstance() {
            static Environment<DIM> instance;
            return instance;
        };

        void init(DataSpace<DIM> gridSize, DataSpace<DIM> devices, DataSpace<DIM> periodic) {
            GridController<DIM>::getInstance().init(devices, periodic);

            StreamController::getInstance();

            TransactionManager::getInstance();

            DataSpace<DIM> localGridSize(gridSize / devices);

            SubGrid<DIM>::getInstance().init(localGridSize, gridSize, GridController<DIM>::getInstance().getPosition() * localGridSize);

            EnvironmentController::getInstance();
        };

        void deinit() {
        };

    private:

        Environment() {

        };
        Environment(const Environment&);

        ~Environment();
        Environment& operator=(const Environment&);

    };

#define __startTransaction(...) (Environment<>::getInstance().getTransactionManager().startTransaction(__VA_ARGS__))
#define __startAtomicTransaction(...) (Environment<>::getInstance().getTransactionManager().startAtomicTransaction(__VA_ARGS__))
#define __endTransaction() (Environment<>::getInstance().getTransactionManager().endTransaction())
#define __startOperation(opType) (Environment<>::getInstance().getTransactionManager().startOperation(opType))
#define __getEventStream(opType) (Environment<>::getInstance().getTransactionManager().getEventStream(opType))
#define __getTransactionEvent() (Environment<>::getInstance().getTransactionManager().getTransactionEvent())
#define __setTransactionEvent(event) (Environment<>::getInstance().getTransactionManager().setTransactionEvent((event)))

}
#endif	/* ENVIRONMENT_HPP */

