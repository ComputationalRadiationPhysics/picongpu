
#include "picongpu/defines.hpp"
#include "picongpu/fields/FieldTmpOperations.hpp"

struct BoundaryConditionsDirichlet
    {
        // return residual
        // return number of iterations 
        void operator()(FieldTmp& fieldV, FieldTmp& fiedlRho, MappingDesk *cellDescription)
        {
            // set boundary conditions on fieldV (Dirichlet or Neuman)

            // normalize the problem based on norm(fieldRho)

            pmacc::GridController<simDim>& gc = pmacc::Environment<simDim>::get().GridController();

            DataSpace<simDim> myGPUpos(gc.getPosition());
            DataSpace<simDim> gpus(gc.getGpuNodes());
            SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();
            auto globalDomain = subGrid.getGlobalDomain();
            auto localDomain = subGrid.getLocalDomain()
            auto const mapper = makeAreaMapper<BORDER>(*cellDescription());

            
            
            for(int dir=0; dir<simDim; dir++)
            {
                
                // todo check for moving window
                if(myGPUpos[dir] == 0){
                    DataSpace<2*Dim> indexLimitsEdge = ;
                    indexLimitsEdge[2*dir+1] = indexLimitsEdge[2*dir] + guardsSize[dir];
                    PMACC_LOCKSTEP_KERNEL().config(mapper.getGridDim(), SuperCellSize{})();}

                if(myGPUpos[dir] == gpus[dir] -1){
                    PMACC_LOCKSTEP_KERNEL(KernelComputeSupercells<BlockArea>{})
                .config(mapper.getGridDim(), pBox)(fieldTmpBox, pBox, solver, iFilter, mapper);
                }



                dirAlpaka = 2 - dir;
                if(hasBoundary_[2*dir] && bcsType_[2*dir]==0)
                {
                    indexLimitsEdge = this->alpakaHelper_.indexLimitsDataAlpaka_;
                    indexLimitsEdge[2*dirAlpaka+1]=indexLimitsEdge[2*dirAlpaka] + guards_[dir];
                    alpaka::exec<Acc>(this->queueSolverNonBlocking1_, workDivExtentApplyDirichletBCsFromFunctionKernel_, applyDirichletBCsFromFunctionKernel, bufMdSpan, this->exactSolutionAndBCs_, 
                                            indexLimitsEdge, this->alpakaHelper_.indexLimitsDataAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.origin_, 
                                            this->alpakaHelper_.globalLocation_, this->alpakaHelper_.nlocal_noguards_, this->alpakaHelper_.haloSize_ );

                }
                if(hasBoundary_[2*dir+1] && bcsType_[2*dir+1]==0)
                {
                    indexLimitsEdge = this->alpakaHelper_.indexLimitsDataAlpaka_;
                    indexLimitsEdge[2*dirAlpaka]=indexLimitsEdge[2*dirAlpaka+1] - guards_[dir];
                    alpaka::exec<Acc>(this->queueSolverNonBlocking1_, workDivExtentApplyDirichletBCsFromFunctionKernel_, applyDirichletBCsFromFunctionKernel, bufMdSpan, this->exactSolutionAndBCs_, 
                                            indexLimitsEdge, this->alpakaHelper_.indexLimitsDataAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.origin_, 
                                            this->alpakaHelper_.globalLocation_, this->alpakaHelper_.nlocal_noguards_, this->alpakaHelper_.haloSize_ );
                }
            }

        }
    };


template<int DIM, typename T_data> 
struct ApplyDirichletBCsFromFunctionKernel
{
    template<typename TAcc, typename TMdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TMdSpan bufData, const auto boundaryFunction, DataSpace<Dim> guardsSize, DataSpace<Dim> extents, DataSpace<Dim> offsets ) const -> void
    {
        // Get indexes
        auto const gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
    
        auto const gridThreadIdxShifted = gridThreadIdx + guardsSize;
        //auto const indxData = gridThreadIdxShifted + adjustIdx;
        //auto const indxGuard = gridThreadIdxShifted - adjustIdx;
        
        auto const iGrid = gridThreadIdxShifted[2];
        auto const jGrid = gridThreadIdxShifted[1];
        auto const kGrid = gridThreadIdxShifted[0];

        const T_data x = (iGrid-indexLimitsData[4])*ds[2] + offsets[2]*ds[2];
        const T_data y = (jGrid-indexLimitsData[2])*ds[1] + offsets[1]*ds[1];
        const T_data z = (kGrid-indexLimitsData[0])*ds[0] + offsets[0]*ds[0];


        if( iGrid>=indexLimitsEdge[4] && iGrid<indexLimitsEdge[5] && jGrid>=indexLimitsEdge[2] && jGrid<indexLimitsEdge[3] && kGrid>=indexLimitsEdge[0] && kGrid<indexLimitsEdge[1])
        {
            bufData(gridThreadIdxShifted[0],gridThreadIdxShifted[1],gridThreadIdxShifted[2]) = boundaryFunction(x,y,z);
        }
    }
};
    