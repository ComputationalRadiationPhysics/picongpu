/**
 * Copyright 2013-2015 Heiko Burau, Rene Widera, Felix Schmitt,
 *                     Richard Pausch
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

#include "math/vector/Int.hpp"
#include "math/vector/Float.hpp"
#include "math/vector/Size_t.hpp"
#include "dataManagement/DataConnector.hpp"
#include "fields/FieldB.hpp"
#include "fields/FieldE.hpp"
#include "math/Vector.hpp"
#include "cuSTL/algorithm/mpi/Gather.hpp"
#include "cuSTL/container/DeviceBuffer.hpp"
#include "cuSTL/container/HostBuffer.hpp"
#include "cuSTL/cursor/tools/slice.hpp"
#include "cuSTL/algorithm/kernel/Foreach.hpp"
#include "cuSTL/algorithm/host/Foreach.hpp"
#include "lambda/Expression.hpp"
#include "SliceFieldPrinter.hpp"
#include <sstream>

namespace picongpu
{

namespace SliceFieldPrinterHelper
{
template<class Field>
class ConversionFunctor
{
public:
  /* convert field data to higher precision and convert to SI units on GPUs */
  DINLINE void operator()(float3_64& target, const typename Field::ValueType fieldData) const
  {
    target = precisionCast<float_64>(fieldData) *  float_64((Field::getUnit())[0]) ;
  }
};
} // end namespace SliceFieldPrinterHelper


template<typename Field>
void SliceFieldPrinter<Field>::pluginLoad()
{
    if( float_X(0.0) <= slicePoint && slicePoint <= float_X(1.0))
      {
        /* in case the slice point is inside of [0.0,1.0] */
        sliceIsOK = true;
        Environment<>::get().PluginConnector().setNotificationPeriod(this, this->notifyFrequency);
        namespace vec = ::PMacc::math;
        typedef SuperCellSize BlockDim;

        vec::Size_t<simDim> size = vec::Size_t<simDim>(this->cellDescription->getGridSuperCells()) * precisionCast<size_t>(BlockDim::toRT())
          - precisionCast<size_t>(2 * BlockDim::toRT());
        this->dBuffer_SI = new container::DeviceBuffer<float3_64, simDim-1>(
                        size.shrink<simDim-1>((this->plane+1)%simDim));
      }
    else
      {
        /* in case the slice point is outside of [0.0,1.0] */
        sliceIsOK = false;
        std::cerr << "In the SliceFieldPrinter plugin a slice point"
                  << " (slice_point=" << slicePoint
                  << ") is outside of [0.0, 1.0]. " << std::endl
                  << "The request will be ignored. " << std::endl;
      }
}

template<typename Field>
void SliceFieldPrinter<Field>::pluginUnload()
{
    __delete(this->dBuffer_SI);
}

template<typename Field>
void SliceFieldPrinter<Field>::pluginRegisterHelp(po::options_description&)
{
    // nothing to do here
}

template<typename Field>
std::string SliceFieldPrinter<Field>::pluginGetName() const
{
    return "SliceFieldPrinter";
}

template<typename Field>
void SliceFieldPrinter<Field>::notify(uint32_t currentStep)
{
    if(sliceIsOK)
    {
      namespace vec = ::PMacc::math;
      typedef SuperCellSize BlockDim;
      DataConnector &dc = Environment<>::get().DataConnector();
      BOOST_AUTO(field_coreBorder,
                 dc.getData<Field > (Field::getName(), true).getGridBuffer().
                 getDeviceBuffer().cartBuffer().
                 view(BlockDim::toRT(), -BlockDim::toRT()));

      std::ostringstream filename;
      filename << this->fileName << "_" << currentStep << ".dat";
      printSlice(field_coreBorder, this->plane, this->slicePoint, filename.str());
    }
}

template<typename Field>
template<typename TField>
void SliceFieldPrinter<Field>::printSlice(const TField& field, int nAxis, float slicePoint, std::string filename)
{
    namespace vec = PMacc::math;

    PMacc::GridController<simDim>& con = PMacc::Environment<simDim>::get().GridController();
    vec::Size_t<simDim> gpuDim = (vec::Size_t<simDim>)con.getGpuNodes();
    vec::Size_t<simDim> globalGridSize = gpuDim * field.size();
    int globalPlane = globalGridSize[nAxis] * slicePoint;
    int localPlane = globalPlane % field.size()[nAxis];
    int gpuPlane = globalPlane / field.size()[nAxis];

    vec::Int<simDim> nVector(vec::Int<simDim>::create(0));
    nVector[nAxis] = 1;

    zone::SphericZone<simDim> gpuGatheringZone(gpuDim, nVector * gpuPlane);
    gpuGatheringZone.size[nAxis] = 1;

    algorithm::mpi::Gather<simDim> gather(gpuGatheringZone);
    if(!gather.participate()) return;

    using namespace lambda;
#if(SIMDIM==DIM3)
    vec::UInt32<3> twistedAxesVec((nAxis+1)%3, (nAxis+2)%3, nAxis);

    /* convert data to higher precision and to SI units */
    SliceFieldPrinterHelper::ConversionFunctor<Field> cf;
    algorithm::kernel::Foreach<vec::CT::UInt32<4,4,1> >()(
      dBuffer_SI->zone(), dBuffer_SI->origin(),
      cursor::tools::slice(field.originCustomAxes(twistedAxesVec)(0,0,localPlane)),
      cf );
#endif
#if(SIMDIM==DIM2)
    vec::UInt32<2> twistedAxesVec((nAxis+1)%2, nAxis);

    /* convert data to higher precision and to SI units */
    SliceFieldPrinterHelper::ConversionFunctor<Field> cf;
    algorithm::kernel::Foreach<vec::CT::UInt32<16,1,1> >()(
      dBuffer_SI->zone(), dBuffer_SI->origin(),
      cursor::tools::slice(field.originCustomAxes(twistedAxesVec)(0,localPlane)),
      cf );
#endif

    /* copy selected plane from device to host */
    container::HostBuffer<float3_64, simDim-1> hBuffer(dBuffer_SI->size());
    hBuffer = *dBuffer_SI;

    /* collect data from all nodes/GPUs */
    container::HostBuffer<float3_64, simDim-1> globalBuffer(hBuffer.size() * gpuDim.shrink<simDim-1>((nAxis+1)%simDim));
    gather(globalBuffer, hBuffer, nAxis);
    if(!gather.root()) return;
    std::ofstream file(filename.c_str());
    file << globalBuffer;
}

} /* end namespace picongpu */
