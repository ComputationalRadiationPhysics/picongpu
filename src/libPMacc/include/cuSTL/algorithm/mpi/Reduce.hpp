#pragma once

#include "mpi.h"
#include "math/vector/Int.hpp"
#include "math/vector/UInt.hpp"
#include "cuSTL/container/HostBuffer.hpp"
#include "cuSTL/zone/SphericZone.hpp"
#include <vector>

namespace PMacc
{
namespace algorithm
{
namespace mpi
{

template<int dim>
class Reduce
{
private:
    MPI_Comm comm;
    bool m_participate;
public:
    Reduce(const zone::SphericZone<dim>& zone, bool setThisAsRoot = false);
    ~Reduce();
    
    template<typename Type, int conDim, typename ExprOrFunctor>
    void operator()(container::HostBuffer<Type, conDim>& dest, 
                    const container::HostBuffer<Type, conDim>& src,
                    ExprOrFunctor) const;
           
    inline bool participate() const {return m_participate;}
    inline bool root() const;
    inline int rank() const;
};
    
} // mpi
} // algorithm
} // PMacc

#include "cuSTL/algorithm/mpi/Reduce.tpp"
