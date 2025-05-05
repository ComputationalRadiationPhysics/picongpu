
#include "picongpu/defines.hpp"
#include "picongpu/fields/FieldTmpOperations.hpp"

struct BICGStab
    {
        // return residual
        // return number of iterations 
        void operator()(FieldTmp& fieldV, FieldTmp& fiedlRho, MappingDesk *cellDescription)
        {
            // set boundary conditions on fieldV (Dirichlet or Neuman)

            // normalize the problem based on norm(fieldRho)



        }
    };