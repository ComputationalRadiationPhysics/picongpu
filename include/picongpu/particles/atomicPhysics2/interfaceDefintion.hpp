
#include <pmacc/math/Vector.hpp>

//! @todo remove

namespace picongpu::particles
{
    namespace compileTime = pmacc::math::CT;

    // in-cell position to initialize macro particle to
    CONST_VECTOR(
        /*datatype*/ float_X,
        /*dim*/ 3,
        /*name*/ InCellOffset,
        /*x*/ 0.0,
        /*y*/ 0.0,
        /*z*/ 0.0);

    // cell in superCell to initialize macro particle to
    CONST_VECTOR(uint32_t, 3, CellIdx, 0u, 0u, 0u);

    struct OneSuperCellPositionParameter
    {
        /** Count of particles per cell at initial state
         *  unit: none
         */
        static constexpr uint32_t numParticlesPerSuperCell = 1u;

        //! initial position of macro particle in-cell, as CONST_VECTOR of relative position in cell (x \in [0.,1.),
        //! ...)
        const InCellOffset_t inCellOffset;

        //! spawnCell index, @attention must be within superCell extent!
        using spawnCellIdx = compileTime::shrinkTo<compileTime::Uint32<0u, 0u, 0u>, picongpu::simDim>::type;
    };
    using OneSuperCellPosition = OneSuperCellPositionImpl<OneSuperCellPositionParameter>;

} // namespace picongpu::particles
