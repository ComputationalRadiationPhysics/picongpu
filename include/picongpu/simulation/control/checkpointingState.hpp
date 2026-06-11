/*
 * SPDX-FileCopyrightText: Tapish Narwal
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <pmacc/simulationControl/Checkpointing.hpp>

namespace picongpu
{

    static constexpr pmacc::simulationControl::CheckpointingAvailability checkpointingEnabled =
#if (ENABLE_OPENPMD == 1)
        pmacc::simulationControl::CheckpointingAvailability::ENABLED;
#else
        pmacc::simulationControl::CheckpointingAvailability::DISABLED;
#endif
} // namespace picongpu
