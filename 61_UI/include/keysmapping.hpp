#ifndef __NBL_KEYSMAPPING_H_INCLUDED__
#define __NBL_KEYSMAPPING_H_INCLUDED__

#include "common.hpp"

/// @brief Draw one editable row per control axis the camera accepts.
///
/// Each row carries the two keys that drive the axis, the rate they drive it at, the mouse movement and scroll
/// gains, and the button that gates relative movement. Edits land in `binding` immediately.
/// @return whether anything in `binding` changed.
bool displayCameraBindingTableInline(SCameraMouseKeyboardBinding& binding, uint32_t acceptedAxes, bool spawnWindow = false);

#endif // __NBL_KEYSMAPPING_H_INCLUDED__
