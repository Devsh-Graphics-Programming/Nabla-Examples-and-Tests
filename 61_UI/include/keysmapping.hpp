#ifndef __NBL_KEYSMAPPING_H_INCLUDED__
#define __NBL_KEYSMAPPING_H_INCLUDED__

#include "common.hpp"

bool handleAddMapping(const char* tableID, IGimbalManipulateEncoder* encoder, IGimbalManipulateEncoder::EncoderType activeController, CVirtualGimbalEvent::VirtualEventType& selectedEventType, ui::E_KEY_CODE& newKey, ui::E_MOUSE_CODE& newMouseCode, bool& addMode);
bool displayKeyMappingsAndVirtualStatesInline(IGimbalManipulateEncoder* encoder, bool spawnWindow = false);

#endif // __NBL_KEYSMAPPING_H_INCLUDED__
