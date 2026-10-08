// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_THIS_EXAMPLE_INPUT_CODE_NAMES_HPP_INCLUDED_
#define _NBL_THIS_EXAMPLE_INPUT_CODE_NAMES_HPP_INCLUDED_

#include <array>
#include <string_view>

#include "nbl/ui/KeyCodes.h"

/// @brief Stable string names for key codes and mouse buttons, used by the binding editor and the scripted input files.
struct CInputCodeNames final
{
public:
	static constexpr nbl::ui::E_KEY_CODE stringToKeyCode(std::string_view str)
	{
		if (str.size() == 1u)
		{
			const char upper = asciiToUpper(str.front());
			if ((upper >= 'A' && upper <= 'Z') || (upper >= '0' && upper <= '9'))
				return static_cast<nbl::ui::E_KEY_CODE>(upper);
		}

		return lookupNamedCode(str, NamedKeyCodes, nbl::ui::E_KEY_CODE::EKC_NONE);
	}

	static constexpr std::string_view keyCodeToString(const nbl::ui::E_KEY_CODE code)
	{
		const auto single = SingleCharacterKeyNames.find(static_cast<char>(code));
		if (single != std::string_view::npos)
			return SingleCharacterKeyNames.substr(single, 1u);

		return lookupCodeName(code, NamedKeyCodes, "NONE");
	}

	/// @brief Mouse button named by `str`, or `EMB_COUNT` when no button has that name.
	static constexpr nbl::ui::E_MOUSE_BUTTON stringToMouseButton(std::string_view str)
	{
		return lookupNamedCode(str, NamedMouseButtons, nbl::ui::EMB_COUNT);
	}

	static constexpr std::string_view mouseButtonToString(const nbl::ui::E_MOUSE_BUTTON button)
	{
		return lookupCodeName(button, NamedMouseButtons, "NONE");
	}

private:
	template<typename Code>
	struct SNamedCode final
	{
		std::string_view name;
		Code code;
	};

	template<typename Code, size_t N>
	static constexpr Code lookupNamedCode(std::string_view str, const std::array<SNamedCode<Code>, N>& table, const Code fallback)
	{
		for (const auto& entry : table)
		{
			if (str == entry.name)
				return entry.code;
		}

		return fallback;
	}

	template<typename Code, size_t N>
	static constexpr std::string_view lookupCodeName(const Code code, const std::array<SNamedCode<Code>, N>& table, const std::string_view fallback)
	{
		for (const auto& entry : table)
		{
			if (code == entry.code)
				return entry.name;
		}

		return fallback;
	}

	static constexpr char asciiToUpper(const char c)
	{
		return (c >= 'a' && c <= 'z') ? static_cast<char>(c - ('a' - 'A')) : c;
	}

	// single character key codes (`EKC_A`..`EKC_Z`, `EKC_0`..`EKC_9`) equal their ASCII value, this string provides stable storage for their names
	static constexpr std::string_view SingleCharacterKeyNames = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";

	static constexpr auto NamedKeyCodes = std::to_array<SNamedCode<nbl::ui::E_KEY_CODE>>({
		{ "BACKSPACE", nbl::ui::E_KEY_CODE::EKC_BACKSPACE },
		{ "TAB", nbl::ui::E_KEY_CODE::EKC_TAB },
		{ "CLEAR", nbl::ui::E_KEY_CODE::EKC_CLEAR },
		{ "ENTER", nbl::ui::E_KEY_CODE::EKC_ENTER },
		{ "LEFT_SHIFT", nbl::ui::E_KEY_CODE::EKC_LEFT_SHIFT },
		{ "RIGHT_SHIFT", nbl::ui::E_KEY_CODE::EKC_RIGHT_SHIFT },
		{ "LEFT_CONTROL", nbl::ui::E_KEY_CODE::EKC_LEFT_CONTROL },
		{ "RIGHT_CONTROL", nbl::ui::E_KEY_CODE::EKC_RIGHT_CONTROL },
		{ "LEFT_ALT", nbl::ui::E_KEY_CODE::EKC_LEFT_ALT },
		{ "RIGHT_ALT", nbl::ui::E_KEY_CODE::EKC_RIGHT_ALT },
		{ "PAUSE", nbl::ui::E_KEY_CODE::EKC_PAUSE },
		{ "CAPS_LOCK", nbl::ui::E_KEY_CODE::EKC_CAPS_LOCK },
		{ "ESCAPE", nbl::ui::E_KEY_CODE::EKC_ESCAPE },
		{ "SPACE", nbl::ui::E_KEY_CODE::EKC_SPACE },
		{ "PAGE_UP", nbl::ui::E_KEY_CODE::EKC_PAGE_UP },
		{ "PAGE_DOWN", nbl::ui::E_KEY_CODE::EKC_PAGE_DOWN },
		{ "END", nbl::ui::E_KEY_CODE::EKC_END },
		{ "HOME", nbl::ui::E_KEY_CODE::EKC_HOME },
		{ "LEFT_ARROW", nbl::ui::E_KEY_CODE::EKC_LEFT_ARROW },
		{ "RIGHT_ARROW", nbl::ui::E_KEY_CODE::EKC_RIGHT_ARROW },
		{ "DOWN_ARROW", nbl::ui::E_KEY_CODE::EKC_DOWN_ARROW },
		{ "UP_ARROW", nbl::ui::E_KEY_CODE::EKC_UP_ARROW },
		{ "SELECT", nbl::ui::E_KEY_CODE::EKC_SELECT },
		{ "PRINT", nbl::ui::E_KEY_CODE::EKC_PRINT },
		{ "EXECUTE", nbl::ui::E_KEY_CODE::EKC_EXECUTE },
		{ "PRINT_SCREEN", nbl::ui::E_KEY_CODE::EKC_PRINT_SCREEN },
		{ "INSERT", nbl::ui::E_KEY_CODE::EKC_INSERT },
		{ "DELETE", nbl::ui::E_KEY_CODE::EKC_DELETE },
		{ "HELP", nbl::ui::E_KEY_CODE::EKC_HELP },
		{ "LEFT_WIN", nbl::ui::E_KEY_CODE::EKC_LEFT_WIN },
		{ "RIGHT_WIN", nbl::ui::E_KEY_CODE::EKC_RIGHT_WIN },
		{ "APPS", nbl::ui::E_KEY_CODE::EKC_APPS },
		{ "COMMA", nbl::ui::E_KEY_CODE::EKC_COMMA },
		{ "PERIOD", nbl::ui::E_KEY_CODE::EKC_PERIOD },
		{ "SEMICOLON", nbl::ui::E_KEY_CODE::EKC_SEMICOLON },
		{ "OPEN_BRACKET", nbl::ui::E_KEY_CODE::EKC_OPEN_BRACKET },
		{ "CLOSE_BRACKET", nbl::ui::E_KEY_CODE::EKC_CLOSE_BRACKET },
		{ "BACKSLASH", nbl::ui::E_KEY_CODE::EKC_BACKSLASH },
		{ "APOSTROPHE", nbl::ui::E_KEY_CODE::EKC_APOSTROPHE },
		{ "ADD", nbl::ui::E_KEY_CODE::EKC_ADD },
		{ "SUBTRACT", nbl::ui::E_KEY_CODE::EKC_SUBTRACT },
		{ "MULTIPLY", nbl::ui::E_KEY_CODE::EKC_MULTIPLY },
		{ "DIVIDE", nbl::ui::E_KEY_CODE::EKC_DIVIDE },
		{ "F1", nbl::ui::E_KEY_CODE::EKC_F1 },
		{ "F2", nbl::ui::E_KEY_CODE::EKC_F2 },
		{ "F3", nbl::ui::E_KEY_CODE::EKC_F3 },
		{ "F4", nbl::ui::E_KEY_CODE::EKC_F4 },
		{ "F5", nbl::ui::E_KEY_CODE::EKC_F5 },
		{ "F6", nbl::ui::E_KEY_CODE::EKC_F6 },
		{ "F7", nbl::ui::E_KEY_CODE::EKC_F7 },
		{ "F8", nbl::ui::E_KEY_CODE::EKC_F8 },
		{ "F9", nbl::ui::E_KEY_CODE::EKC_F9 },
		{ "F10", nbl::ui::E_KEY_CODE::EKC_F10 },
		{ "F11", nbl::ui::E_KEY_CODE::EKC_F11 },
		{ "F12", nbl::ui::E_KEY_CODE::EKC_F12 },
		{ "F13", nbl::ui::E_KEY_CODE::EKC_F13 },
		{ "F14", nbl::ui::E_KEY_CODE::EKC_F14 },
		{ "F15", nbl::ui::E_KEY_CODE::EKC_F15 },
		{ "F16", nbl::ui::E_KEY_CODE::EKC_F16 },
		{ "F17", nbl::ui::E_KEY_CODE::EKC_F17 },
		{ "F18", nbl::ui::E_KEY_CODE::EKC_F18 },
		{ "F19", nbl::ui::E_KEY_CODE::EKC_F19 },
		{ "F20", nbl::ui::E_KEY_CODE::EKC_F20 },
		{ "F21", nbl::ui::E_KEY_CODE::EKC_F21 },
		{ "F22", nbl::ui::E_KEY_CODE::EKC_F22 },
		{ "F23", nbl::ui::E_KEY_CODE::EKC_F23 },
		{ "F24", nbl::ui::E_KEY_CODE::EKC_F24 },
		{ "NUMPAD_0", nbl::ui::E_KEY_CODE::EKC_NUMPAD_0 },
		{ "NUMPAD_1", nbl::ui::E_KEY_CODE::EKC_NUMPAD_1 },
		{ "NUMPAD_2", nbl::ui::E_KEY_CODE::EKC_NUMPAD_2 },
		{ "NUMPAD_3", nbl::ui::E_KEY_CODE::EKC_NUMPAD_3 },
		{ "NUMPAD_4", nbl::ui::E_KEY_CODE::EKC_NUMPAD_4 },
		{ "NUMPAD_5", nbl::ui::E_KEY_CODE::EKC_NUMPAD_5 },
		{ "NUMPAD_6", nbl::ui::E_KEY_CODE::EKC_NUMPAD_6 },
		{ "NUMPAD_7", nbl::ui::E_KEY_CODE::EKC_NUMPAD_7 },
		{ "NUMPAD_8", nbl::ui::E_KEY_CODE::EKC_NUMPAD_8 },
		{ "NUMPAD_9", nbl::ui::E_KEY_CODE::EKC_NUMPAD_9 },
		{ "NUM_LOCK", nbl::ui::E_KEY_CODE::EKC_NUM_LOCK },
		{ "SCROLL_LOCK", nbl::ui::E_KEY_CODE::EKC_SCROLL_LOCK },
		{ "VOLUME_MUTE", nbl::ui::E_KEY_CODE::EKC_VOLUME_MUTE },
		{ "VOLUME_UP", nbl::ui::E_KEY_CODE::EKC_VOLUME_UP },
		{ "VOLUME_DOWN", nbl::ui::E_KEY_CODE::EKC_VOLUME_DOWN }
	});

	// one table for both directions so the two mappings cannot drift apart
	static constexpr auto NamedMouseButtons = std::to_array<SNamedCode<nbl::ui::E_MOUSE_BUTTON>>({
		{ "LEFT_BUTTON", nbl::ui::EMB_LEFT_BUTTON },
		{ "RIGHT_BUTTON", nbl::ui::EMB_RIGHT_BUTTON },
		{ "MIDDLE_BUTTON", nbl::ui::EMB_MIDDLE_BUTTON },
		{ "BUTTON_4", nbl::ui::EMB_BUTTON_4 },
		{ "BUTTON_5", nbl::ui::EMB_BUTTON_5 }
	});
};

#endif // _NBL_THIS_EXAMPLE_INPUT_CODE_NAMES_HPP_INCLUDED_
