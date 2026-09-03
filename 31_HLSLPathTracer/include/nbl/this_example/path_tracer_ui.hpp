#ifndef __NBL_THIS_EXAMPLE_PATH_TRACER_UI_HPP_INCLUDED__
#define __NBL_THIS_EXAMPLE_PATH_TRACER_UI_HPP_INCLUDED__

#include "nbl/this_example/common.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>

namespace nbl::this_example::pt_ui
{
struct SFloatSliderRow
{
	const char* label;
	float* value;
	float min;
	float max;
	const char* format;
};

struct SIntSliderRow
{
	const char* label;
	int* value;
	int min;
	int max;
};

struct SCheckboxRow
{
	const char* label;
	bool* value;
};

struct SComboRow
{
	const char* label;
	int* value;
	const char* const* items;
	int count;
};

struct STextRow
{
	const char* label;
	std::string value;
};

template<class Range, class ToText>
inline float calcMaxTextWidth(const Range& items, ToText&& toText)
{
	float width = 0.f;
	for (const auto& item : items)
		width = std::max(width, ImGui::CalcTextSize(toText(item)).x);
	return width;
}

inline std::string makeReadyText(const size_t ready, const size_t total)
{
	return std::to_string(ready) + "/" + std::to_string(total);
}

inline std::string makeRunQueueText(const size_t running, const size_t queued)
{
	return std::to_string(running) + " / " + std::to_string(queued);
}

inline bool beginSectionTable(const char* id)
{
	return ImGui::BeginTable(id, 2, ImGuiTableFlags_SizingFixedFit);
}

inline void setupSectionTable(const float tableLabelColumnWidth)
{
	ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthFixed, tableLabelColumnWidth);
	ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch);
}

inline void sliderFloatRow(const SFloatSliderRow& row)
{
	ImGui::TableNextRow();
	ImGui::TableSetColumnIndex(0);
	ImGui::TextUnformatted(row.label);
	ImGui::TableSetColumnIndex(1);
	ImGui::SetNextItemWidth(-FLT_MIN);
	ImGui::PushID(row.label);
	ImGui::SliderFloat("##value", row.value, row.min, row.max, row.format, ImGuiSliderFlags_AlwaysClamp);
	ImGui::PopID();
}

inline void sliderIntRow(const SIntSliderRow& row)
{
	ImGui::TableNextRow();
	ImGui::TableSetColumnIndex(0);
	ImGui::TextUnformatted(row.label);
	ImGui::TableSetColumnIndex(1);
	ImGui::SetNextItemWidth(-FLT_MIN);
	ImGui::PushID(row.label);
	ImGui::SliderInt("##value", row.value, row.min, row.max);
	ImGui::PopID();
}

inline void comboRow(const SComboRow& row)
{
	ImGui::TableNextRow();
	ImGui::TableSetColumnIndex(0);
	ImGui::TextUnformatted(row.label);
	ImGui::TableSetColumnIndex(1);
	ImGui::SetNextItemWidth(-FLT_MIN);
	ImGui::PushID(row.label);
	ImGui::Combo("##value", row.value, row.items, row.count);
	ImGui::PopID();
}

inline void checkboxRow(const SCheckboxRow& row)
{
	ImGui::TableNextRow();
	ImGui::TableSetColumnIndex(0);
	ImGui::TextUnformatted(row.label);
	ImGui::TableSetColumnIndex(1);
	ImGui::PushID(row.label);
	ImGui::Checkbox("##value", row.value);
	ImGui::PopID();
}

inline void textRow(const STextRow& row)
{
	ImGui::TableNextRow();
	ImGui::TableSetColumnIndex(0);
	ImGui::TextUnformatted(row.label);
	ImGui::TableSetColumnIndex(1);
	ImGui::TextUnformatted(row.value.c_str());
}
}

#endif
