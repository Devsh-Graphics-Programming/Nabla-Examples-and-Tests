// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "gui/CSessionWindow.h"
#include "renderer/CSession.h"
#include "renderer/CScene.h"
#include "renderer/CRenderer.h"
#include "nbl/ext/ImGui/ImGui.h"
#include "nbl/system/to_string.h"

namespace nbl::this_example::gui
{

void CSessionWindow::setSession(CSession* session)
{
	m_session = session;
	if (m_session)
	{
		// Read initial state
		const auto& dynamics = m_session->getActiveResources().currentSensorState;
		m_cachedDynamics = dynamics;
		m_state.tMax = dynamics.tMax;

		// Read render mode from creation params
		const auto& params = m_session->getConstructionParams();
		m_state.renderMode = params.mode;

		m_state.cropWidth = params.cropResolution.x;
		m_state.cropHeight = params.cropResolution.y;
		m_state.cropOffsetX = params.cropOffsets.x;
		m_state.cropOffsetY = params.cropOffsets.y;
	}
}

void CSessionWindow::setBufferTextureIDs(const std::array<uint32_t, static_cast<size_t>(BufferType::Count)>& textureIDs)
{
	m_bufferTextureIDs = textureIDs;
}

void CSessionWindow::draw(const bool forceReposition)
{
	if (!m_isOpen)
		return;

	// Position on LEFT side
	const ImGuiViewport* viewport = ImGui::GetMainViewport();
	if (viewport->WorkSize.x > 64.0f && viewport->WorkSize.y > 64.0f)
	{
		const float windowWidth = 320.0f;
		const ImGuiCond cond = forceReposition ? ImGuiCond_Always : ImGuiCond_Appearing;

		ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + 20, viewport->WorkPos.y + 20), cond);
		ImGui::SetNextWindowSize(ImVec2(windowWidth, viewport->WorkSize.y - 40), cond);
		ImGui::SetNextWindowSizeConstraints(ImVec2(280, 400), ImVec2(FLT_MAX, FLT_MAX));

		if (ImGui::Begin("Session", &m_isOpen))
		{
			if (m_session)
			{
				drawRenderModeSection();
				ImGui::Separator();
				drawDynamicsSection();
				ImGui::Separator();
				drawMutablesSection();
				ImGui::Separator();
				drawOutputBufferSection();
			}
			else
			{
				ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "No active session");
			}
		}
		ImGui::End();
	}
}

void CSessionWindow::drawRenderModeSection()
{
	if (ImGui::CollapsingHeader("Render Mode", ImGuiTreeNodeFlags_DefaultOpen))
	{
		const char* modes[] = { "Previs", "Beauty", "Debug" };
		if (ImGui::Combo("Mode", reinterpret_cast<int*>(&m_state.renderMode), modes, IM_ARRAYSIZE(modes)))
		{
			if (m_callbacks.onRenderModeChanged)
			{
				m_callbacks.onRenderModeChanged(m_state.renderMode, m_session);
			}
		}

		if (m_session)
			ImGui::ProgressBar(m_session->getProgress(), ImVec2(-1, 0), "Progress");
	}
}

void CSessionWindow::drawDynamicsSection()
{
	if (ImGui::CollapsingHeader("Dynamics", ImGuiTreeNodeFlags_DefaultOpen))
	{
		bool changed = false;

		if (ImGui::DragFloat("Crop Offset X", &m_state.cropOffsetX, 1.0f, 0.0f, 10000.0f)) changed = true;
		if (ImGui::DragFloat("Crop Offset Y", &m_state.cropOffsetY, 1.0f, 0.0f, 10000.0f)) changed = true;
		if (ImGui::SliderFloat("T Max", &m_state.tMax, 0.0f, 20000.0f, "%.1f", ImGuiSliderFlags_Logarithmic)) changed = true;

		if (changed && m_callbacks.onDynamicsChanged)
		{
			SSensorDynamics newDynamics = m_cachedDynamics;
			newDynamics.tMax = m_state.tMax;
			// TODO: updated crop offsets when theyre moved too
			m_callbacks.onDynamicsChanged(newDynamics, m_session);
		}
	}
}

void CSessionWindow::drawMutablesSection()
{
	if (ImGui::CollapsingHeader("Mutables"))
	{
		ImGui::InputInt("Crop Width", &m_state.cropWidth);
		ImGui::InputInt("Crop Height", &m_state.cropHeight);

		ImGui::InputFloat("Near Clip", &m_state.nearClip);
		ImGui::InputFloat("Far Clip", &m_state.farClip);

		if (ImGui::Button("Apply Changes"))
		{
			if (m_callbacks.onMutablesChanged)
			{
				SSensorDynamics newDynamics = m_cachedDynamics;
				m_callbacks.onMutablesChanged(newDynamics, m_session);
			}

			if (m_callbacks.onResolutionChanged)
			{
				m_callbacks.onResolutionChanged((uint16_t)m_state.cropWidth, (uint16_t)m_state.cropHeight);
			}
		}

		ImGui::SameLine();
		if (ImGui::Button("Discard"))
		{
			if (m_session)
			{
				const auto& params = m_session->getConstructionParams();
				m_state.cropWidth = params.cropResolution.x;
				m_state.cropHeight = params.cropResolution.y;
			}
		}
	}
}

void CSessionWindow::drawOutputBufferSection()
{
	if (ImGui::CollapsingHeader("Output Buffers", ImGuiTreeNodeFlags_DefaultOpen))
	{
		const auto& resources = m_session->getActiveResources();
		const auto& immutables = resources.immutables;

		// Thumbnail size
		constexpr float thumbnailSize = 80.0f;
		constexpr float padding = 4.0f;

		// Calculate how many thumbnails fit per row
		const float availableWidth = ImGui::GetContentRegionAvail().x;
		const int thumbsPerRow = std::max(1, static_cast<int>((availableWidth + padding) / (thumbnailSize + padding)));

		int currentColumn = 0;

		// Helper to show a buffer thumbnail
		auto showBufferThumbnail = [&](BufferType type, const CSession::SImageWithViews& imageWithViews)
		{
			const int idx = static_cast<int>(type);
			const uint32_t texID = m_bufferTextureIDs[idx];
			const std::string name = nbl::system::to_string(type);
			const bool isValid = imageWithViews && (texID != 0xFFFFFFFF);
			const bool isSelected = (m_state.selectedBufferIndex == idx);

			// Start new line if needed
			if (currentColumn > 0)
				ImGui::SameLine();

			ImGui::BeginGroup();

			// Draw selection highlight
			if (isSelected)
			{
				ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.3f, 0.5f, 0.8f, 1.0f));
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.4f, 0.6f, 0.9f, 1.0f));
			}
			else
			{
				ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.2f, 0.2f, 1.0f));
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.3f, 1.0f));
			}

			bool clicked;
			if (isValid)
			{
				// Create SImResourceInfo for the texture
				SImResourceInfo texInfo;
				texInfo.textureID = texID;
				texInfo.samplerIx = static_cast<uint16_t>(nbl::ext::imgui::UI::DefaultSamplerIx::USER);

				// Use ImageButton for clickable thumbnail
				ImGui::PushID(idx);
				clicked = ImGui::ImageButton(name.c_str(), texInfo, ImVec2(thumbnailSize, thumbnailSize));
				ImGui::PopID();
			}
			else
			{
				// Show placeholder for unavailable buffers
				ImGui::PushID(idx);
				clicked = ImGui::Button("N/A", ImVec2(thumbnailSize, thumbnailSize));
				ImGui::PopID();
			}

			ImGui::PopStyleColor(2);

			// Show label below thumbnail
			ImGui::TextUnformatted(name.c_str());

			ImGui::EndGroup();

			// Handle click
			if (clicked)
			{
				m_state.selectedBufferIndex = idx;
				if (m_callbacks.onBufferSelected)
					m_callbacks.onBufferSelected(idx);
			}

			// Update column counter
			currentColumn++;
			if (currentColumn >= thumbsPerRow)
				currentColumn = 0;
		};

		// Show all buffer thumbnails
		showBufferThumbnail(BufferType::Beauty, immutables.beauty);
		showBufferThumbnail(BufferType::Albedo, immutables.albedo);
		showBufferThumbnail(BufferType::Normal, immutables.normal);
		showBufferThumbnail(BufferType::Motion, immutables.motion);
		showBufferThumbnail(BufferType::Mask, immutables.mask);
		showBufferThumbnail(BufferType::RWMCCascades, immutables.rwmcCascades);
		showBufferThumbnail(BufferType::SampleCount, immutables.sampleCount);

		ImGui::Spacing();
		ImGui::TextDisabled("Click to select buffer for viewing");
	}
}

} // namespace nbl::this_example::gui

namespace nbl::system::impl
{
	template<>
	struct to_string_helper<this_example::gui::CSessionWindow::BufferType>
	{
		static std::string __call(const this_example::gui::CSessionWindow::BufferType& value)
		{
			switch (value)
			{
				case this_example::gui::CSessionWindow::BufferType::Beauty: return "Beauty";
				case this_example::gui::CSessionWindow::BufferType::Albedo: return "Albedo";
				case this_example::gui::CSessionWindow::BufferType::Normal: return "Normal";
				case this_example::gui::CSessionWindow::BufferType::Motion: return "Motion";
				case this_example::gui::CSessionWindow::BufferType::Mask: return "Mask";
				case this_example::gui::CSessionWindow::BufferType::RWMCCascades: return "RWMC Cascades";
				case this_example::gui::CSessionWindow::BufferType::SampleCount: return "Sample Count";
				default: return "Unknown";
			}
		}
	};
}