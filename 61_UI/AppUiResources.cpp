#include "app/App.hpp"
#include "app/AppResourceUtilities.hpp"

template<size_t Count>
struct SUiSampledDescriptorWrites final
{
	std::array<IGPUDescriptorSet::SDescriptorInfo, Count> descriptorInfo = {};
	std::array<IGPUDescriptorSet::SWriteDescriptorSet, Count> writes = {};
};

template<size_t Count>
inline void finalizeUiSampledDescriptorWrites(SUiSampledDescriptorWrites<Count>& output)
{
	for (uint32_t descriptorIx = 0u; descriptorIx < output.writes.size(); ++descriptorIx)
		output.writes[descriptorIx].info = output.descriptorInfo.data() + descriptorIx;
}

template<size_t Count>
inline SUiSampledDescriptorWrites<Count> buildUiSampledDescriptorWrites(
	nbl::ext::imgui::UI& uiManager,
	IGPUDescriptorSet* descriptorSet,
	std::span<const SWindowControlBinding> windowBindings)
{
	SUiSampledDescriptorWrites<Count> output = {};
	const auto fallbackView = core::smart_refctd_ptr<nbl::video::IGPUImageView>(uiManager.getFontAtlasView());

	for (uint32_t descriptorIx = 0u; descriptorIx < output.descriptorInfo.size(); ++descriptorIx)
	{
		output.descriptorInfo[descriptorIx].info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
		output.descriptorInfo[descriptorIx].desc = fallbackView;
	}

	output.descriptorInfo[nbl::ext::imgui::UI::FontAtlasTexId].desc = fallbackView;

	for (uint32_t windowIx = 0u; windowIx < windowBindings.size(); ++windowIx)
	{
		const uint32_t textureIx = SCameraAppUiTextureSlots::viewport(windowIx);
		output.descriptorInfo[textureIx].desc =
			static_cast<bool>(windowBindings[windowIx].sceneColorView) ?
			windowBindings[windowIx].sceneColorView :
			fallbackView;
	}

	for (uint32_t descriptorIx = 0u; descriptorIx < output.writes.size(); ++descriptorIx)
	{
		output.writes[descriptorIx].dstSet = descriptorSet;
		output.writes[descriptorIx].binding = 0u;
		output.writes[descriptorIx].arrayElement = descriptorIx;
		output.writes[descriptorIx].count = 1u;
	}

	return output;
}

inline IDescriptorPool::SCreateInfo buildUiDescriptorPoolInfo(const uint32_t imageCount)
{
	IDescriptorPool::SCreateInfo descriptorPoolInfo = {};
	descriptorPoolInfo.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_SAMPLER)] =
		static_cast<uint32_t>(nbl::ext::imgui::UI::DefaultSamplerIx::COUNT);
	descriptorPoolInfo.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_SAMPLED_IMAGE)] =
		imageCount;
	descriptorPoolInfo.maxSets = 1u;
	descriptorPoolInfo.flags = IDescriptorPool::E_CREATE_FLAGS::ECF_UPDATE_AFTER_BIND_BIT;
	return descriptorPoolInfo;
}

template<size_t WindowCount>
inline void initializeViewportLayoutFromDisplaySize(
	SAppWindowInitState<WindowCount>& windowInit,
	const float32_t2& displaySize)
{
	windowInit.trsEditor.iPos = SCameraAppViewportDefaults::WindowPaddingOffset;
	windowInit.trsEditor.iSize = { 0.0f, displaySize.y - windowInit.trsEditor.iPos.y * 2 };

	const float panelWidth = std::clamp(
		displaySize.x * SCameraAppViewportLayoutDefaults::ControlPanelWidthRatio,
		SCameraAppViewportLayoutDefaults::ControlPanelMinWidth,
		displaySize.x * SCameraAppViewportLayoutDefaults::ControlPanelMaxWidthRatio);
	windowInit.planars.iSize = { panelWidth, displaySize.y - SCameraAppViewportDefaults::WindowPaddingOffset.y * 2 };
	windowInit.planars.iPos = {
		displaySize.x - windowInit.planars.iSize.x - SCameraAppViewportDefaults::WindowPaddingOffset.x,
		SCameraAppViewportDefaults::WindowPaddingOffset.y
	};

	const float leftX = SCameraAppViewportLayoutDefaults::RenderPaddingX;
	const float splitGap = SCameraAppViewportLayoutDefaults::SplitGap;
	const float eachXSize = std::max(0.0f, displaySize.x - leftX * 2.0f);
	const float eachYSize =
		(displaySize.y - SCameraAppViewportLayoutDefaults::RenderPaddingY * 2.0f - (windowInit.renderWindows.size() - 1u) * splitGap) /
		windowInit.renderWindows.size();

	for (size_t windowIx = 0u; windowIx < windowInit.renderWindows.size(); ++windowIx)
	{
		auto& renderWindow = windowInit.renderWindows[windowIx];
		renderWindow.iPos = {
			leftX,
			SCameraAppViewportLayoutDefaults::RenderPaddingY + windowIx * (eachYSize + splitGap)
		};
		renderWindow.iSize = { eachXSize, eachYSize };
	}
}

bool App::updateGUIDescriptorSet()
{
	auto sampledWrites = buildUiSampledDescriptorWrites<TotalUISampleTexturesAmount>(
		*m_ui.manager,
		m_ui.descriptorSet.get(),
		m_viewports.windowBindings);
	finalizeUiSampledDescriptorWrites(sampledWrites);
	return m_device->updateDescriptorSets(sampledWrites.writes, {});
}

bool App::initializeUiResources()
{
	nbl::ext::imgui::UI::SCreationParameters params;
	params.resources.texturesInfo = { .setIx = 0u, .bindingIx = 0u };
	params.resources.samplersInfo = { .setIx = 0u, .bindingIx = 1u };
	params.assetManager = m_assetMgr;
	params.pipelineCache = nullptr;
	params.pipelineLayout = nbl::ext::imgui::UI::createDefaultPipelineLayout(m_utils->getLogicalDevice(), params.resources.texturesInfo, params.resources.samplersInfo, TotalUISampleTexturesAmount);
	params.renderpass = smart_refctd_ptr<IGPURenderpass>(m_renderpass);
	params.subpassIx = 0u;
	params.transfer = getTransferUpQueue();
	params.utilities = m_utils;

	const auto vertexKey = nbl::this_example::builtin::build::get_spirv_key<"imgui_vertex">(m_device.get());
	const auto fragmentKey = nbl::this_example::builtin::build::get_spirv_key<"imgui_fragment">(m_device.get());
	auto vertexShader = nbl::system::loadPrecompiledShaderFromAppResources(*m_assetMgr, m_logger.get(), vertexKey);
	auto fragmentShader = nbl::system::loadPrecompiledShaderFromAppResources(*m_assetMgr, m_logger.get(), fragmentKey);
	if (!vertexShader || !fragmentShader)
		return logFail("Failed to load precompiled ImGui shaders.");

	params.spirv = nbl::ext::imgui::UI::SCreationParameters::PrecompiledShaders{
		.vertex = std::move(vertexShader),
		.fragment = std::move(fragmentShader)
	};

	m_ui.manager = nbl::ext::imgui::UI::create(std::move(params));
	if (!m_ui.manager)
		return false;

	const auto* descriptorSetLayout = m_ui.manager->getPipeline()->getLayout()->getDescriptorSetLayout(0u);
	m_descriptorSetPool = m_device->createDescriptorPool(buildUiDescriptorPoolInfo(TotalUISampleTexturesAmount));
	assert(m_descriptorSetPool);

	m_descriptorSetPool->createDescriptorSets(1u, &descriptorSetLayout, &m_ui.descriptorSet);
	assert(m_ui.descriptorSet);

	m_ui.manager->registerListener([this]() -> void { imguiListen(); });

	const auto displaySize = float32_t2{ m_window->getWidth(), m_window->getHeight() };
	initializeViewportLayoutFromDisplaySize(m_viewports.windowInit, displaySize);

	return true;
}
