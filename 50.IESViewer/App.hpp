#ifndef _THIS_EXAMPLE_APP_HPP_
#define _THIS_EXAMPLE_APP_HPP_

// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/examples/examples.hpp"
#include "nbl/ui/ICursorControl.h"
#include "nbl/ext/ImGui/ImGui.h"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"
#include "IES.hpp"
#include "CSimpleIESRenderer.hpp"

// 3D plot only, full window render and no imgui
// #define DEBUG_SWPCHAIN_FRAMEBUFFERS_ONLY

NBL_EXPOSE_NAMESPACES

template<typename T>
concept AppIESByteCount = std::unsigned_integral<T>;

template<typename T>
concept AppIESContainer = std::ranges::sized_range<T> &&
    (std::same_as<std::ranges::range_value_t<T>, float> ||
     std::same_as<std::ranges::range_value_t<T>, IESTextureInfo>);
static_assert(alignof(IESTextureInfo) == 4u, "IESTextureInfo must be 4 byte aligned");

template<typename T>
concept AppIESBufferCreationAllowed = AppIESByteCount<T> || AppIESContainer<T>;

class IESViewer final : public MonoWindowApplication, public BuiltinResourcesApplication
{
    using device_base_t = MonoWindowApplication;
    using asset_base_t = BuiltinResourcesApplication;

public:
    IESViewer(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD);

    bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override;
    IQueue::SSubmitInfo::SSemaphoreInfo renderFrame(const std::chrono::microseconds nextPresentationTimestamp) override;

protected:
    const IGPURenderpass::SCreationParams::SSubpassDependency* getDefaultSubpassDependencies() const override;

private:
    smart_refctd_ptr<IGPUGraphicsPipeline> m_graphicsPipeline;
    smart_refctd_ptr<IGPUComputePipeline> m_computePipeline;
    std::array<smart_refctd_ptr<IGPUDescriptorSet>, IGPUPipelineLayout::DESCRIPTOR_SET_COUNT> m_descriptors;

    bool m_running = true;
    std::vector<IES> m_assets;
    size_t m_activeAssetIx = 0;

    size_t m_realFrameIx = 0;
    smart_refctd_ptr<ISemaphore> m_semaphore;
    std::array<smart_refctd_ptr<IGPUCommandBuffer>, device_base_t::MaxFramesInFlight> m_cmdBuffers;

#ifndef DEBUG_SWPCHAIN_FRAMEBUFFERS_ONLY
    std::array<core::smart_refctd_ptr<IGPUFramebuffer>, device_base_t::MaxFramesInFlight> m_frameBuffers2D, m_frameBuffers3D;
#endif

    smart_refctd_ptr<CGeometryCreatorScene> m_scene;
    smart_refctd_ptr<CSimpleIESRenderer> m_renderer;
    Camera camera = Camera(core::vectorSIMDf(0, 0, 0), core::vectorSIMDf(0, 0, 0), core::matrix4SIMD()); // TODO: orbit would be better
    uint32_t m_plot3DWidth = 640u;
    uint32_t m_plot3DHeight = 640u;
    float m_plotRadius = 100.0f;
    float m_cameraMoveSpeed = 1.0f;
    float m_cameraRotateSpeed = 1.0f;
    float m_cameraFovDeg = 60.0f;
    bool m_cameraControlEnabled = false;
    bool m_cameraControlApplied = false;
    bool m_fullscreen3D = false;
    bool m_wireframeEnabled = false;
    bool m_showOctaMapPreview = true;
    std::vector<std::string> m_assetLabels;
    std::vector<bool> m_candelaDirty;

    InputSystem::ChannelReader<IMouseEventChannel> mouse;
    InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

    struct {
        smart_refctd_ptr<ext::imgui::UI> it;
        smart_refctd_ptr<SubAllocatedDescriptorSet> descriptor;
    } ui;

	struct {
		IES::E_MODE view = IES::EM_CDC;
		bitflag<this_example::ies::E_SPHERE_MODE> sphere =
			bitflag<this_example::ies::E_SPHERE_MODE>(this_example::ies::ESM_OCTAHEDRAL_UV_INTERPOLATE) | this_example::ies::ESM_FALSE_COLOR;
	} mode;

    void processMouse(const IMouseEventChannel::range_t& events);
    void processKeyboard(const IKeyboardEventChannel::range_t& events);

    smart_refctd_ptr<IGPUImageView> createImageView(const size_t width, const size_t height, E_FORMAT format, std::string name, 
        bitflag<IImage::E_USAGE_FLAGS> usage = bitflag(IImage::EUF_SAMPLED_BIT) | IImage::EUF_STORAGE_BIT,
        bitflag<IImage::E_ASPECT_FLAGS> aspectFlags = bitflag(IImage::EAF_COLOR_BIT));
    bool recreate3DPlotFramebuffers(uint32_t width, uint32_t height);
    void applyWindowMode();

	template<typename T>
	requires AppIESBufferCreationAllowed<T>
    smart_refctd_ptr<IGPUBuffer> createBuffer(const T& in, std::string name, bool unmap = true)
	{
		const void* src = nullptr; size_t bytes = {};
		if constexpr (AppIESByteCount<T>)
			bytes = static_cast<size_t>(in);
		else if (AppIESContainer<T>)
		{
			using element_t = std::ranges::range_value_t<T>;
			static_assert(alignof(element_t) == 4u, "IESViewer::createBuffer: AppIESContainer<T>'s \"T\" must be 4 byte aligned");
			bytes = sizeof(element_t) * in.size();
			src = static_cast<const void*>(std::data(in));
		}
		return implCreateBuffer(src, bytes, name, unmap);
	}
	smart_refctd_ptr<IGPUBuffer> implCreateBuffer(const void* src, size_t bytes, const std::string& name, bool unmap);

    void uiListener();
};

#endif // _THIS_EXAMPLE_APP_HPP_
