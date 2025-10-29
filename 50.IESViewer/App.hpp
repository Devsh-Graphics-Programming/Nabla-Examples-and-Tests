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

NBL_EXPOSE_NAMESPACES

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
    std::array<core::smart_refctd_ptr<IGPUFramebuffer>, device_base_t::MaxFramesInFlight> m_frameBuffers2D, m_frameBuffers3D;

    smart_refctd_ptr<CGeometryCreatorScene> m_scene;
    smart_refctd_ptr<CSimpleDebugRenderer> m_renderer; // TODO: will need to derive from it + have my own pixel shader
    Camera camera = Camera(core::vectorSIMDf(0, 0, 0), core::vectorSIMDf(0, 0, 0), core::matrix4SIMD()); // TODO: orbit would be better

    InputSystem::ChannelReader<IMouseEventChannel> mouse;
    InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

    struct {
        smart_refctd_ptr<ext::imgui::UI> it;
        smart_refctd_ptr<SubAllocatedDescriptorSet> descriptor;
    } ui;

    void processMouse(const IMouseEventChannel::range_t& events);
    void processKeyboard(const IKeyboardEventChannel::range_t& events);

    smart_refctd_ptr<IGPUImageView> createImageView(const size_t width, const size_t height, E_FORMAT format, std::string name, 
        bitflag<IImage::E_USAGE_FLAGS> usage = bitflag(IImage::EUF_SAMPLED_BIT) | IImage::EUF_STORAGE_BIT,
        bitflag<IImage::E_ASPECT_FLAGS> aspectFlags = bitflag(IImage::EAF_COLOR_BIT));
    smart_refctd_ptr<IGPUBuffer> createBuffer(const core::vector<CIESProfile::IES_STORAGE_FORMAT>& in, std::string name);

    void uiListener();
};

#endif // _THIS_EXAMPLE_APP_HPP_