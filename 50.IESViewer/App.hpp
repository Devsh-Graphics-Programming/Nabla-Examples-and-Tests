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
    smart_refctd_ptr<IGPUGraphicsPipeline> graphicsPipeline;
    smart_refctd_ptr<IGPUComputePipeline> computePipeline;
    std::array<smart_refctd_ptr<IGPUDescriptorSet>, IGPUPipelineLayout::DESCRIPTOR_SET_COUNT> descriptors;

    bool running = true;
    std::vector<IES> assets;
    size_t activeAssetIx = 0;

    size_t m_realFrameIx = 0;
    smart_refctd_ptr<ISemaphore> m_semaphore;
    std::array<smart_refctd_ptr<IGPUCommandBuffer>, device_base_t::MaxFramesInFlight> m_cmdBufs;
    InputSystem::ChannelReader<IMouseEventChannel> mouse;
    InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

    struct {
        smart_refctd_ptr<ext::imgui::UI> it;
        smart_refctd_ptr<SubAllocatedDescriptorSet> descriptor;
    } ui;

    void processMouse(const IMouseEventChannel::range_t& events);
    void processKeyboard(const IKeyboardEventChannel::range_t& events);

    smart_refctd_ptr<IGPUImageView> createImageView(const size_t width, const size_t height, E_FORMAT format, std::string name);
    smart_refctd_ptr<IGPUBuffer> createBuffer(const core::vector<CIESProfile::IES_STORAGE_FORMAT>& in, std::string name);

    void uiListener();
};

#endif // _THIS_EXAMPLE_APP_HPP_