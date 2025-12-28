#ifndef _THIS_EXAMPLE_APP_HPP_
#define _THIS_EXAMPLE_APP_HPP_

// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/examples/examples.hpp"
#include "nbl/examples/common/SimpleWindowedApplication.hpp"
#include "nbl/examples/common/CSwapchainFramebuffersAndDepth.hpp"
#include "nbl/examples/common/CEventCallback.hpp"
#include "nbl/examples/common/InputSystem.hpp"
#include "nbl/ui/ICursorControl.h"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"
#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "IES.hpp"
#include "CSimpleIESRenderer.hpp"
#include <utility>


NBL_EXPOSE_NAMESPACES

namespace nbl::ext::imgui
{
    class UI;
}

template<typename T>
concept AppIESByteCount = std::unsigned_integral<T>;

template<typename T>
concept AppIESContainer = std::ranges::sized_range<T> &&
    (std::same_as<std::ranges::range_value_t<T>, float> ||
     std::same_as<std::ranges::range_value_t<T>, IESTextureInfo>);
static_assert(alignof(IESTextureInfo) == 4u, "IESTextureInfo must be 4 byte aligned");

template<typename T>
concept AppIESBufferCreationAllowed = AppIESByteCount<T> || AppIESContainer<T>;

class IESWindowedApplication : public virtual SimpleWindowedApplication
{
    using base_t = SimpleWindowedApplication;

public:
    constexpr static inline uint8_t MaxFramesInFlight = 3;

    template<typename... Args>
    IESWindowedApplication(const hlsl::uint16_t2 _initialResolution, const asset::E_FORMAT _depthFormat, Args&&... args) :
        base_t(std::forward<Args>(args)...), m_initialResolution(_initialResolution), m_depthFormat(_depthFormat) {}

    using surface_list_t = decltype(std::declval<const base_t&>().getSurfaces());

    inline surface_list_t getSurfaces() const override
    {
        if (!m_surface)
        {
            auto windowCallback = make_smart_refctd_ptr<examples::CEventCallback>(smart_refctd_ptr(m_inputSystem), smart_refctd_ptr(m_logger));
            IWindow::SCreationParams params = {};
            params.callback = make_smart_refctd_ptr<video::ISimpleManagedSurface::ICallback>();
            params.width = m_initialResolution[0];
            params.height = m_initialResolution[1];
            params.x = 32;
            params.y = 32;
            params.flags = ui::IWindow::ECF_HIDDEN | IWindow::ECF_CAN_MINIMIZE | IWindow::ECF_CAN_MAXIMIZE | IWindow::ECF_CAN_RESIZE;
            params.windowCaption = "IESViewer";
            params.callback = windowCallback;
            const_cast<std::remove_const_t<decltype(m_window)>&>(m_window) = m_winMgr->createWindow(std::move(params));
        }

        auto surface = CSurfaceVulkanWin32::create(smart_refctd_ptr(m_api), smart_refctd_ptr_static_cast<IWindowWin32>(m_window));
        const_cast<std::remove_const_t<decltype(m_surface)>&>(m_surface) = CSimpleResizeSurface<CSwapchainFramebuffersAndDepth>::create(std::move(surface));

        if (m_surface)
            return { {m_surface->getSurface()} };

        return {};
    }

    inline bool onAppInitialized(core::smart_refctd_ptr<system::ISystem>&& system) override
    {
        using namespace nbl::core;
        using namespace nbl::video;
        if (!MonoSystemMonoLoggerApplication::onAppInitialized(std::move(system)))
            return false;

        m_inputSystem = make_smart_refctd_ptr<InputSystem>(system::logger_opt_smart_ptr(smart_refctd_ptr(m_logger)));
        if (!base_t::onAppInitialized(std::move(system)))
            return false;

        ISwapchain::SCreationParams swapchainParams = { .surface = smart_refctd_ptr<ISurface>(m_surface->getSurface()) };
        if (!swapchainParams.deduceFormat(m_physicalDevice))
            return logFail("Could not choose a Surface Format for the Swapchain!");

        auto scResources = std::make_unique<CSwapchainFramebuffersAndDepth>(m_device.get(), m_depthFormat, swapchainParams.surfaceFormat.format, getDefaultSubpassDependencies());
        auto* renderpass = scResources->getRenderpass();

        if (!renderpass)
            return logFail("Failed to create Renderpass!");

        auto gQueue = getGraphicsQueue();
        if (!m_surface || !m_surface->init(gQueue, std::move(scResources), swapchainParams.sharedParams))
            return logFail("Could not create Window & Surface or initialize the Surface!");

        return true;
    }

    inline void workLoopBody() override final
    {
        using namespace nbl::core;
        using namespace nbl::video;
        if (m_window && m_surface && !m_window->isMinimized())
        {
            if (auto* scRes = m_surface->getSwapchainResources())
            {
                if (auto* sc = scRes->getSwapchain())
                {
                    const auto& params = sc->getCreationParameters().sharedParams;
                    if (params.width != m_window->getWidth() || params.height != m_window->getHeight())
                    {
                        m_surface->recreateSwapchain();
                        return;
                    }
                }
            }
        }

        const uint32_t framesInFlightCount = hlsl::min(MaxFramesInFlight, m_surface->getMaxAcquiresInFlight());
        if (m_framesInFlight.size() >= framesInFlightCount)
        {
            const ISemaphore::SWaitInfo framesDone[] =
            {
                {
                    .semaphore = m_framesInFlight.front().semaphore.get(),
                    .value = m_framesInFlight.front().value
                }
            };
            if (m_device->blockForSemaphores(framesDone) != ISemaphore::WAIT_RESULT::SUCCESS)
                return;
            m_framesInFlight.pop_front();
        }

        auto updatePresentationTimestamp = [&]()
        {
            m_currentImageAcquire = m_surface->acquireNextImage();

            oracle.reportEndFrameRecord();
            const auto timestamp = oracle.getNextPresentationTimeStamp();
            oracle.reportBeginFrameRecord();

            return timestamp;
        };

        const auto nextPresentationTimestamp = updatePresentationTimestamp();

        if (!m_currentImageAcquire)
            return;

        const IQueue::SSubmitInfo::SSemaphoreInfo rendered[] = { renderFrame(nextPresentationTimestamp) };
        m_surface->present(m_currentImageAcquire.imageIndex, rendered);
        if (rendered->semaphore)
            m_framesInFlight.emplace_back(smart_refctd_ptr<ISemaphore>(rendered->semaphore), rendered->value);
    }

    inline bool keepRunning() override final
    {
        if (m_surface->irrecoverable())
            return false;

        return true;
    }

    inline bool onAppTerminated() override
    {
        m_inputSystem = nullptr;
        m_device->waitIdle();
        m_framesInFlight.clear();
        m_surface = nullptr;
        m_window = nullptr;
        return base_t::onAppTerminated();
    }

protected:
    inline void onAppInitializedFinish()
    {
        m_winMgr->show(m_window.get());
        oracle.reportBeginFrameRecord();
    }
    inline const auto& getCurrentAcquire() const { return m_currentImageAcquire; }

    virtual const video::IGPURenderpass::SCreationParams::SSubpassDependency* getDefaultSubpassDependencies() const = 0;
    virtual video::IQueue::SSubmitInfo::SSemaphoreInfo renderFrame(const std::chrono::microseconds nextPresentationTimestamp) = 0;

    const hlsl::uint16_t2 m_initialResolution;
    const asset::E_FORMAT m_depthFormat;
    core::smart_refctd_ptr<InputSystem> m_inputSystem;
    core::smart_refctd_ptr<ui::IWindow> m_window;
    core::smart_refctd_ptr<video::CSimpleResizeSurface<CSwapchainFramebuffersAndDepth>> m_surface;

private:
    struct SSubmittedFrame
    {
        core::smart_refctd_ptr<video::ISemaphore> semaphore;
        uint64_t value;
    };
    core::deque<SSubmittedFrame> m_framesInFlight;
    video::ISimpleManagedSurface::SAcquireResult m_currentImageAcquire = {};
    video::CDumbPresentationOracle oracle;
};

class IESViewer final : public IESWindowedApplication, public BuiltinResourcesApplication
{
    using device_base_t = IESWindowedApplication;
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
    smart_refctd_ptr<CSimpleIESRenderer> m_renderer;
    Camera camera;
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
    bool m_showHints = true;
    bool m_plot2DRectValid = false;
    hlsl::float32_t2 m_plot2DRectMin = hlsl::float32_t2(0.f, 0.f);
    hlsl::float32_t2 m_plot2DRectMax = hlsl::float32_t2(0.f, 0.f);
    std::vector<std::string> m_assetLabels;
    std::vector<bool> m_candelaDirty;

    InputSystem::ChannelReader<IMouseEventChannel> mouse;
    InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

    struct {
        smart_refctd_ptr<core::IReferenceCounted> it;
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
