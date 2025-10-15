// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/examples/examples.hpp"
#include "nbl/this_example/builtin/build/spirv/keys.hpp"
#include "AppInputParser.hpp"

using namespace nbl;
using namespace core;
using namespace hlsl;
using namespace system;
using namespace asset;
using namespace ui;
using namespace video;
using namespace scene;
using namespace nbl::examples;

#define BENCHMARK_TILL_FIRST_FRAME

#ifdef BENCHMARK_TILL_FIRST_FRAME
const std::chrono::steady_clock::time_point startBenchmark = std::chrono::high_resolution_clock::now();
bool stopBenchamrkFlag = false;
#endif

class IESViewer final : public MonoWindowApplication, public BuiltinResourcesApplication
{
    using device_base_t = MonoWindowApplication;
    using asset_base_t = BuiltinResourcesApplication;

public:
    IESViewer(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
        : IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD),
        device_base_t({ 1280,720 }, EF_D16_UNORM, _localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {
    }

    inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
    {
        if (!asset_base_t::onAppInitialized(smart_refctd_ptr(system)))
            return false;
        if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
            return false;

        AppInputParser::Output out;
        AppInputParser parser(system::logger_opt_ptr(m_logger.get()));
        if (!parser.parse(out, "../inputs.json", "../../media"))
            return false;

        std::vector<asset::SAssetBundle> assets;
        {
            m_logger->log("Loading IES assets..", system::ILogger::ELL_INFO);
            auto start = std::chrono::high_resolution_clock::now();
            size_t loaded = {}, total = out.inputList.size();
            IAssetLoader::SAssetLoadParams lp = {};
            lp.logger = system::logger_opt_ptr(m_logger.get());

            for (const auto& in : out.inputList)
            {
                auto asset = m_assetMgr->getAsset(in.c_str(), lp);

                if (asset.getMetadata())
                {
                    assets.emplace_back(std::move(asset));
                    ++loaded;

                    m_logger->log("Loaded \"%s\".", system::ILogger::ELL_INFO, in.c_str());
                }
                else
                    m_logger->log("Failed to load metadata for \"%s\"! Skipping..", system::ILogger::ELL_WARNING, in.c_str());
            }
            const auto sl = std::to_string(loaded), st = std::to_string(total);
            const bool passed = loaded == total;

            if (not passed)
            {
                auto diff = std::to_string(total - loaded);
                m_logger->log("Failed to load [%s/%s] IES assets!", system::ILogger::ELL_ERROR, diff.c_str(), st.c_str());
            }
            auto elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start);
            auto took = std::to_string(elapsed.count());
            m_logger->log("Finished loading IES assets, took %s seconds.", system::ILogger::ELL_PERFORMANCE, took.c_str());
        }

        auto createShader = [&]<core::StringLiteral in>() -> smart_refctd_ptr<IShader>
        {
            IAssetLoader::SAssetLoadParams lp = {};
            lp.logger = system::logger_opt_ptr(m_logger.get());
            lp.workingDirectory = "app_resources";

            auto key = nbl::this_example::builtin::build::get_spirv_key<in>(m_device.get());
            auto assetBundle = m_assetMgr->getAsset(key, lp);
            const auto assets = assetBundle.getContents();

            if (assets.empty())
            {
                m_logger->log("Failed to load \"%s\" shader!", system::ILogger::ELL_ERROR, key.data());
                return nullptr;
            }

            auto spirvShader = IAsset::castDown<IShader>(assets[0]);

            if (spirvShader)
                m_logger->log("Loaded \"%s\".", system::ILogger::ELL_INFO, key.data());
            else
                m_logger->log("Failed to cast \"%s\" asset to IShader!", system::ILogger::ELL_ERROR, key.data());

            return spirvShader;
        };

        #define CREATE_SHADER(SHADER, PATH) \
		if (!(SHADER = createShader.template operator()<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE(PATH)>() )) return false;

        smart_refctd_ptr<IShader> compute, pixel, vertex;
        {
            m_logger->log("Loading GPU shaders..", system::ILogger::ELL_INFO);
            auto start = std::chrono::high_resolution_clock::now();
            CREATE_SHADER(compute, "compute")
            CREATE_SHADER(pixel, "pixel")
            CREATE_SHADER(vertex, "vertex")
            auto elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start);
            auto took = std::to_string(elapsed.count());
            m_logger->log("Finished loading GPU shaders, took %s seconds!", system::ILogger::ELL_PERFORMANCE, took.c_str());
        }

        return true;
    }

    inline IQueue::SSubmitInfo::SSemaphoreInfo renderFrame(const std::chrono::microseconds nextPresentationTimestamp) override
    {
        m_inputSystem->getDefaultMouse(&mouse);
        m_inputSystem->getDefaultKeyboard(&keyboard);

        const auto resourceIx = m_realFrameIx % device_base_t::MaxFramesInFlight;

        auto* const cb = m_cmdBufs.data()[resourceIx].get();
        cb->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
        cb->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
        cb->beginDebugMarker("IESViewer Frame");
        {
            camera.beginInputProcessing(nextPresentationTimestamp);
            mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { camera.mouseProcess(events); mouseProcess(events); }, m_logger.get());
            keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void { camera.keyboardProcess(events); }, m_logger.get());
            camera.endInputProcessing(nextPresentationTimestamp);
        }


        asset::SViewport viewport;
        {
            viewport.minDepth = 1.f;
            viewport.maxDepth = 0.f;
            viewport.x = 0u;
            viewport.y = 0u;
            viewport.width = m_window->getWidth();
            viewport.height = m_window->getHeight();
        }
        cb->setViewport(0u, 1u, &viewport);

        VkRect2D scissor =
        {
            .offset = { 0, 0 },
            .extent = { m_window->getWidth(), m_window->getHeight() },
        };
        cb->setScissor(0u, 1u, &scissor);

        {
            const VkRect2D currentRenderArea =
            {
                .offset = {0,0},
                .extent = {m_window->getWidth(),m_window->getHeight()}
            };

            const IGPUCommandBuffer::SClearColorValue clearValue = { .float32 = {1.f,0.f,1.f,1.f} };
            const IGPUCommandBuffer::SClearDepthStencilValue depthValue = { .depth = 0.f };
            auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
            const IGPUCommandBuffer::SRenderpassBeginInfo info =
            {
                .framebuffer = scRes->getFramebuffer(device_base_t::getCurrentAcquire().imageIndex),
                .colorClearValues = &clearValue,
                .depthStencilClearValues = &depthValue,
                .renderArea = currentRenderArea
            };

            cb->beginRenderPass(info, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
        }

        float32_t3x4 viewMatrix;
        float32_t4x4 viewProjMatrix;
        // TODO: get rid of legacy matrices
        {
            memcpy(&viewMatrix, camera.getViewMatrix().pointer(), sizeof(viewMatrix));
            memcpy(&viewProjMatrix, camera.getConcatenatedMatrix().pointer(), sizeof(viewProjMatrix));
        }
        const auto viewParams = CSimpleDebugRenderer::SViewParams(viewMatrix, viewProjMatrix);

        // tear down scene every frame
        //m_renderer->m_instances[0].packedGeo = m_renderer->getGeometries().data() + gcIndex;
        //m_renderer->render(cb, viewParams);

        cb->endRenderPass();
        cb->endDebugMarker();
        cb->end();

        IQueue::SSubmitInfo::SSemaphoreInfo retval =
        {
            .semaphore = m_semaphore.get(),
            .value = ++m_realFrameIx,
            .stageMask = PIPELINE_STAGE_FLAGS::ALL_GRAPHICS_BITS
        };
        const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[] =
        {
            {.cmdbuf = cb }
        };
        const IQueue::SSubmitInfo::SSemaphoreInfo acquired[] = {
            {
                .semaphore = device_base_t::getCurrentAcquire().semaphore,
                .value = device_base_t::getCurrentAcquire().acquireCount,
                .stageMask = PIPELINE_STAGE_FLAGS::NONE
            }
        };
        const IQueue::SSubmitInfo infos[] =
        {
            {
                .waitSemaphores = acquired,
                .commandBuffers = commandBuffers,
                .signalSemaphores = {&retval,1}
            }
        };

        if (getGraphicsQueue()->submit(infos) != IQueue::RESULT::SUCCESS)
        {
            retval.semaphore = nullptr; // so that we don't wait on semaphore that will never signal
            m_realFrameIx--;
        }

        std::string caption = "[Nabla Engine] IES Viewer";
        {
            caption += ", displaying [";
            //caption += m_scene->getInitParams().geometryNames[gcIndex];
            caption += "]";
            m_window->setCaption(caption);
        }
        return retval;
    }

protected:
    const video::IGPURenderpass::SCreationParams::SSubpassDependency* getDefaultSubpassDependencies() const override
    {
        // Subsequent submits don't wait for each other, hence its important to have External Dependencies which prevent users of the depth attachment overlapping.
        const static IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] = {
            // wipe-transition of Color to ATTACHMENT_OPTIMAL and depth
            {
                .srcSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
                .dstSubpass = 0,
                .memoryBarrier = {
                // last place where the depth can get modified in previous frame, `COLOR_ATTACHMENT_OUTPUT_BIT` is implicitly later
                .srcStageMask = PIPELINE_STAGE_FLAGS::LATE_FRAGMENT_TESTS_BIT,
                // don't want any writes to be available, we'll clear 
                .srcAccessMask = ACCESS_FLAGS::NONE,
                // destination needs to wait as early as possible
                // TODO: `COLOR_ATTACHMENT_OUTPUT_BIT` shouldn't be needed, because its a logically later stage, see TODO in `ECommonEnums.h`
                .dstStageMask = PIPELINE_STAGE_FLAGS::EARLY_FRAGMENT_TESTS_BIT | PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
                // because depth and color get cleared first no read mask
                .dstAccessMask = ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
            }
            // leave view offsets and flags default
        },
            // color from ATTACHMENT_OPTIMAL to PRESENT_SRC
            {
                .srcSubpass = 0,
                .dstSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
                .memoryBarrier = {
                // last place where the color can get modified, depth is implicitly earlier
                .srcStageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
                // only write ops, reads can't be made available
                .srcAccessMask = ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
                // spec says nothing is needed when presentation is the destination
            }
            // leave view offsets and flags default
        },
        IGPURenderpass::SCreationParams::DependenciesEnd
        };
        return dependencies;
    }

private:
    //
    //smart_refctd_ptr<CGeometryCreatorScene> m_scene;
    //smart_refctd_ptr<CSimpleDebugRenderer> m_renderer;
    //
    smart_refctd_ptr<ISemaphore> m_semaphore;
    uint64_t m_realFrameIx = 0;
    std::array<smart_refctd_ptr<IGPUCommandBuffer>, device_base_t::MaxFramesInFlight> m_cmdBufs;
    //
    InputSystem::ChannelReader<IMouseEventChannel> mouse;
    InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

    //
    Camera camera = Camera(core::vectorSIMDf(0, 0, 0), core::vectorSIMDf(0, 0, 0), core::matrix4SIMD());

    uint16_t gcIndex = {};

    void mouseProcess(const nbl::ui::IMouseEventChannel::range_t& events)
    {
        for (auto eventIt = events.begin(); eventIt != events.end(); eventIt++)
        {
            auto ev = *eventIt;

            /*
            if (ev.type == nbl::ui::SMouseEvent::EET_SCROLL && m_renderer)
            {
                gcIndex += int16_t(core::sign(ev.scrollEvent.verticalScroll));
                gcIndex = core::clamp(gcIndex, 0ull, m_renderer->getGeometries().size() - 1);
            }
            */
        }
    }
};

NBL_MAIN_FUNC(IESViewer)