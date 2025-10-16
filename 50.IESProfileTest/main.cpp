// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/examples/examples.hpp"
#include "nbl/this_example/builtin/build/spirv/keys.hpp"
#include "app_resources/common.hlsl"
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

constexpr static std::string_view InputsJson = "../inputs.json";
constexpr static std::string_view MediaEntry = "../../media";

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

        const auto media = absolute(path(MediaEntry.data()));

        AppInputParser::Output out;
        AppInputParser parser(system::logger_opt_ptr(m_logger.get()));
        if (!parser.parse(out, InputsJson.data(), media.string()))
            return false;

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
                    auto& ies = assets.emplace_back();
                    ies.bundle = std::move(asset);
                    ies.key = path(in).lexically_relative(media).string();
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

        {
            m_logger->log("Creating GPU IES resources..", system::ILogger::ELL_INFO);
            auto start = std::chrono::high_resolution_clock::now();
            for (auto& ies : assets)
            {
                const auto* profile = ies.getProfile();
                const auto resolution = profile->getOptimalIESResolution();

                #define CREATE_VIEW(VIEW, FORMAT, NAME) \
		        if (!(VIEW = createImageView(resolution.x, resolution.y, FORMAT, NAME + ies.key) )) return false;

                CREATE_VIEW(ies.views.candela, asset::EF_R16_UNORM, "IES Candela Data Image: ")
                CREATE_VIEW(ies.views.spherical, asset::EF_R32G32_SFLOAT, "IES Spherical Data Image: ")
                CREATE_VIEW(ies.views.direction, asset::EF_R32G32B32A32_SFLOAT, "IES Direction Data Image: ")
                CREATE_VIEW(ies.views.mask, asset::EF_R8G8_UNORM, "IES Mask Data Image: ")

                #define CREATE_BUFFER(BUFFER, DATA, NAME) \
		        if (!(BUFFER = createBuffer(DATA, NAME + ies.key) )) return false;

                CREATE_BUFFER(ies.buffers.vAngles, profile->getVertAngles(), "IES Vertical Angles Buffer: ")
                CREATE_BUFFER(ies.buffers.hAngles, profile->getHoriAngles(), "IES Horizontal Angles Buffer: ")
                CREATE_BUFFER(ies.buffers.data, profile->getData(), "IES Data Buffer: ")
            }
            auto elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start);
            auto took = std::to_string(elapsed.count());
            m_logger->log("Finished creating GPU IES resources, took %s seconds.", system::ILogger::ELL_PERFORMANCE, took.c_str());
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

        // Pipelines & Descriptor Sets
        {
            using binding_flags_t = video::IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS;
            using stage_flags_t = asset::IShader::E_SHADER_STAGE;
            static constexpr auto TexturesCreateFlags = core::bitflag(binding_flags_t::ECF_UPDATE_AFTER_BIND_BIT) | binding_flags_t::ECF_PARTIALLY_BOUND_BIT | binding_flags_t::ECF_UPDATE_UNUSED_WHILE_PENDING_BIT;
            static constexpr auto SamplersCreateFlags = core::bitflag(binding_flags_t::ECF_UPDATE_AFTER_BIND_BIT);

            const uint32_t texturesCount = assets.size();
            auto computeBindings = std::to_array<IGPUDescriptorSetLayout::SBinding>
            ({
                {.binding = 0, .type = IDescriptor::E_TYPE::ET_STORAGE_IMAGE, .createFlags = TexturesCreateFlags, .stageFlags = stage_flags_t::ESS_COMPUTE, .count = texturesCount, .immutableSamplers = nullptr},
                {.binding = 1, .type = IDescriptor::E_TYPE::ET_STORAGE_IMAGE, .createFlags = TexturesCreateFlags, .stageFlags = stage_flags_t::ESS_COMPUTE, .count = texturesCount, .immutableSamplers = nullptr},
                {.binding = 2, .type = IDescriptor::E_TYPE::ET_STORAGE_IMAGE, .createFlags = TexturesCreateFlags, .stageFlags = stage_flags_t::ESS_COMPUTE, .count = texturesCount, .immutableSamplers = nullptr},
                {.binding = 3, .type = IDescriptor::E_TYPE::ET_STORAGE_IMAGE, .createFlags = TexturesCreateFlags, .stageFlags = stage_flags_t::ESS_COMPUTE, .count = texturesCount, .immutableSamplers = nullptr}
            });

            auto pixelBindings = std::to_array<IGPUDescriptorSetLayout::SBinding>
            ({
                {.binding = 0, .type = IDescriptor::E_TYPE::ET_SAMPLED_IMAGE, .createFlags = TexturesCreateFlags, .stageFlags = stage_flags_t::ESS_FRAGMENT, .count = texturesCount, .immutableSamplers = nullptr},
                {.binding = 1, .type = IDescriptor::E_TYPE::ET_SAMPLED_IMAGE, .createFlags = TexturesCreateFlags, .stageFlags = stage_flags_t::ESS_FRAGMENT, .count = texturesCount, .immutableSamplers = nullptr},
                {.binding = 2, .type = IDescriptor::E_TYPE::ET_SAMPLED_IMAGE, .createFlags = TexturesCreateFlags, .stageFlags = stage_flags_t::ESS_FRAGMENT, .count = texturesCount, .immutableSamplers = nullptr},
                {.binding = 3, .type = IDescriptor::E_TYPE::ET_SAMPLED_IMAGE, .createFlags = TexturesCreateFlags, .stageFlags = stage_flags_t::ESS_FRAGMENT, .count = texturesCount, .immutableSamplers = nullptr},
                {.binding = 3, .type = IDescriptor::E_TYPE::ET_SAMPLER, .createFlags = SamplersCreateFlags, .stageFlags = stage_flags_t::ESS_FRAGMENT, .count = 1u, .immutableSamplers = nullptr}
            });

            smart_refctd_ptr<IGPUSampler> generalSampler;
            {
                IGPUSampler::SParams params;
                params.AnisotropicFilter = 1u;
                params.TextureWrapU = ISampler::E_TEXTURE_CLAMP::ETC_CLAMP_TO_EDGE;
                params.TextureWrapV = ISampler::E_TEXTURE_CLAMP::ETC_CLAMP_TO_EDGE;
                params.TextureWrapW = ISampler::E_TEXTURE_CLAMP::ETC_CLAMP_TO_EDGE;
                params.BorderColor = ISampler::ETBC_FLOAT_OPAQUE_BLACK;
                params.MinFilter = ISampler::ETF_LINEAR;
                params.MaxFilter = ISampler::ETF_LINEAR;
                params.MipmapMode = ISampler::ESMM_LINEAR;
                params.AnisotropicFilter = 0u;
                params.CompareEnable = false;
                params.CompareFunc = ISampler::ECO_ALWAYS;

                generalSampler = m_device->createSampler(params);

                if (not generalSampler)
                {
                    m_logger->log("Failed to create sampler!", system::ILogger::ELL_ERROR);
                    return false;
                }

                generalSampler->setObjectDebugName("Default IES sampler");
            }

            auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
            scRes->getRenderpass();

            // Graphics Pipeline
            {
                auto descriptorSetLayout = m_device->createDescriptorSetLayout(pixelBindings);

                if(not descriptorSetLayout)
                    return logFail("Failed to create descriptor set layout!");

                auto range = std::to_array<asset::SPushConstantRange>({ {stage_flags_t::ESS_FRAGMENT, 0u, sizeof(PushConstants)} });
                auto graphicsPipelineLayout = m_device->createPipelineLayout(range, nullptr, nullptr, nullptr, core::smart_refctd_ptr(descriptorSetLayout));

                if(not graphicsPipelineLayout)
                    return logFail("Failed to create pipeline layout!");

                video::IGPUPipelineBase::SShaderSpecInfo specInfo[] = 
                {
                    { .shader = vertex.get(), .entryPoint = "VSMain" },
                    { .shader = pixel.get(), .entryPoint = "PSMain" }
                };

                auto params = std::to_array<IGPUGraphicsPipeline::SCreationParams>({ {} });
                params[0].layout = graphicsPipelineLayout.get();
                params[0].cached = {
                    .vertexInput = {},
                    .primitiveAssembly = {
                        .primitiveType = E_PRIMITIVE_TOPOLOGY::EPT_TRIANGLE_LIST,
                    },
                    .rasterization = {
                        .polygonMode = EPM_FILL,
                        .faceCullingMode = EFCM_NONE,
                        .depthWriteEnable = false,
                    },
                    .blend = {}
                };
                params[0].renderpass = scRes->getRenderpass();
                params[0].vertexShader = specInfo[0];
                params[0].fragmentShader = specInfo[1];

                if (!m_device->createGraphicsPipelines(nullptr, params, &graphicsPipeline))
                    return logFail("Failed to create graphics pipeline!");
            }

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
            mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { mouseProcess(events); }, m_logger.get());
            keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void { keyboardProcess(events); }, m_logger.get());
        }

        auto& ies = assets[activeAssetIx];
        PushConstants pc;
        updatePushConstants(pc, ies);

        for (auto& buffer : { ies.buffers.data, ies.buffers.hAngles, ies.buffers.vAngles }) // flush request for sanity
        {
            auto bound = buffer->getBoundMemory();
            if (bound.memory->haveToMakeVisible())
            {
                const ILogicalDevice::MappedMemoryRange range(bound.memory, bound.offset, buffer->getSize());
                m_device->flushMappedMemoryRanges(1, &range);
            }
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
    enum E_MODE : uint32_t
    {
        EM_CDC,         //! Candlepower Distribution Curve
        EM_IES_C,       //! IES Candela
        EM_SPERICAL_C,  //! Sperical coordinates
        EM_DIRECTION,   //! Sample direction
        EM_PASS_T_MASK, //! Test mask

        EM_SIZE
    };

    struct IES 
    {
        struct 
        {
            core::smart_refctd_ptr<video::IGPUImageView> candela = nullptr, spherical = nullptr, direction = nullptr, mask = nullptr;
        } views;

        struct
        {
            core::smart_refctd_ptr<video::IGPUBuffer> vAngles = nullptr, hAngles = nullptr, data = nullptr;
        } buffers;

        asset::SAssetBundle bundle;
        std::string key;

        float zDegree;
        E_MODE mode;

        inline const asset::CIESProfile* getProfile() const
        { 
            auto* meta = bundle.getMetadata();
            if (meta)
                return &meta->selfCast<const asset::CIESProfileMetadata>()->profile;

            return nullptr;
        }
    };

    smart_refctd_ptr<IGPUGraphicsPipeline> graphicsPipeline;

    bool running = true;
    std::vector<IES> assets;
    size_t activeAssetIx = 0;

    smart_refctd_ptr<ISemaphore> m_semaphore;
    uint64_t m_realFrameIx = 0;
    std::array<smart_refctd_ptr<IGPUCommandBuffer>, device_base_t::MaxFramesInFlight> m_cmdBufs;
    InputSystem::ChannelReader<IMouseEventChannel> mouse;
    InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

    // TODO: lets have this stuff in nice imgui
    void mouseProcess(const nbl::ui::IMouseEventChannel::range_t& events)
    {
        for (auto it = events.begin(); it != events.end(); it++)
        {
            auto ev = *it;

            if (ev.type == nbl::ui::SMouseEvent::EET_SCROLL)
            {
                auto& ies = assets[activeAssetIx];
                auto* profile = ies.getProfile();

                auto impulse = ev.scrollEvent.verticalScroll;
                ies.zDegree = std::clamp<float>(ies.zDegree + impulse, profile->getHoriAngles().front(), profile->getHoriAngles().back());
            }
        }
    }

    void keyboardProcess(const nbl::ui::IKeyboardEventChannel::range_t& events)
    {
        for (auto it = events.begin(); it != events.end(); it++)
        {
            const auto ev = *it;

            if (ev.action == nbl::ui::SKeyboardEvent::ECA_RELEASED)
            {
                auto& ies = assets[activeAssetIx];
                auto* profile = ies.getProfile();

                if (ev.keyCode == nbl::ui::EKC_UP_ARROW)
                    activeAssetIx = std::clamp<size_t>(activeAssetIx + 1, 0, assets.size());
                else if(ev.keyCode == nbl::ui::EKC_DOWN_ARROW)
                    activeAssetIx = std::clamp<size_t>(activeAssetIx - 1, 0, assets.size());

                if (ev.keyCode == nbl::ui::EKC_C)
                    ies.mode = EM_CDC;
                else if (ev.keyCode == nbl::ui::EKC_V)
                    ies.mode = EM_IES_C;
                else if (ev.keyCode == nbl::ui::EKC_S)
                    ies.mode = EM_SPERICAL_C;
                else if (ev.keyCode == nbl::ui::EKC_D)
                    ies.mode = EM_DIRECTION;
                else if (ev.keyCode == nbl::ui::EKC_M)
                    ies.mode = EM_PASS_T_MASK;

                if (ev.keyCode == nbl::ui::EKC_Q)
                    running = false;
            }
        }
    }
    // <-

    core::smart_refctd_ptr<IGPUImageView> createImageView(const size_t width, const size_t height, asset::E_FORMAT format, std::string name)
    {
        IGPUImage::SCreationParams imageParams {};
        imageParams.type = IImage::E_TYPE::ET_2D;
        imageParams.extent.height = height;
        imageParams.extent.width = width;
        imageParams.extent.depth = 1u;
        imageParams.format = format;
        imageParams.mipLevels = 1u;
        imageParams.flags = IImage::ECF_NONE;
        imageParams.arrayLayers = 1u;
        imageParams.samples = IImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT;

        auto image = m_device->createImage(std::move(imageParams));
        image->setObjectDebugName(name.c_str());

        if (!image)
        {
            m_logger->log("Failed to create \"%s\" image!", system::ILogger::ELL_ERROR, name.c_str());
            return nullptr;
        }

        auto allocation = m_device->allocate(image->getMemoryReqs(), image.get(), nbl::video::IDeviceMemoryAllocation::EMAF_NONE);
        if (!allocation.isValid())
        {
            m_logger->log("Failed to allocate device memory for \"%s\" image!", system::ILogger::ELL_ERROR, name.c_str());
            return nullptr;
        }

        IGPUImageView::SCreationParams viewParams {};
        viewParams.image = std::move(image);
        viewParams.format = format;
        viewParams.viewType = IGPUImageView::ET_2D;
        viewParams.flags = IImageViewBase::ECF_NONE;
        viewParams.subresourceRange.baseArrayLayer = 0u;
        viewParams.subresourceRange.baseMipLevel = 0u;
        viewParams.subresourceRange.layerCount = 1u;
        viewParams.subresourceRange.levelCount = 1u;
        viewParams.subresourceRange.aspectMask = core::bitflag(asset::IImage::EAF_COLOR_BIT);

        auto imageView = m_device->createImageView(std::move(viewParams));

        if(not imageView)
            m_logger->log("Failed to create image view for \"%s\" image!", system::ILogger::ELL_ERROR, name.c_str());

        return imageView;
    }

    core::smart_refctd_ptr<IGPUBuffer> createBuffer(const core::vector<asset::CIESProfile::IES_STORAGE_FORMAT>& in, std::string name)
    {        
        IGPUBuffer::SCreationParams bufferParams = {};
        bufferParams.usage = core::bitflag(asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | IGPUBuffer::EUF_TRANSFER_DST_BIT /*TODO: <- double check*/;;
        bufferParams.size = sizeof(asset::CIESProfile::IES_STORAGE_FORMAT) * in.size();

        auto buffer = m_device->createBuffer(std::move(bufferParams));
        buffer->setObjectDebugName(name.c_str());

        if (not buffer)
        {
            m_logger->log("Failed to create \"%s\" buffer!", ILogger::ELL_ERROR, name.c_str());
            return nullptr;
        }

        auto memoryReqs = buffer->getMemoryReqs();

        if(m_utils)
            memoryReqs.memoryTypeBits &= m_utils->getLogicalDevice()->getPhysicalDevice()->getUpStreamingMemoryTypeBits();

        auto allocation = m_device->allocate(memoryReqs, buffer.get(), core::bitflag<video::IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS>(video::IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT));
        if (not allocation.isValid())
        {
            m_logger->log("Failed to allocate \"%s\" buffer!", ILogger::ELL_ERROR, name.c_str());
            return nullptr;
        }

        auto* mappedPointer = allocation.memory->map({ 0ull, memoryReqs.size }, IDeviceMemoryAllocation::EMCAF_READ_AND_WRITE);

        if (not mappedPointer)
        {
            m_logger->log("Failed to map device memory for \"%s\" buffer!", ILogger::ELL_ERROR, name.c_str());
            return nullptr;
        }

        memcpy(mappedPointer, in.data(), buffer->getSize());

        if (not allocation.memory->unmap())
        {
            m_logger->log("Failed to unmap device memory for \"%s\" buffer!", ILogger::ELL_ERROR, name.c_str());
            return nullptr;
        }

        return buffer;
    }

    inline void updatePushConstants(PushConstants& out, const IES& in)
    {
        out.vAnglesBDA = in.buffers.vAngles->getDeviceAddress();
        out.hAnglesBDA = in.buffers.hAngles->getDeviceAddress();
        out.dataBDA = in.buffers.data->getDeviceAddress();

        const auto* profile = in.getProfile();

        out.maxIValue = profile->getMaxCandelaValue();
        out.vAnglesCount = profile->getVertAngles().size();
        out.hAnglesCount = profile->getHoriAngles().size();
        out.dataCount = profile->getData().size();

        out.zAngleDegreeRotation = in.zDegree;
        out.mode = in.mode;
        out.texIx = activeAssetIx;
    }
};

NBL_MAIN_FUNC(IESViewer)