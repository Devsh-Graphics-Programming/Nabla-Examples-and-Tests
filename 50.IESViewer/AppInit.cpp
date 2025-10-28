// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "App.hpp"
#include "AppInputParser.hpp"
#include "app_resources/common.hlsl"
#include "app_resources/imgui.opts.hlsl"
#include "nbl/this_example/builtin/build/spirv/keys.hpp"

#define MEDIA_ENTRY "../../media"
#define INPUT_JSON_FILE "../inputs.json"

bool IESViewer::onAppInitialized(smart_refctd_ptr<ISystem>&& system)
{
    if (!asset_base_t::onAppInitialized(smart_refctd_ptr(system)))
        return false;
    if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
        return false;

    const auto media = absolute(path(MEDIA_ENTRY));

    AppInputParser::Output out;
    AppInputParser parser(system::logger_opt_ptr(m_logger.get()));
    if (!parser.parse(out, INPUT_JSON_FILE, media.string()))
        return false;

    m_logger->log("Loading IES m_assets..", system::ILogger::ELL_INFO);
    {
        auto start = std::chrono::high_resolution_clock::now();
        size_t loaded = {}, total = out.inputList.size();
        IAssetLoader::SAssetLoadParams lp = {};
        lp.logger = system::logger_opt_ptr(m_logger.get());

        for (const auto& in : out.inputList)
        {
            auto asset = m_assetMgr->getAsset(in.c_str(), lp);

            if (asset.getMetadata())
            {
                auto& ies = m_assets.emplace_back();
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
            m_logger->log("Failed to load [%s/%s] IES m_assets!", system::ILogger::ELL_ERROR, diff.c_str(), st.c_str());
        }
        auto elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start);
        auto took = std::to_string(elapsed.count());
        m_logger->log("Finished loading IES m_assets, took %s seconds.", system::ILogger::ELL_PERFORMANCE, took.c_str());
    }

    m_logger->log("Creating GPU IES resources..", system::ILogger::ELL_INFO);
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (auto& ies : m_assets)
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
        const auto m_assets = assetBundle.getContents();

        if (m_assets.empty())
        {
            m_logger->log("Failed to load \"%s\" shader!", system::ILogger::ELL_ERROR, key.data());
            return nullptr;
        }

        auto spirvShader = IAsset::castDown<IShader>(m_assets[0]);

        if (spirvShader)
            m_logger->log("Loaded \"%s\".", system::ILogger::ELL_INFO, key.data());
        else
            m_logger->log("Failed to cast \"%s\" asset to IShader!", system::ILogger::ELL_ERROR, key.data());

        return spirvShader;
    };

    #define CREATE_SHADER(SHADER, PATH) \
	if (!(SHADER = createShader.template operator()<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE(PATH)>() )) return false;

    m_logger->log("Loading GPU shaders..", system::ILogger::ELL_INFO);
    smart_refctd_ptr<IShader> compute, pixel, vertex, imguiVertex, imguiPixel;
    {
        auto start = std::chrono::high_resolution_clock::now();
        CREATE_SHADER(compute, "compute")
        CREATE_SHADER(pixel, "pixel")
        CREATE_SHADER(vertex, "vertex")
        CREATE_SHADER(imguiVertex, "imgui.vertex")
        CREATE_SHADER(imguiPixel, "imgui.pixel")
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
        static constexpr auto StageFlags = core::bitflag(stage_flags_t::ESS_FRAGMENT) | stage_flags_t::ESS_COMPUTE;

        //! single descriptor for both compute & graphics, we will only need to trasition images' layout with a barrier
        #define BINDING_TEXTURE(IX, TYPE) { .binding = IX, .type = TYPE, .createFlags = TexturesCreateFlags, .stageFlags = StageFlags, .count = MAX_IES_IMAGES, .immutableSamplers = nullptr }
        #define BINDING_SAMPLER(IX) { .binding = IX, .type = IDescriptor::E_TYPE::ET_SAMPLER, .createFlags = SamplersCreateFlags, .stageFlags = StageFlags, .count = 1u, .immutableSamplers = nullptr }
        static constexpr auto bindings = std::to_array<IGPUDescriptorSetLayout::SBinding>
        ({
            BINDING_TEXTURE(0u, IDescriptor::E_TYPE::ET_SAMPLED_IMAGE), BINDING_TEXTURE(0u + 10u, IDescriptor::E_TYPE::ET_STORAGE_IMAGE), // candela
            BINDING_TEXTURE(1u, IDescriptor::E_TYPE::ET_SAMPLED_IMAGE), BINDING_TEXTURE(1u + 10u, IDescriptor::E_TYPE::ET_STORAGE_IMAGE), // spherical
            BINDING_TEXTURE(2u, IDescriptor::E_TYPE::ET_SAMPLED_IMAGE), BINDING_TEXTURE(2u + 10u, IDescriptor::E_TYPE::ET_STORAGE_IMAGE), // direction
            BINDING_TEXTURE(3u, IDescriptor::E_TYPE::ET_SAMPLED_IMAGE), BINDING_TEXTURE(3u + 10u, IDescriptor::E_TYPE::ET_STORAGE_IMAGE), // mask
            BINDING_SAMPLER(0u + 100u)
        });

        const uint32_t texturesCount = m_assets.size();
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

            generalSampler->setObjectDebugName("General IES sampler");
        }

        auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
        scRes->getRenderpass(); // note it also creates rp if nulled
        {
            auto descriptorSetLayout = m_device->createDescriptorSetLayout(bindings);

            if (not descriptorSetLayout)
                return logFail("Failed to create descriptor set layout!");

            auto range = std::to_array<asset::SPushConstantRange>({ {StageFlags.value, 0u, sizeof(PushConstants)} });
            auto pipelineLayout = m_device->createPipelineLayout(range, core::smart_refctd_ptr(descriptorSetLayout), nullptr, nullptr, nullptr);

            if (not pipelineLayout)
                return logFail("Failed to create pipeline layout!");

            // Compute Pipeline
            {
                auto params = std::to_array<IGPUComputePipeline::SCreationParams>({ {} });;
                params[0].layout = pipelineLayout.get();
                params[0].shader.shader = compute.get();
                params[0].shader.entryPoint = "main";

                if (!m_device->createComputePipelines(nullptr, params, &m_computePipeline))
                    return logFail("Failed to create compute pipeline!");
            }

            // Graphics Pipeline
            {
                IGPUPipelineBase::SShaderEntryMap specConstants;
                const auto orientationAsUint32 = static_cast<uint32_t>(hlsl::SurfaceTransform::FLAG_BITS::IDENTITY_BIT);
                specConstants[0] = std::span{ reinterpret_cast<const uint8_t*>(&orientationAsUint32), sizeof(orientationAsUint32) };

                video::IGPUPipelineBase::SShaderSpecInfo specInfo[] =
                {
                    {.shader = vertex.get(), .entryPoint = "main", .entries = &specConstants },
                    {.shader = pixel.get(), .entryPoint = "PSMain" }
                };

                auto params = std::to_array<IGPUGraphicsPipeline::SCreationParams>({ {} });
                params[0].renderpass = scRes->getRenderpass();
                params[0].vertexShader = specInfo[0];
                params[0].fragmentShader = specInfo[1];
                params[0].layout = pipelineLayout.get();
                params[0].cached =
                {
                    .vertexInput = {}, // full screen tri ext, no inputs
                    .primitiveAssembly = {},
                    .rasterization = {
                        .polygonMode = EPM_FILL,
                        .faceCullingMode = EFCM_NONE,
                        .depthWriteEnable = false,
                    },
                    .blend = {},
                    .subpassIx = 0u
                };

                if (!m_device->createGraphicsPipelines(nullptr, params, &m_graphicsPipeline))
                    return logFail("Failed to create graphics pipeline!");
            }

            const auto dscLayoutPtrs = m_graphicsPipeline->getLayout()->getDescriptorSetLayouts();
            auto pool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_UPDATE_AFTER_BIND_BIT, dscLayoutPtrs);
            pool->createDescriptorSets(dscLayoutPtrs.size(), dscLayoutPtrs.data(), m_descriptors.data());
            {
                std::array<std::vector<IGPUDescriptorSet::SDescriptorInfo>, 4u + 1u> infos;
#define FILL_INFO(DESC, IX) \
                { \
                    auto& info = infos[IX].emplace_back(); \
                    info.desc = DESC; \
                    info.info.image.imageLayout = IImage::LAYOUT::GENERAL; \
                }

                for (uint32_t i = 0; i < m_assets.size(); ++i)
                {
                    auto& ies = m_assets[i];

                    FILL_INFO(ies.views.candela, 0u)
                        FILL_INFO(ies.views.spherical, 1u)
                        FILL_INFO(ies.views.direction, 2u)
                        FILL_INFO(ies.views.mask, 3u)
                }
                FILL_INFO(generalSampler, 4u);
                auto* samplerInfo = infos.back().data();
                samplerInfo->info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;

                std::array<IGPUDescriptorSet::SWriteDescriptorSet, 4u * 2u + 1u > writes;
                for (uint32_t i = 0; i < 4u; ++i)
                {
                    auto& write = writes[i];
                    write.count = m_assets.size();
                    write.info = infos[i].data();
                    write.dstSet = m_descriptors[0u].get();
                    write.arrayElement = 0u;
                    write.binding = i;
                }

                for (uint32_t i = 4u; i < 8u; ++i)
                {
                    auto ix = i - 4u;
                    auto& write = writes[i] = writes[ix];
                    write.binding = ix + 10u;
                }

                auto& write = writes.back();
                write.count = 1u;
                write.info = samplerInfo;
                write.dstSet = m_descriptors[0u].get();
                write.arrayElement = 0u;
                write.binding = 0u + 100u;

                if (!m_device->updateDescriptorSets(writes, {}))
                    return logFail("Failed to write descriptor sets");
            }
        }
    }

    // frame buffers
    {
        // TODO: I will create my own
        auto renderpass = smart_refctd_ptr<IGPURenderpass>(static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources())->getRenderpass());

        for (uint32_t i = 0u; i < m_frameBuffers2D.size(); ++i)
        {
            auto& fb2D = m_frameBuffers2D[i];
            auto& fb3D = m_frameBuffers3D[i];
            auto ixs = std::to_string(i);

            // TODO: may actually change it, temporary hardcoding
            constexpr auto WIDTH = 640, HEIGHT = 640;

            {
                auto color = createImageView(WIDTH, HEIGHT, EF_R8G8B8A8_SRGB, "[2D Plot]: framebuffer[" + ixs + "].color attachement", IGPUImage::EUF_RENDER_ATTACHMENT_BIT | IGPUImage::EUF_SAMPLED_BIT, IImage::EAF_COLOR_BIT);
                fb2D = m_device->createFramebuffer
                (
                    { {
                        .renderpass = renderpass,
                        .depthStencilAttachments = nullptr,
                        .colorAttachments = &color.get(),
                        .width = WIDTH,
                        .height = HEIGHT
                    } }
                );
            }

            {
                auto color = createImageView(WIDTH, HEIGHT, EF_R8G8B8A8_SRGB, "[3D Plot]: framebuffer[" + ixs + "].color attachement", IGPUImage::EUF_RENDER_ATTACHMENT_BIT | IGPUImage::EUF_SAMPLED_BIT, IImage::EAF_COLOR_BIT);
                auto depth = createImageView(WIDTH, HEIGHT, EF_D32_SFLOAT, "[3D Plot]: framebuffer[" + ixs + "].depth attachement", IGPUImage::EUF_RENDER_ATTACHMENT_BIT | IGPUImage::EUF_SAMPLED_BIT, IGPUImage::EAF_DEPTH_BIT);

                fb3D = m_device->createFramebuffer
                (
                    { {
                        .renderpass = renderpass,
                        .depthStencilAttachments = nullptr,
                        .colorAttachments = &color.get(),
                        .width = WIDTH,
                        .height = HEIGHT
                    } }
                );
            }
        }
    }

    // imGUI
    {
        auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
        ext::imgui::UI::SCreationParameters params = {};
        params.resources.texturesInfo = { .setIx = NBL_TEXTURES_SET_IX, .bindingIx = NBL_TEXTURES_BINDING_IX };
        params.resources.samplersInfo = { .setIx = NBL_SAMPLER_STATES_SET_IX, .bindingIx = NBL_SAMPLER_STATES_BINDING_IX };
        params.utilities = m_utils;
        params.transfer = getTransferUpQueue();
        params.pipelineLayout = ext::imgui::UI::createDefaultPipelineLayout(m_utils->getLogicalDevice(), params.resources.texturesInfo, params.resources.samplersInfo, NBL_TEXTURES_COUNT);
        params.assetManager = make_smart_refctd_ptr<IAssetManager>(smart_refctd_ptr(m_system));
        params.renderpass = smart_refctd_ptr<IGPURenderpass>(scRes->getRenderpass());
        params.subpassIx = 0u;
        params.pipelineCache = nullptr;

        using imgui_precompiled_spirv_t = ext::imgui::UI::SCreationParameters::PrecompiledShaders;
        params.spirv = std::make_optional(imgui_precompiled_spirv_t{ .vertex = imguiVertex, .fragment = imguiPixel });

        auto* imgui = (ui.it = ext::imgui::UI::create(std::move(params))).get();
        if (not imgui)
            return logFail("Failed to create `nbl::ext::imgui::UI` class");

        {
            const auto* layout = imgui->getPipeline()->getLayout()->getDescriptorSetLayout(0u);
            auto pool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::E_CREATE_FLAGS::ECF_UPDATE_AFTER_BIND_BIT, { &layout,1 });
            auto ds = pool->createDescriptorSet(smart_refctd_ptr<const IGPUDescriptorSetLayout>(layout));
            ui.descriptor = make_smart_refctd_ptr<SubAllocatedDescriptorSet>(std::move(ds));
            if (!ui.descriptor)
                return logFail("Failed to create the descriptor set");

            {
                std::array<SubAllocatedDescriptorSet::value_type, 1u + 2u * MaxFramesInFlight> addresses;
                addresses.fill(SubAllocatedDescriptorSet::invalid_value);
                ui.descriptor->multi_allocate(0, addresses.size(), addresses.data());

                bool ok = true;
                ok &= addresses.front() == ext::imgui::UI::FontAtlasTexId;
                for (auto i = ext::imgui::UI::FontAtlasTexId; i < addresses.size(); ++i)
                    ok &= addresses[i] == i;

                assert(ok);

                std::array<IGPUDescriptorSet::SDescriptorInfo, addresses.size()> infos;
                for (auto& it : infos) it.info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;

                auto* ix = addresses.data();
                infos[*ix].desc = smart_refctd_ptr<nbl::video::IGPUImageView>(imgui->getFontAtlasView()); ++ix;
                for (uint8_t i = 0u; i < MaxFramesInFlight; ++i, ++ix) infos[*ix].desc = m_frameBuffers2D[i]->getCreationParameters().colorAttachments[0u];
                for (uint8_t i = 0u; i < MaxFramesInFlight; ++i, ++ix) infos[*ix].desc = m_frameBuffers3D[i]->getCreationParameters().colorAttachments[0u];
                
                auto writes = std::to_array({ IGPUDescriptorSet::SWriteDescriptorSet{
                    .dstSet = ui.descriptor->getDescriptorSet(),
                    .binding = NBL_TEXTURES_BINDING_IX,
                    .arrayElement = 0u,
                    .count = infos.size(),
                    .info = infos.data()
                }});

                if (!m_device->updateDescriptorSets(writes, {}))
                    return logFail("Failed to write the descriptor set");
            }
        }

        imgui->registerListener([this]()
        {
            uiListener();
        });
    }

    m_semaphore = m_device->createSemaphore(m_realFrameIx);
    if (!m_semaphore)
        return logFail("Failed to Create a Semaphore!");

    using pool_flags_t = IGPUCommandPool::CREATE_FLAGS;

    auto createCommandBuffers = [&](auto* queue, const std::span<core::smart_refctd_ptr<IGPUCommandBuffer>> out, pool_flags_t flags) -> bool
    {
        auto pool = m_device->createCommandPool(queue->getFamilyIndex(), flags);
        if (!pool)
            return logFail("Couldn't create command pool!");
        if (!pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, out))
            return logFail("Couldn't create command buffer!");
        return true;
    };

    // render loop command buffers
    if (not createCommandBuffers(getGraphicsQueue(), m_cmdBuffers, pool_flags_t::RESET_COMMAND_BUFFER_BIT))
        return false;

    // transient command buffer
    {
        auto* queue = getGraphicsQueue();
        auto cbs = std::to_array({ smart_refctd_ptr<nbl::video::IGPUCommandBuffer>() });
        if (not createCommandBuffers(queue, cbs, pool_flags_t::RESET_COMMAND_BUFFER_BIT | pool_flags_t::TRANSIENT_BIT))
            return false;

        std::vector<IGPUImage*> images;
        for (uint32_t i = 0; i < m_assets.size(); ++i)
        {
            auto& ies = m_assets[i];

            images.emplace_back() = ies.views.candela->getCreationParameters().image.get();
            images.emplace_back() = ies.views.spherical->getCreationParameters().image.get();
            images.emplace_back() = ies.views.direction->getCreationParameters().image.get();
            images.emplace_back() = ies.views.mask->getCreationParameters().image.get();
        }

        auto* cb = cbs.front().get();
        cb->setObjectDebugName("Transient Command Buffer");

        if (not cb->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT))
            return logFail("Couldn't begin command buffer!");

        if (not IES::barrier<IImage::LAYOUT::READ_ONLY_OPTIMAL, true>(cb, images))
            return logFail("Failed to record pipeline barriers!");

        if (not cb->end())
            return logFail("Couldn't end command buffer!");

        core::smart_refctd_ptr<ISemaphore> semaphore = m_device->createSemaphore(0);
        semaphore->setObjectDebugName("Scratch Semaphore");
        {
            IQueue::SSubmitInfo::SSemaphoreInfo signal =
            {
                .semaphore = semaphore.get(),
                .value = 1u,
                .stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
            };

            const IQueue::SSubmitInfo::SCommandBufferInfo cmds[] = { {.cmdbuf = cb } };

            const IQueue::SSubmitInfo infos[] =
            { {
                .waitSemaphores = {},
                .commandBuffers = cmds,
                .signalSemaphores = {&signal,1}
            } };

            if (queue->submit(infos) != IQueue::RESULT::SUCCESS)
                return logFail("Failed to submit queue!");
        }

        {
            const ISemaphore::SWaitInfo infos[] =
            { {
                .semaphore = semaphore.get(),
                .value = 1u
            } };

            if (m_device->blockForSemaphores(infos) != ISemaphore::WAIT_RESULT::SUCCESS)
                return logFail("Couldn't block for scratch semaphore!");
        }
    }

    onAppInitializedFinish();

    return true;
}