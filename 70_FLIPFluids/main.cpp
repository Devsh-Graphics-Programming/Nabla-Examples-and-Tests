// Copyright (C) 2024-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/this_example/builtin/build/spirv/keys.hpp"

#include "nbl/examples/examples.hpp"
// TODO: why is it not in nabla.h ?
#include "nbl/asset/metadata/CHLSLMetadata.h"
#include <nbl/builtin/hlsl/math/thin_lens_projection.hlsl>

using namespace nbl;
using namespace nbl::core;
using namespace nbl::hlsl;
using namespace nbl::system;
using namespace nbl::asset;
using namespace nbl::ui;
using namespace nbl::video;
using namespace nbl::examples;

#include "app_resources/common.hlsl"
#include "app_resources/gridUtils.hlsl"
#include "app_resources/render_common.hlsl"
#include "app_resources/descriptor_bindings.hlsl"


enum SimPresets
{
    CENTER_DROP,
    LONG_BOX
};

struct SMVPParams
{
    float cameraPosition[4];

    float MVP[4*4];
    float M[4*4];
    float V[4*4];
    float P[4*4];
};

class CSwapchainFramebuffersAndDepth final : public nbl::video::CDefaultSwapchainFramebuffers
{
    using scbase_t = CDefaultSwapchainFramebuffers;

public:
    template<typename... Args>
    inline CSwapchainFramebuffersAndDepth(ILogicalDevice* device, const asset::E_FORMAT _desiredDepthFormat, Args&&... args)
        : CDefaultSwapchainFramebuffers(device, std::forward<Args>(args)...)
    {
        const IPhysicalDevice::SImageFormatPromotionRequest req = {
            .originalFormat = _desiredDepthFormat,
            .usages = {IGPUImage::EUF_RENDER_ATTACHMENT_BIT}
        };
        m_depthFormat = m_device->getPhysicalDevice()->promoteImageFormat(req, IGPUImage::TILING::OPTIMAL);

        const static IGPURenderpass::SCreationParams::SDepthStencilAttachmentDescription depthAttachments[] = {
            {{
                {
                    .format = m_depthFormat,
                    .samples = IGPUImage::ESCF_1_BIT,
                    .mayAlias = false
                },
            /*.loadOp = */{IGPURenderpass::LOAD_OP::CLEAR},
            /*.storeOp = */{IGPURenderpass::STORE_OP::STORE},
            /*.initialLayout = */{IGPUImage::LAYOUT::UNDEFINED}, // because we clear we don't care about contents
            /*.finalLayout = */{IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL} // transition to presentation right away so we can skip a barrier
        }},
        IGPURenderpass::SCreationParams::DepthStencilAttachmentsEnd
        };
        m_params.depthStencilAttachments = depthAttachments;

        static IGPURenderpass::SCreationParams::SSubpassDescription subpasses[] = {
            m_params.subpasses[0],
            IGPURenderpass::SCreationParams::SubpassesEnd
        };
        subpasses[0].depthStencilAttachment.render = { .attachmentIndex = 0,.layout = IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL };
        m_params.subpasses = subpasses;

        // TODO: Two subpass external dependencies SRC and DST needed!
    }

protected:
    inline bool onCreateSwapchain_impl(const uint8_t qFam) override
    {
        auto device = const_cast<ILogicalDevice*>(m_renderpass->getOriginDevice());

        const auto depthFormat = m_renderpass->getCreationParameters().depthStencilAttachments[0].format;
        const auto& sharedParams = getSwapchain()->getCreationParameters().sharedParams;
        auto image = device->createImage({ IImage::SCreationParams{
            .type = IGPUImage::ET_2D,
            .samples = IGPUImage::ESCF_1_BIT,
            .format = depthFormat,
            .extent = {sharedParams.width,sharedParams.height,1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .depthUsage = IGPUImage::EUF_RENDER_ATTACHMENT_BIT
        } });

        device->allocate(image->getMemoryReqs(), image.get());

        m_depthBuffer = device->createImageView({
            .flags = IGPUImageView::ECF_NONE,
            .subUsages = IGPUImage::EUF_RENDER_ATTACHMENT_BIT,
            .image = std::move(image),
            .viewType = IGPUImageView::ET_2D,
            .format = depthFormat,
            .subresourceRange = {IGPUImage::EAF_DEPTH_BIT,0,1,0,1}
            });

        const auto retval = scbase_t::onCreateSwapchain_impl(qFam);
        m_depthBuffer = nullptr;
        return retval;
    }

    inline smart_refctd_ptr<IGPUFramebuffer> createFramebuffer(IGPUFramebuffer::SCreationParams&& params) override
    {
        params.depthStencilAttachments = &m_depthBuffer.get();
        return m_device->createFramebuffer(std::move(params));
    }

    E_FORMAT m_depthFormat;
    smart_refctd_ptr<IGPUImageView> m_depthBuffer;
};

class CEventCallback : public ISimpleManagedSurface::ICallback
{
public:
    CEventCallback(nbl::core::smart_refctd_ptr<InputSystem>&& m_inputSystem, nbl::system::logger_opt_smart_ptr&& logger) : m_inputSystem(std::move(m_inputSystem)), m_logger(std::move(logger)) {}
    CEventCallback() {}

    void setLogger(nbl::system::logger_opt_smart_ptr& logger)
    {
        m_logger = logger;
    }
    void setInputSystem(nbl::core::smart_refctd_ptr<InputSystem>&& m_inputSystem)
    {
        m_inputSystem = std::move(m_inputSystem);
    }
private:

    void onMouseConnected_impl(nbl::core::smart_refctd_ptr<nbl::ui::IMouseEventChannel>&& mch) override
    {
        m_logger.log("A mouse %p has been connected", nbl::system::ILogger::ELL_INFO, mch.get());
        m_inputSystem.get()->add(m_inputSystem.get()->m_mouse, std::move(mch));
    }
    void onMouseDisconnected_impl(nbl::ui::IMouseEventChannel* mch) override
    {
        m_logger.log("A mouse %p has been disconnected", nbl::system::ILogger::ELL_INFO, mch);
        m_inputSystem.get()->remove(m_inputSystem.get()->m_mouse, mch);
    }
    void onKeyboardConnected_impl(nbl::core::smart_refctd_ptr<nbl::ui::IKeyboardEventChannel>&& kbch) override
    {
        m_logger.log("A keyboard %p has been connected", nbl::system::ILogger::ELL_INFO, kbch.get());
        m_inputSystem.get()->add(m_inputSystem.get()->m_keyboard, std::move(kbch));
    }
    void onKeyboardDisconnected_impl(nbl::ui::IKeyboardEventChannel* kbch) override
    {
        m_logger.log("A keyboard %p has been disconnected", nbl::system::ILogger::ELL_INFO, kbch);
        m_inputSystem.get()->remove(m_inputSystem.get()->m_keyboard, kbch);
    }

private:
    nbl::core::smart_refctd_ptr<InputSystem> m_inputSystem = nullptr;
    nbl::system::logger_opt_smart_ptr m_logger = nullptr;
};

class FLIPFluidsApp final : public SimpleWindowedApplication, public BuiltinResourcesApplication
{
    using device_base_t = SimpleWindowedApplication;
    using asset_base_t = BuiltinResourcesApplication;
    using clock_t = std::chrono::steady_clock;

    constexpr static inline uint32_t WIN_WIDTH = 1280, WIN_HEIGHT = 720;
    constexpr static inline uint32_t MaxFramesInFlight = 3u;

    constexpr static inline clock_t::duration DisplayImageDuration = std::chrono::milliseconds(900);

public:
    inline FLIPFluidsApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
        : IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

    inline SPhysicalDeviceFeatures getPreferredDeviceFeatures() const override
    {
        auto retval = device_base_t::getPreferredDeviceFeatures();
        retval.pipelineExecutableInfo = true;
        return retval;
    }

    inline core::vector<video::SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const override
    {
        if (!m_surface)
        {
            {
                auto windowCallback = core::make_smart_refctd_ptr<CEventCallback>(smart_refctd_ptr(m_inputSystem), smart_refctd_ptr(m_logger));
                IWindow::SCreationParams params{
                    .callback = core::make_smart_refctd_ptr<ISimpleManagedSurface::ICallback>(),
                    .x = 32,
                    .y = 32,
                    .width = WIN_WIDTH,
                    .height = WIN_HEIGHT,
                    .flags = IWindow::ECF_HIDDEN | IWindow::ECF_BORDERLESS | IWindow::ECF_RESIZABLE,
                    .windowCaption = "FLIPFluidsApp"
                };
                params.callback = windowCallback;
                const_cast<std::remove_const_t<decltype(m_window)>&>(m_window) = m_winMgr->createWindow(std::move(params));
            }

            auto surface = CSurfaceVulkanWin32::create(smart_refctd_ptr(m_api), smart_refctd_ptr_static_cast<IWindowWin32>(m_window));
            const_cast<std::remove_const_t<decltype(m_surface)>&>(m_surface) = nbl::video::CSimpleResizeSurface<CSwapchainFramebuffersAndDepth>::create(std::move(surface));
        }

        if (m_surface)
            return { { m_surface->getSurface() } };

        return {};
    }

    inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
    {
        m_inputSystem = make_smart_refctd_ptr<InputSystem>(logger_opt_smart_ptr(smart_refctd_ptr(m_logger)));

        if (!device_base_t::onAppInitialized(std::move(system)))
            return false;
        if (!asset_base_t::onAppInitialized(std::move(system)))
            return false;

        // init grid params
        usePreset(CENTER_DROP);
        
        WorkgroupCountParticles = (numParticles + WorkgroupSize - 1) / WorkgroupSize;
        WorkgroupCountGrid = {
            (m_gridData.gridSize.x + WorkgroupGridDim - 1) / WorkgroupGridDim,
            (m_gridData.gridSize.y + WorkgroupGridDim - 1) / WorkgroupGridDim,
            (m_gridData.gridSize.z + WorkgroupGridDim - 1) / WorkgroupGridDim
        };

        {
            float zNear = 0.1f, zFar = 10000.f;
            core::vectorSIMDf cameraPosition(14, 8, 12);
            core::vectorSIMDf cameraTarget(0, 0, 0);
            hlsl::float32_t4x4 projectionMatrix = hlsl::math::thin_lens::lhPerspectiveFovMatrix(core::radians(60.0f), float(WIN_WIDTH) / WIN_HEIGHT, zNear, zFar);
            camera = Camera(cameraPosition, cameraTarget, projectionMatrix, 1.069f, 0.4f);

            m_pRenderParams.zNear = zNear;
            m_pRenderParams.zFar = zFar;
        }
        m_pRenderParams.radius = m_gridData.gridCellSize * 0.4f;

        // create buffers
        video::IGPUBuffer::SCreationParams params = {};
        params.size = sizeof(SGridData);
        params.usage = IGPUBuffer::EUF_UNIFORM_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_INLINE_UPDATE_VIA_CMDBUF;
        createBuffer(gridDataBuffer, params);

        params.size = 2 * sizeof(float32_t4);
        createBuffer(pressureParamsBuffer, params);
        
        params.size = sizeof(SMVPParams);
        createBuffer(cameraBuffer, params);

        params.size = sizeof(SParticleRenderParams);
        createBuffer(pParamsBuffer, params);

        params.size = numParticles * sizeof(float32_t3);
        params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;;
        createBuffer(particleData.positionBuffer, params, IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
        createBuffer(particleData.velocityBuffer, params, IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);

        params.size = numParticles * 6 * sizeof(VertexInfo);
        createBuffer(particleVertexBuffer, params, IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);

        asset::VkExtent3D gridExtent = { m_gridData.gridSize.x, m_gridData.gridSize.y, m_gridData.gridSize.z };

        // cell materials
        createGridTexture(gridCellMaterialImageView, asset::EF_R16_UINT, gridExtent, asset::IImage::EUF_STORAGE_BIT, "cell material0");
        createGridTexture(tempCellMaterialImageView, asset::EF_R16_UINT, gridExtent, asset::IImage::EUF_STORAGE_BIT, "cell material1");

        // TODO: What is this texture used for, and why is it 128 bits!?>
        createGridTexture(gridAxisCellMaterialImageView, asset::EF_R32G32B32A32_UINT, gridExtent, asset::IImage::EUF_STORAGE_BIT, "axis cell material0");
        createGridTexture(tempAxisCellMaterialImageView, asset::EF_R32G32B32A32_UINT, gridExtent, asset::IImage::EUF_STORAGE_BIT, "axis cell material1");

        createGridTexture(gridParticleCountImageView, asset::EF_R32_UINT, gridExtent,
            asset::IImage::EUF_STORAGE_BIT | asset::IImage::EUF_TRANSFER_DST_BIT, "particle counts");

        // diffusion grids
        createGridTexture(gridDiffusionImageView, asset::EF_R32G32B32A32_SFLOAT, gridExtent, asset::IImage::EUF_STORAGE_BIT, "diffusion");

        // pressure grids
        createGridTexture(pressureImageView, asset::EF_R32_SFLOAT, gridExtent, asset::IImage::EUF_STORAGE_BIT, "pressure");
        createGridTexture(divergenceImageView, asset::EF_R32_SFLOAT, gridExtent, asset::IImage::EUF_STORAGE_BIT, "divergence");

        // velocity field stuffs
        core::bitflag<asset::IImage::E_CREATE_FLAGS> imgCreateFlags = asset::IImage::ECF_EXTENDED_USAGE_BIT;
        imgCreateFlags |= asset::IImage::ECF_MUTABLE_FORMAT_BIT;

        for (uint32_t i = 0; i < 3; i++)
        {
            {
                createGridTexture(velocityFieldImageViews[i], asset::EF_R32_SFLOAT, gridExtent,
                    asset::IImage::EUF_STORAGE_BIT | asset::IImage::EUF_TRANSFER_DST_BIT | asset::IImage::EUF_SAMPLED_BIT,
                    "velocity field" + std::to_string(i), imgCreateFlags);

                // view as uint for cas atomic op
                IGPUImageView::SCreationParams imgViewInfo;
                imgViewInfo.image = velocityFieldImageViews[i]->getCreationParameters().image;
                imgViewInfo.format = asset::EF_R32_UINT;
                imgViewInfo.viewType = IGPUImageView::ET_3D;
                imgViewInfo.flags = IGPUImageView::E_CREATE_FLAGS::ECF_NONE;
                imgViewInfo.subresourceRange.aspectMask = asset::IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
                imgViewInfo.subresourceRange.baseArrayLayer = 0u;
                imgViewInfo.subresourceRange.baseMipLevel = 0u;
                imgViewInfo.subresourceRange.layerCount = 1u;
                imgViewInfo.subresourceRange.levelCount = 1u;

                velocityFieldUintViews[i] = m_device->createImageView(std::move(imgViewInfo));
            }

            {
                createGridTexture(prevVelocityFieldImageViews[i], asset::EF_R32_SFLOAT, gridExtent,
                    asset::IImage::EUF_STORAGE_BIT | asset::IImage::EUF_TRANSFER_DST_BIT | asset::IImage::EUF_SAMPLED_BIT,
                    "prev velocity field" + std::to_string(i), imgCreateFlags);

                IGPUImageView::SCreationParams imgViewInfo;
                imgViewInfo.image = prevVelocityFieldImageViews[i]->getCreationParameters().image;
                imgViewInfo.format = asset::EF_R32_UINT;
                imgViewInfo.viewType = IGPUImageView::ET_3D;
                imgViewInfo.flags = IGPUImageView::E_CREATE_FLAGS::ECF_NONE;
                imgViewInfo.subresourceRange.aspectMask = asset::IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
                imgViewInfo.subresourceRange.baseArrayLayer = 0u;
                imgViewInfo.subresourceRange.baseMipLevel = 0u;
                imgViewInfo.subresourceRange.layerCount = 1u;
                imgViewInfo.subresourceRange.levelCount = 1u;

                prevVelocityFieldUintViews[i] = m_device->createImageView(std::move(imgViewInfo));
            }
        }

        IGPUSampler::SParams samplerParams = {};
        samplerParams.TextureWrapU = IGPUSampler::E_TEXTURE_CLAMP::ETC_CLAMP_TO_BORDER;
        samplerParams.TextureWrapV = IGPUSampler::E_TEXTURE_CLAMP::ETC_CLAMP_TO_BORDER;
        samplerParams.TextureWrapW = IGPUSampler::E_TEXTURE_CLAMP::ETC_CLAMP_TO_BORDER;
        samplerParams.BorderColor  = IGPUSampler::E_TEXTURE_BORDER_COLOR::ETBC_FLOAT_TRANSPARENT_BLACK;
        samplerParams.MinFilter		= IGPUSampler::E_TEXTURE_FILTER::ETF_LINEAR;
        samplerParams.MaxFilter		= IGPUSampler::E_TEXTURE_FILTER::ETF_LINEAR;
        samplerParams.MipmapMode	= IGPUSampler::E_SAMPLER_MIPMAP_MODE::ESMM_LINEAR;
        samplerParams.AnisotropicFilter = 3;
        samplerParams.CompareEnable = false;
        velocityFieldSampler = m_device->createSampler(samplerParams);

        // init render pipeline
        if (!initGraphicsPipeline())
            return logFail("Failed to initialize render pipeline!\n");

        
        auto createComputePipeline = [&]<core::StringLiteral ShaderKey>(smart_refctd_ptr<IGPUComputePipeline>& pipeline, smart_refctd_ptr<IDescriptorPool>& pool,
            smart_refctd_ptr<IGPUDescriptorSet>& set, const std::string& entryPoint,
            const std::span<const IGPUDescriptorSetLayout::SBinding> bindings, const asset::SPushConstantRange& pcRange = {}) -> void
            {
                auto shader = loadPrecompiledShader<ShaderKey>();

                auto descriptorSetLayout1 = m_device->createDescriptorSetLayout(bindings);

                const std::array<IGPUDescriptorSetLayout*, ICPUPipelineLayout::DESCRIPTOR_SET_COUNT> dscLayoutPtrs = {
                    nullptr,
                    descriptorSetLayout1.get(),
                    nullptr,
                    nullptr
                };
                pool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_UPDATE_AFTER_BIND_BIT, std::span(dscLayoutPtrs.begin(), dscLayoutPtrs.end()));
                set = pool->createDescriptorSet(descriptorSetLayout1);

                smart_refctd_ptr<nbl::video::IGPUPipelineLayout> pipelineLayout;
                if (pcRange.size == 0)
                    pipelineLayout = m_device->createPipelineLayout({}, nullptr, smart_refctd_ptr(descriptorSetLayout1), nullptr, nullptr);
                else
                    pipelineLayout = m_device->createPipelineLayout({ &pcRange, 1 }, nullptr, smart_refctd_ptr(descriptorSetLayout1), nullptr, nullptr);

                IGPUComputePipeline::SCreationParams params = {};
                params.layout = pipelineLayout.get();
                params.shader.entryPoint = entryPoint;
                params.shader.shader = shader.get();
                if (m_device->getEnabledFeatures().pipelineExecutableInfo)
                {
                    params.flags |= IGPUComputePipeline::SCreationParams::FLAGS::CAPTURE_STATISTICS;
                    params.flags |= IGPUComputePipeline::SCreationParams::FLAGS::CAPTURE_INTERNAL_REPRESENTATIONS;
                }
                m_device->createComputePipelines(nullptr, { &params,1 }, &pipeline);

                if (m_device->getEnabledFeatures().pipelineExecutableInfo && pipeline)
                {
                    auto report = m_device->getPipelineExecutableReport(pipeline.get(), true);
                    m_logger->log("%s Pipeline Executable Report:\n%s", ILogger::ELL_PERFORMANCE, ShaderKey.value, report.c_str());
                }
            };

        {
            // init particles pipeline
            const asset::SPushConstantRange pcRange = { .stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE, .offset = 0, .size = 2 * sizeof(uint64_t) };
            createComputePipeline.operator()<"particlesInit">(m_initParticlePipeline, m_initParticlePool, m_initParticleDs,
                 "main", piParticlesInit_bs1, pcRange);

            {
                IGPUDescriptorSet::SDescriptorInfo infos[1];
                infos[0].desc = smart_refctd_ptr(gridDataBuffer);
                infos[0].info.buffer = {.offset = 0, .size = gridDataBuffer->getSize()};
                IGPUDescriptorSet::SWriteDescriptorSet writes[1] = {
                    {.dstSet = m_initParticleDs.get(), .binding = b_piGridData, .arrayElement = 0, .count = 1, .info = &infos[0]}
                };
                m_device->updateDescriptorSets(std::span(writes, 1), {});
            }
        }
        // TODO: get rid of this pipeline!
        {
            // generate particle vertex pipeline
            const asset::SPushConstantRange pcRange = { .stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE, .offset = 0, .size = 3 * sizeof(uint64_t) };
            createComputePipeline.operator()<"genParticleVertices">(m_genParticleVerticesPipeline, m_genVerticesPool, m_genVerticesDs,
                "main", gpvGenVertices_bs1, pcRange);

            {
                IGPUDescriptorSet::SDescriptorInfo infos[2];
                infos[0].desc = smart_refctd_ptr(cameraBuffer);
                infos[0].info.buffer = {.offset = 0, .size = cameraBuffer->getSize()};
                infos[1].desc = smart_refctd_ptr(pParamsBuffer);
                infos[1].info.buffer = {.offset = 0, .size = pParamsBuffer->getSize()};
                IGPUDescriptorSet::SWriteDescriptorSet writes[2] = {
                    {.dstSet = m_genVerticesDs.get(), .binding = b_gpvCamData, .arrayElement = 0, .count = 1, .info = &infos[0]},
                    {.dstSet = m_genVerticesDs.get(), .binding = b_gpvPParams, .arrayElement = 0, .count = 1, .info = &infos[1]},
                };
                m_device->updateDescriptorSets(std::span(writes, 2), {});
            }
        }
        // update fluid cells pipelines
        {
            const asset::SPushConstantRange pcRange = { .stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE, .offset = 0, .size = 2 * sizeof(uint64_t) };
            createComputePipeline.operator()<"prepareCellUpdate">(m_accumulateWeightsPipeline, m_accumulateWeightsPool, m_accumulateWeightsDs,
                "main", ufcAccWeights_bs1, pcRange);

            {
                IGPUDescriptorSet::SDescriptorInfo infos[2];
                infos[0].desc = smart_refctd_ptr(gridDataBuffer);
                infos[0].info.buffer = { .offset = 0, .size = gridDataBuffer->getSize() };
                infos[1].desc = gridParticleCountImageView;
                infos[1].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                infos[1].info.combinedImageSampler.sampler = nullptr;

                IGPUDescriptorSet::SDescriptorInfo imgInfosVel0[3];
                imgInfosVel0[0].desc = velocityFieldUintViews[0];
                imgInfosVel0[0].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                imgInfosVel0[0].info.combinedImageSampler.sampler = nullptr;
                imgInfosVel0[1].desc = velocityFieldUintViews[1];
                imgInfosVel0[1].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                imgInfosVel0[1].info.combinedImageSampler.sampler = nullptr;
                imgInfosVel0[2].desc = velocityFieldUintViews[2];
                imgInfosVel0[2].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                imgInfosVel0[2].info.combinedImageSampler.sampler = nullptr;

                IGPUDescriptorSet::SDescriptorInfo imgInfosVel1[3];
                imgInfosVel1[0].desc = prevVelocityFieldUintViews[0];
                imgInfosVel1[0].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                imgInfosVel1[0].info.combinedImageSampler.sampler = nullptr;
                imgInfosVel1[1].desc = prevVelocityFieldUintViews[1];
                imgInfosVel1[1].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                imgInfosVel1[1].info.combinedImageSampler.sampler = nullptr;
                imgInfosVel1[2].desc = prevVelocityFieldUintViews[2];
                imgInfosVel1[2].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                imgInfosVel1[2].info.combinedImageSampler.sampler = nullptr;

                IGPUDescriptorSet::SWriteDescriptorSet writes[4] = {
                    {.dstSet = m_accumulateWeightsDs.get(), .binding = b_ufcGridData, .arrayElement = 0, .count = 1, .info = &infos[0]},
                    {.dstSet = m_accumulateWeightsDs.get(), .binding = b_ufcGridPCount, .arrayElement = 0, .count = 1, .info = &infos[1]},
                    {.dstSet = m_accumulateWeightsDs.get(), .binding = b_ufcVel, .arrayElement = 0, .count = 3, .info = imgInfosVel0},
                    {.dstSet = m_accumulateWeightsDs.get(), .binding = b_ufcPrevVel, .arrayElement = 0, .count = 3, .info = imgInfosVel1},
                };
                m_device->updateDescriptorSets(std::span(writes, 4), {});
            }
        }
        {
            createComputePipeline.operator()<"updateFluidCells">(m_updateFluidCellsPipeline, m_updateFluidCellsPool, m_updateFluidCellsDs,
                "updateFluidCells", ufcFluidCell_bs1);

            {
                IGPUDescriptorSet::SDescriptorInfo infos[3];
                infos[0].desc = smart_refctd_ptr(gridDataBuffer);
                infos[0].info.buffer = {.offset = 0, .size = gridDataBuffer->getSize()};
                infos[1].desc = gridParticleCountImageView;
                infos[1].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                infos[1].info.combinedImageSampler.sampler = nullptr;
                infos[2].desc = tempCellMaterialImageView;
                infos[2].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                infos[2].info.combinedImageSampler.sampler = nullptr;
                IGPUDescriptorSet::SWriteDescriptorSet writes[3] = {
                    {.dstSet = m_updateFluidCellsDs.get(), .binding = b_ufcGridData, .arrayElement = 0, .count = 1, .info = &infos[0]},
                    {.dstSet = m_updateFluidCellsDs.get(), .binding = b_ufcGridPCount, .arrayElement = 0, .count = 1, .info = &infos[1]},
                    {.dstSet = m_updateFluidCellsDs.get(), .binding = b_ufcCMOut, .arrayElement = 0, .count = 1, .info = &infos[2]},
                };
                m_device->updateDescriptorSets(std::span(writes, 3), {});
            }
        }
        {
            createComputePipeline.operator()<"updateFluidCells">(m_updateNeighborCellsPipeline, m_updateNeighborCellsPool, m_updateNeighborCellsDs,
                "updateNeighborFluidCells", ufcNeighborCell_bs1);

            {
                IGPUDescriptorSet::SDescriptorInfo infos[3];
                infos[0].desc = smart_refctd_ptr(gridDataBuffer);
                infos[0].info.buffer = {.offset = 0, .size = gridDataBuffer->getSize()};
                infos[1].desc = tempCellMaterialImageView;
                infos[1].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                infos[1].info.combinedImageSampler.sampler = nullptr;
                infos[2].desc = gridCellMaterialImageView;
                infos[2].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                infos[2].info.combinedImageSampler.sampler = nullptr;

                IGPUDescriptorSet::SDescriptorInfo imgInfosVel0[3];
                imgInfosVel0[0].desc = velocityFieldImageViews[0];
                imgInfosVel0[0].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                imgInfosVel0[0].info.combinedImageSampler.sampler = nullptr;
                imgInfosVel0[1].desc = velocityFieldImageViews[1];
                imgInfosVel0[1].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                imgInfosVel0[1].info.combinedImageSampler.sampler = nullptr;
                imgInfosVel0[2].desc = velocityFieldImageViews[2];
                imgInfosVel0[2].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                imgInfosVel0[2].info.combinedImageSampler.sampler = nullptr;

                IGPUDescriptorSet::SDescriptorInfo imgInfosVel1[3];
                imgInfosVel1[0].desc = prevVelocityFieldImageViews[0];
                imgInfosVel1[0].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                imgInfosVel1[0].info.combinedImageSampler.sampler = nullptr;
                imgInfosVel1[1].desc = prevVelocityFieldImageViews[1];
                imgInfosVel1[1].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                imgInfosVel1[1].info.combinedImageSampler.sampler = nullptr;
                imgInfosVel1[2].desc = prevVelocityFieldImageViews[2];
                imgInfosVel1[2].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                imgInfosVel1[2].info.combinedImageSampler.sampler = nullptr;

                IGPUDescriptorSet::SWriteDescriptorSet writes[5] = {
                    {.dstSet = m_updateNeighborCellsDs.get(), .binding = b_ufcGridData, .arrayElement = 0, .count = 1, .info = &infos[0]},
                    {.dstSet = m_updateNeighborCellsDs.get(), .binding = b_ufcCMIn, .arrayElement = 0, .count = 1, .info = &infos[1]},
                    {.dstSet = m_updateNeighborCellsDs.get(), .binding = b_ufcCMOut, .arrayElement = 0, .count = 1, .info = &infos[2]},
                    {.dstSet = m_updateNeighborCellsDs.get(), .binding = b_ufcVel, .arrayElement = 0, .count = 3, .info = imgInfosVel0},
                    {.dstSet = m_updateNeighborCellsDs.get(), .binding = b_ufcPrevVel, .arrayElement = 0, .count = 3, .info = imgInfosVel1},
                };
                m_device->updateDescriptorSets(std::span(writes, 5), {});
            }
        }
        {
            // apply forces pipeline
            createComputePipeline.operator()<"applyBodyForces">(m_applyBodyForcesPipeline, m_applyForcesPool, m_applyForcesDs, 
                "main", abfApplyForces_bs1);

            {
                IGPUDescriptorSet::SDescriptorInfo infos[2];
                infos[0].desc = smart_refctd_ptr(gridDataBuffer);
                infos[0].info.buffer = {.offset = 0, .size = gridDataBuffer->getSize()};
                infos[1].desc = gridCellMaterialImageView;
                infos[1].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                infos[1].info.combinedImageSampler.sampler = nullptr;

                IGPUDescriptorSet::SDescriptorInfo imgInfosVel0[3];
                imgInfosVel0[0].desc = velocityFieldImageViews[0];
                imgInfosVel0[0].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                imgInfosVel0[0].info.combinedImageSampler.sampler = nullptr;
                imgInfosVel0[1].desc = velocityFieldImageViews[1];
                imgInfosVel0[1].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                imgInfosVel0[1].info.combinedImageSampler.sampler = nullptr;
                imgInfosVel0[2].desc = velocityFieldImageViews[2];
                imgInfosVel0[2].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                imgInfosVel0[2].info.combinedImageSampler.sampler = nullptr;

                IGPUDescriptorSet::SWriteDescriptorSet writes[3] = {
                    {.dstSet = m_applyForcesDs.get(), .binding = b_abfGridData, .arrayElement = 0, .count = 1, .info = &infos[0]},
                    {.dstSet = m_applyForcesDs.get(), .binding = b_abfVelField, .arrayElement = 0, .count = 3, .info = imgInfosVel0},
                    {.dstSet = m_applyForcesDs.get(), .binding = b_abfCM, .arrayElement = 0, .count = 1, .info = &infos[1]},
                };
                m_device->updateDescriptorSets(std::span(writes, 3), {});
            }
        }
        // apply diffusion pipelines
        {
            createComputePipeline.operator()<"diffusion">(m_axisCellsPipeline, m_axisCellsPool, m_axisCellsDs, 
                "setAxisCellMaterial", dAxisCM_bs1);

            {
                IGPUDescriptorSet::SDescriptorInfo infos[3];
                infos[0].desc = smart_refctd_ptr(gridDataBuffer);
                infos[0].info.buffer = {.offset = 0, .size = gridDataBuffer->getSize()};
                infos[1].desc = gridCellMaterialImageView;
                infos[1].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                infos[1].info.combinedImageSampler.sampler = nullptr;
                infos[2].desc = tempAxisCellMaterialImageView;
                infos[2].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                infos[2].info.combinedImageSampler.sampler = nullptr;
                IGPUDescriptorSet::SWriteDescriptorSet writes[3] = {
                    {.dstSet = m_axisCellsDs.get(), .binding = b_dGridData, .arrayElement = 0, .count = 1, .info = &infos[0]},
                    {.dstSet = m_axisCellsDs.get(), .binding = b_dCM, .arrayElement = 0, .count = 1, .info = &infos[1]},
                    {.dstSet = m_axisCellsDs.get(), .binding = b_dAxisOut, .arrayElement = 0, .count = 1, .info = &infos[2]},
                };
                m_device->updateDescriptorSets(std::span(writes, 3), {});
            }
        }
        {
            createComputePipeline.operator()<"diffusion">(m_neighborAxisCellsPipeline, m_neighborAxisCellsPool, m_neighborAxisCellsDs, 
                "setNeighborAxisCellMaterial", dNeighborAxisCM_bs1);

            {
                IGPUDescriptorSet::SDescriptorInfo infos[3];
                infos[0].desc = smart_refctd_ptr(gridDataBuffer);
                infos[0].info.buffer = {.offset = 0, .size = gridDataBuffer->getSize()};				
                infos[1].desc = tempAxisCellMaterialImageView;
                infos[1].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                infos[1].info.combinedImageSampler.sampler = nullptr;
                infos[2].desc = gridAxisCellMaterialImageView;
                infos[2].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                infos[2].info.combinedImageSampler.sampler = nullptr;
                IGPUDescriptorSet::SWriteDescriptorSet writes[3] = {
                    {.dstSet = m_neighborAxisCellsDs.get(), .binding = b_dGridData, .arrayElement = 0, .count = 1, .info = &infos[0]},
                    {.dstSet = m_neighborAxisCellsDs.get(), .binding = b_dAxisIn, .arrayElement = 0, .count = 1, .info = &infos[1]},
                    {.dstSet = m_neighborAxisCellsDs.get(), .binding = b_dAxisOut, .arrayElement = 0, .count = 1, .info = &infos[2]},
                };
                m_device->updateDescriptorSets(std::span(writes, 3), {});
            }
        }
        {
            smart_refctd_ptr<IShader> diffusion = loadPrecompiledShader<"diffusion">(); // "app_resources/compute/diffusion.comp.hlsl"

            auto descriptorSetLayout1 = m_device->createDescriptorSetLayout(dDiffuse_bs1);

            const std::array<IGPUDescriptorSetLayout*, ICPUPipelineLayout::DESCRIPTOR_SET_COUNT> dscLayoutPtrs = {
                nullptr,
                descriptorSetLayout1.get()
            };
            const uint32_t setCounts[2u] = { 0u, 2u };
            m_diffusionPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_UPDATE_AFTER_BIND_BIT, std::span(dscLayoutPtrs.begin(), dscLayoutPtrs.end()), setCounts);
            m_diffusionDs = m_diffusionPool->createDescriptorSet(descriptorSetLayout1);

            const asset::SPushConstantRange pcRange = { .stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE, .offset = 0, .size = 4 * sizeof(uint32_t) };

            smart_refctd_ptr<nbl::video::IGPUPipelineLayout> pipelineLayout = m_device->createPipelineLayout({ &pcRange, 1 }, nullptr, smart_refctd_ptr(descriptorSetLayout1), nullptr, nullptr);

            {
                IGPUComputePipeline::SCreationParams params = {};
                params.layout = pipelineLayout.get();
                params.shader.entryPoint = "iterateDiffusion";
                params.shader.shader = diffusion.get();
                if (m_device->getEnabledFeatures().pipelineExecutableInfo)
                {
                    params.flags |= IGPUComputePipeline::SCreationParams::FLAGS::CAPTURE_STATISTICS;
                    params.flags |= IGPUComputePipeline::SCreationParams::FLAGS::CAPTURE_INTERNAL_REPRESENTATIONS;
                }
                if (!m_device->createComputePipelines(nullptr, { &params,1 }, &m_iterateDiffusionPipeline))
					m_logger->log("Failed to create iterateDiffusion pipeline!\n");

                if (m_device->getEnabledFeatures().pipelineExecutableInfo)
                {
                    auto report = m_device->getPipelineExecutableReport(m_iterateDiffusionPipeline.get(), true);
                    m_logger->log("iterateDiffusion Pipeline Executable Report:\n%s", ILogger::ELL_PERFORMANCE, report.c_str());
                }
            }
            {
                IGPUComputePipeline::SCreationParams params = {};
                params.layout = pipelineLayout.get();
                params.shader.entryPoint = "applyDiffusion";
                params.shader.shader = diffusion.get();
                if (m_device->getEnabledFeatures().pipelineExecutableInfo)
                {
                    params.flags |= IGPUComputePipeline::SCreationParams::FLAGS::CAPTURE_STATISTICS;
                    params.flags |= IGPUComputePipeline::SCreationParams::FLAGS::CAPTURE_INTERNAL_REPRESENTATIONS;
                }
                if (!m_device->createComputePipelines(nullptr, { &params,1 }, &m_diffusionPipeline))
					m_logger->log("Failed to create applyDiffusion pipeline!\n");

                if (m_device->getEnabledFeatures().pipelineExecutableInfo)
                {
                    auto report = m_device->getPipelineExecutableReport(m_diffusionPipeline.get(), true);
                    m_logger->log("applyDiffusion Pipeline Executable Report:\n%s", ILogger::ELL_PERFORMANCE, report.c_str());
                }
            }

            {
                IGPUDescriptorSet::SDescriptorInfo infos[4];
                infos[0].desc = smart_refctd_ptr(gridDataBuffer);
                infos[0].info.buffer = {.offset = 0, .size = gridDataBuffer->getSize()};				
                infos[1].desc = gridCellMaterialImageView;
                infos[1].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                infos[1].info.combinedImageSampler.sampler = nullptr;
                infos[2].desc = gridAxisCellMaterialImageView;
                infos[2].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                infos[2].info.combinedImageSampler.sampler = nullptr;
                infos[3].desc = gridDiffusionImageView;
                infos[3].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                infos[3].info.combinedImageSampler.sampler = nullptr;

                IGPUDescriptorSet::SDescriptorInfo imgInfosVel0[3];
                imgInfosVel0[0].desc = velocityFieldImageViews[0];
                imgInfosVel0[0].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                imgInfosVel0[0].info.combinedImageSampler.sampler = nullptr;
                imgInfosVel0[1].desc = velocityFieldImageViews[1];
                imgInfosVel0[1].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                imgInfosVel0[1].info.combinedImageSampler.sampler = nullptr;
                imgInfosVel0[2].desc = velocityFieldImageViews[2];
                imgInfosVel0[2].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                imgInfosVel0[2].info.combinedImageSampler.sampler = nullptr;

                IGPUDescriptorSet::SWriteDescriptorSet writes[5] = {
                    {.dstSet = m_diffusionDs.get(), .binding = b_dGridData, .arrayElement = 0, .count = 1, .info = &infos[0]},
                    {.dstSet = m_diffusionDs.get(), .binding = b_dCM, .arrayElement = 0, .count = 1, .info = &infos[1]},
                    {.dstSet = m_diffusionDs.get(), .binding = b_dVel, .arrayElement = 0, .count = 3, .info = imgInfosVel0},
                    {.dstSet = m_diffusionDs.get(), .binding = b_dAxisIn, .arrayElement = 0, .count = 1, .info = &infos[2]},
                    {.dstSet = m_diffusionDs.get(), .binding = b_dDiff, .arrayElement = 0, .count = 1, .info = &infos[3]}
                };
                m_device->updateDescriptorSets(std::span(writes, 5), {});
            }
        }
        // solve pressure system pipelines
        {
            createComputePipeline.operator()<"pressureSolver">(m_calcDivergencePipeline, m_calcDivergencePool, m_calcDivergenceDs, 
                "calculateNegativeDivergence", psDivergence_bs1);

            {
                IGPUDescriptorSet::SDescriptorInfo infos[3];
                infos[0].desc = smart_refctd_ptr(gridDataBuffer);
                infos[0].info.buffer = {.offset = 0, .size = gridDataBuffer->getSize()};
                infos[1].desc = gridCellMaterialImageView;
                infos[1].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                infos[1].info.combinedImageSampler.sampler = nullptr;
                infos[2].desc = divergenceImageView;
                infos[2].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                infos[2].info.combinedImageSampler.sampler = nullptr;

                IGPUDescriptorSet::SDescriptorInfo imgInfosVel0[3];
                imgInfosVel0[0].desc = velocityFieldImageViews[0];
                imgInfosVel0[0].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                imgInfosVel0[0].info.combinedImageSampler.sampler = nullptr;
                imgInfosVel0[1].desc = velocityFieldImageViews[1];
                imgInfosVel0[1].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                imgInfosVel0[1].info.combinedImageSampler.sampler = nullptr;
                imgInfosVel0[2].desc = velocityFieldImageViews[2];
                imgInfosVel0[2].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                imgInfosVel0[2].info.combinedImageSampler.sampler = nullptr;

                IGPUDescriptorSet::SWriteDescriptorSet writes[4] = {
                    {.dstSet = m_calcDivergenceDs.get(), .binding = b_psGridData, .arrayElement = 0, .count = 1, .info = &infos[0]},
                    {.dstSet = m_calcDivergenceDs.get(), .binding = b_psCM, .arrayElement = 0, .count = 1, .info = &infos[1]},
                    {.dstSet = m_calcDivergenceDs.get(), .binding = b_psVel, .arrayElement = 0, .count = 3, .info = imgInfosVel0},
                    {.dstSet = m_calcDivergenceDs.get(), .binding = b_psDiv, .arrayElement = 0, .count = 1, .info = &infos[2]},
                };
                m_device->updateDescriptorSets(std::span(writes, 4), {});
            }
        }
        {
            createComputePipeline.operator()<"pressureSolver">(m_iteratePressurePipeline, m_iteratePressurePool, m_iteratePressureDs,
                "iteratePressureSystem", psIteratePressure_bs1);

            {
                IGPUDescriptorSet::SDescriptorInfo infos[5];
                infos[0].desc = smart_refctd_ptr(gridDataBuffer);
                infos[0].info.buffer = { .offset = 0, .size = gridDataBuffer->getSize() };
                infos[1].desc = smart_refctd_ptr(pressureParamsBuffer);
                infos[1].info.buffer = { .offset = 0, .size = pressureParamsBuffer->getSize() };
                infos[2].desc = gridCellMaterialImageView;
                infos[2].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                infos[2].info.combinedImageSampler.sampler = nullptr;
                infos[3].desc = divergenceImageView;
                infos[3].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                infos[3].info.combinedImageSampler.sampler = nullptr;
                infos[4].desc = pressureImageView;
                infos[4].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                infos[4].info.combinedImageSampler.sampler = nullptr;
                IGPUDescriptorSet::SWriteDescriptorSet writes[5] = {
                    {.dstSet = m_iteratePressureDs.get(), .binding = b_psGridData, .arrayElement = 0, .count = 1, .info = &infos[0]},
                    {.dstSet = m_iteratePressureDs.get(), .binding = b_psParams, .arrayElement = 0, .count = 1, .info = &infos[1]},
                    {.dstSet = m_iteratePressureDs.get(), .binding = b_psCM, .arrayElement = 0, .count = 1, .info = &infos[2]},
                    {.dstSet = m_iteratePressureDs.get(), .binding = b_psDiv, .arrayElement = 0, .count = 1, .info = &infos[3]},
                    {.dstSet = m_iteratePressureDs.get(), .binding = b_psPres, .arrayElement = 0, .count = 1, .info = &infos[4]},
                };
                m_device->updateDescriptorSets(std::span(writes, 5), {});
            }
        }
        {
            createComputePipeline.operator()<"pressureSolver">(m_updateVelPsPipeline, m_updateVelPsPool, m_updateVelPsDs, 
                "updateVelocities", psUpdateVelPs_bs1);

            {
                IGPUDescriptorSet::SDescriptorInfo infos[4];
                infos[0].desc = smart_refctd_ptr(gridDataBuffer);
                infos[0].info.buffer = {.offset = 0, .size = gridDataBuffer->getSize()};
                infos[1].desc = smart_refctd_ptr(pressureParamsBuffer);
                infos[1].info.buffer = {.offset = 0, .size = pressureParamsBuffer->getSize()};
                infos[2].desc = gridCellMaterialImageView;
                infos[2].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                infos[2].info.combinedImageSampler.sampler = nullptr;
                infos[3].desc = pressureImageView;
                infos[3].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                infos[3].info.combinedImageSampler.sampler = nullptr;

                IGPUDescriptorSet::SDescriptorInfo imgInfosVel0[3];
                imgInfosVel0[0].desc = velocityFieldImageViews[0];
                imgInfosVel0[0].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                imgInfosVel0[0].info.combinedImageSampler.sampler = nullptr;
                imgInfosVel0[1].desc = velocityFieldImageViews[1];
                imgInfosVel0[1].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                imgInfosVel0[1].info.combinedImageSampler.sampler = nullptr;
                imgInfosVel0[2].desc = velocityFieldImageViews[2];
                imgInfosVel0[2].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                imgInfosVel0[2].info.combinedImageSampler.sampler = nullptr;

                IGPUDescriptorSet::SWriteDescriptorSet writes[5] = {
                    {.dstSet = m_updateVelPsDs.get(), .binding = b_psGridData, .arrayElement = 0, .count = 1, .info = &infos[0]},
                    {.dstSet = m_updateVelPsDs.get(), .binding = b_psParams, .arrayElement = 0, .count = 1, .info = &infos[1]},
                    {.dstSet = m_updateVelPsDs.get(), .binding = b_psCM, .arrayElement = 0, .count = 1, .info = &infos[2]},
                    {.dstSet = m_updateVelPsDs.get(), .binding = b_psVel, .arrayElement = 0, .count = 3, .info = imgInfosVel0},
                    {.dstSet = m_updateVelPsDs.get(), .binding = b_psPres, .arrayElement = 0, .count = 1, .info = &infos[3]},
                };
                m_device->updateDescriptorSets(std::span(writes, 5), {});
            }
        }
        {
            // advect particles pipeline
            const asset::SPushConstantRange pcRange = { .stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE, .offset = 0, .size = 2 * sizeof(uint64_t) };
            createComputePipeline.operator()<"advectParticles">(m_advectParticlesPipeline, m_advectParticlesPool, m_advectParticlesDs,
                "main", apAdvectParticles_bs1, pcRange);

            {
                IGPUDescriptorSet::SDescriptorInfo infos[2];
                infos[0].desc = smart_refctd_ptr(gridDataBuffer);
                infos[0].info.buffer = {.offset = 0, .size = gridDataBuffer->getSize()};
                infos[1].desc = velocityFieldSampler;

                IGPUDescriptorSet::SDescriptorInfo imgInfosVel0[3];
                imgInfosVel0[0].desc = velocityFieldImageViews[0];
                imgInfosVel0[0].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                imgInfosVel0[0].info.combinedImageSampler.sampler = nullptr;
                imgInfosVel0[1].desc = velocityFieldImageViews[1];
                imgInfosVel0[1].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                imgInfosVel0[1].info.combinedImageSampler.sampler = nullptr;
                imgInfosVel0[2].desc = velocityFieldImageViews[2];
                imgInfosVel0[2].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                imgInfosVel0[2].info.combinedImageSampler.sampler = nullptr;

                IGPUDescriptorSet::SDescriptorInfo imgInfosVel1[3];
                imgInfosVel1[0].desc = prevVelocityFieldImageViews[0];
                imgInfosVel1[0].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                imgInfosVel1[0].info.combinedImageSampler.sampler = nullptr;
                imgInfosVel1[1].desc = prevVelocityFieldImageViews[1];
                imgInfosVel1[1].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                imgInfosVel1[1].info.combinedImageSampler.sampler = nullptr;
                imgInfosVel1[2].desc = prevVelocityFieldImageViews[2];
                imgInfosVel1[2].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                imgInfosVel1[2].info.combinedImageSampler.sampler = nullptr;

                IGPUDescriptorSet::SWriteDescriptorSet writes[4] = {
                    {.dstSet = m_advectParticlesDs.get(), .binding = b_apGridData, .arrayElement = 0, .count = 1, .info = &infos[0]},
                    {.dstSet = m_advectParticlesDs.get(), .binding = b_apVelField, .arrayElement = 0, .count = 3, .info = imgInfosVel0},
                    {.dstSet = m_advectParticlesDs.get(), .binding = b_apPrevVelField, .arrayElement = 0, .count = 3, .info = imgInfosVel1},
                    {.dstSet = m_advectParticlesDs.get(), .binding = b_apVelSampler, .arrayElement = 0, .count = 1, .info = &infos[1]}
                };
                m_device->updateDescriptorSets(std::span(writes, 4), {});
            }
        }

        m_winMgr->show(m_window.get());

        return true;
    }

    inline void workLoopBody() override
    {
        const uint32_t framesInFlight = core::min(MaxFramesInFlight, m_surface->getMaxAcquiresInFlight());

        if (m_realFrameIx >= framesInFlight)
        {
            const ISemaphore::SWaitInfo cbDonePending[] =
            {
                {
                    .semaphore = m_renderSemaphore.get(),
                    .value = m_realFrameIx + 1 - framesInFlight
                }
            };
            if (m_device->blockForSemaphores(cbDonePending) != ISemaphore::WAIT_RESULT::SUCCESS)
                return;
        }

        const auto resourceIx = m_realFrameIx % MaxFramesInFlight;

        m_inputSystem->getDefaultMouse(&mouse);
        m_inputSystem->getDefaultKeyboard(&keyboard);

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

        auto* const cmdbuf = m_cmdBufs.data()[resourceIx].get();
        cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
        cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
        cmdbuf->beginDebugMarker("Frame Debug FLIP sim begin");
        {
            camera.beginInputProcessing(nextPresentationTimestamp);
            mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { camera.mouseProcess(events); mouseProcess(events); }, m_logger.get());
            keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void { camera.keyboardProcess(events); }, m_logger.get());
            camera.endInputProcessing(nextPresentationTimestamp);
        }

        // TODO: also need to protect from previous frame still reading while we overwrite UBO

        SMVPParams camData;
        SBufferRange<IGPUBuffer> camDataRange;
        {
            const auto viewMatrix = camera.getViewMatrix();
            const auto projectionMatrix = camera.getProjectionMatrix();
            const auto viewProjectionMatrix = camera.getConcatenatedMatrix();

            hlsl::float32_t3x4 modelMatrix = hlsl::math::linalg::identity<float32_t3x4>();

            hlsl::float32_t3x4 modelViewMatrix = viewMatrix;
            hlsl::float32_t4x4 modelViewProjectionMatrix = viewProjectionMatrix;

            auto modelMat = hlsl::math::linalg::promote_affine<4, 4, 3, 4>(modelMatrix);

            const core::vector3df camPos = camera.getPosition().getAsVector3df();

            camPos.getAs4Values(camData.cameraPosition);
            memcpy(camData.MVP, &modelViewProjectionMatrix[0][0], sizeof(camData.MVP));
            memcpy(camData.M, &modelMat[0][0], sizeof(camData.M));
            memcpy(camData.V, &viewMatrix[0][0], sizeof(camData.V));
            memcpy(camData.P, &projectionMatrix[0][0], sizeof(camData.P));
            {
                camDataRange.buffer = cameraBuffer;
                camDataRange.size = cameraBuffer->getSize();

                cmdbuf->updateBuffer(camDataRange, &camData);
            }
        }

        bool bCaptureTestInitParticles = false;
        float32_t4 pressureSolverParams[2];
        SBufferRange<IGPUBuffer> gridDataRange;
        SBufferRange<IGPUBuffer> pParamsRange;
        SBufferRange<IGPUBuffer> pressureParamsRange;
        if (m_shouldInitParticles) // TODO: why on earth is this in `workLoopBody()` and not `onAppInitialized` ?
        {
            bCaptureTestInitParticles = true;

            {
                gridDataRange.size = gridDataBuffer->getSize();
                gridDataRange.buffer = gridDataBuffer;
            }
            cmdbuf->updateBuffer(gridDataRange, &m_gridData);

            {
                pParamsRange.size = pParamsBuffer->getSize();
                pParamsRange.buffer = pParamsBuffer;
            }
            cmdbuf->updateBuffer(pParamsRange, &m_pRenderParams);

            float a = m_gridData.gridInvCellSize * m_gridData.gridInvCellSize;
            float b = 1.f / (2.f * (a * 3));
            pressureSolverParams[0] = float32_t4(b * a, b * a, b * a, -b);
            pressureSolverParams[1] = float32_t4(m_gridData.gridInvCellSize);

            {
                pressureParamsRange.size = pressureParamsBuffer->getSize();
                pressureParamsRange.buffer = pressureParamsBuffer;
            }
            cmdbuf->updateBuffer(pressureParamsRange, &pressureSolverParams);

            initializeParticles(cmdbuf);

            transitionGridImageLayouts(cmdbuf);

            // TODO: fat pipeline barrier! The bottom one only ever protected the UBO write.
        }

        {
            SMemoryBarrier memBarrier;
            memBarrier.srcStageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS;
            memBarrier.srcAccessMask = ACCESS_FLAGS::MEMORY_WRITE_BITS;
            //memBarrier.srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT; // after the initialization with compute shaders is moved out
            memBarrier.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
            memBarrier.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;
            cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, {.memBarriers = {&memBarrier, 1}});
        }

        // simulation steps
        for (uint32_t i = 0; i < m_substepsPerFrame; i++)
        {
            dispatchUpdateFluidCells(cmdbuf);			// particle to grid
            dispatchApplyBodyForces(cmdbuf, i == 0);	// external forces, e.g. gravity
            dispatchApplyDiffusion(cmdbuf);
            dispatchApplyPressure(cmdbuf);
            dispatchAdvection(cmdbuf);				// update/advect fluid
        }

    // TODO: remove the compute shader generating vertices, collapse the two barriers around it into one. Need to think about next frame compute/transfer stepping on our toes.
    // The pipeline barrier shouldn't really be here and should be expressed through a subpass external dependency
        {
            SMemoryBarrier memBarrier;
            memBarrier.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
            memBarrier.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
            memBarrier.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
            memBarrier.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;
            cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, {.memBarriers = {&memBarrier, 1}});
        }

        // prepare particle vertices for render
        {
            const uint64_t bufferAddr[3] = {
                particleData.positionBuffer->getDeviceAddress(),
                particleData.velocityBuffer->getDeviceAddress(),
                particleVertexBuffer->getDeviceAddress()
            };

            cmdbuf->bindComputePipeline(m_genParticleVerticesPipeline.get());
            cmdbuf->pushConstants(m_genParticleVerticesPipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0, 3 * sizeof(uint64_t), bufferAddr);
            cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_genParticleVerticesPipeline->getLayout(), 1, 1, &m_genVerticesDs.get());
            cmdbuf->dispatch(WorkgroupCountParticles, 1, 1);
        }

        {
            SMemoryBarrier memBarrier;
            memBarrier.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
            memBarrier.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
            memBarrier.dstStageMask = PIPELINE_STAGE_FLAGS::VERTEX_INPUT_BITS;
            memBarrier.dstAccessMask = ACCESS_FLAGS::MEMORY_READ_BITS;
            cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, {.memBarriers = {&memBarrier, 1}});
        }

        // draw particles
        auto* queue = getGraphicsQueue();

        asset::SViewport viewport;
        {
            viewport.minDepth = 1.f;
            viewport.maxDepth = 0.f;
            viewport.x = 0u;
            viewport.y = 0u;
            viewport.width = m_window->getWidth();
            viewport.height = m_window->getHeight();
        }
        cmdbuf->setViewport(0u, 1u, &viewport);

        VkRect2D scissor{
            .offset = { 0, 0 },
            .extent = { m_window->getWidth(), m_window->getHeight() }
        };
        cmdbuf->setScissor(0u, 1u, &scissor);		

        IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
        VkRect2D currentRenderArea;
        const IGPUCommandBuffer::SClearColorValue clearValue = { .float32 = {0.f,0.f,0.f,1.f} };
        const IGPUCommandBuffer::SClearDepthStencilValue depthValue = { .depth = 0.f };
        {
            currentRenderArea =
            {
                .offset = {0,0},
                .extent = {m_window->getWidth(),m_window->getHeight()}
            };
    
            auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
            beginInfo =
            {
                .framebuffer = scRes->getFramebuffer(m_currentImageAcquire.imageIndex),
                .colorClearValues = &clearValue,
                .depthStencilClearValues = &depthValue,
                .renderArea = currentRenderArea
            };
        }
        cmdbuf->beginRenderPass(beginInfo, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);

        const uint64_t bufferAddr = particleVertexBuffer->getDeviceAddress();

        cmdbuf->bindGraphicsPipeline(m_graphicsPipeline.get());
        cmdbuf->pushConstants(m_graphicsPipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_VERTEX, 0, sizeof(uint64_t), &bufferAddr);
        cmdbuf->bindDescriptorSets(EPBP_GRAPHICS, m_graphicsPipeline->getLayout(), 1, 1, &m_renderDs.get());

        // TODO: INDEXED and INSTANCED DRAWS!
        cmdbuf->draw(numParticles * 6, 1, 0, 0);

        cmdbuf->endRenderPass();

        // turn into a subpass external dst dependency
        {
            SMemoryBarrier memBarrier;
            memBarrier.srcStageMask = PIPELINE_STAGE_FLAGS::ALL_GRAPHICS_BITS;
            memBarrier.srcAccessMask = ACCESS_FLAGS::MEMORY_WRITE_BITS;
            memBarrier.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
            memBarrier.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS; // TODO: also write bits
            cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, {.memBarriers = {&memBarrier, 1}});
        }

        cmdbuf->endDebugMarker();
        cmdbuf->end();

        // submit
        {
            const IQueue::SSubmitInfo::SSemaphoreInfo rendered[] =
            {
                {
                    .semaphore = m_renderSemaphore.get(),
                    .value = ++m_realFrameIx,
                    .stageMask = PIPELINE_STAGE_FLAGS::ALL_GRAPHICS_BITS
                }
            };
            {
                {
                    const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[] =
                    {
                        { .cmdbuf = cmdbuf }
                    };

                    const IQueue::SSubmitInfo::SSemaphoreInfo acquired[] =
                    {
                        {
                            .semaphore = m_currentImageAcquire.semaphore,
                            .value = m_currentImageAcquire.acquireCount,
                            .stageMask = PIPELINE_STAGE_FLAGS::NONE
                        }
                    };
                    const IQueue::SSubmitInfo infos[] =
                    {
                        {
                            .waitSemaphores = acquired,
                            .commandBuffers = commandBuffers,
                            .signalSemaphores = rendered
                        }
                    };

                    if (queue->submit(infos) == IQueue::RESULT::SUCCESS)
                    {
                        const nbl::video::ISemaphore::SWaitInfo waitInfos[] =
                        { {
                            .semaphore = m_renderSemaphore.get(),
                            .value = m_realFrameIx
                        } };

                        m_device->blockForSemaphores(waitInfos); // this is not solution, quick wa to not throw validation errors
                    }
                    else
                        --m_realFrameIx;
                }
            }

            m_surface->present(m_currentImageAcquire.imageIndex, rendered);
        }
    }

    inline bool keepRunning() override
    {
        if (m_surface->irrecoverable())
            return false;

        return true;
    }

    inline bool onAppTerminated() override
    {
        return device_base_t::onAppTerminated();
    }

    void dispatchUpdateFluidCells(IGPUCommandBuffer* cmdbuf)
    {
        {
            SMemoryBarrier memBarrier;
            memBarrier.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
            memBarrier.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
            memBarrier.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
            memBarrier.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;
            cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .memBarriers = {&memBarrier, 1} });
        }

        // clear velocity stuffs
        IGPUCommandBuffer::SClearColorValue clear = {};

        asset::IImage::SSubresourceRange subresourceRange = {};
        subresourceRange.aspectMask = asset::IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
        subresourceRange.baseArrayLayer = 0u;
        subresourceRange.baseMipLevel = 0u;
        subresourceRange.layerCount = 1u;
        subresourceRange.levelCount = 1u;

        for (uint32_t i = 0; i < 3; i++)
        {
            cmdbuf->clearColorImage(velocityFieldImageViews[i]->getCreationParameters().image.get(),
                asset::IImage::LAYOUT::GENERAL, &clear, 1, &subresourceRange);

            cmdbuf->clearColorImage(prevVelocityFieldImageViews[i]->getCreationParameters().image.get(),
                asset::IImage::LAYOUT::GENERAL, &clear, 1, &subresourceRange);
        }

        cmdbuf->clearColorImage(gridParticleCountImageView->getCreationParameters().image.get(),
            asset::IImage::LAYOUT::GENERAL, &clear, 1, &subresourceRange);

        {
            SMemoryBarrier memBarrier;
            memBarrier.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
            memBarrier.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
            memBarrier.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
            memBarrier.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;
            cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .memBarriers = {&memBarrier, 1} });
        }

        const uint64_t bufferAddr[2] = {
            particleData.positionBuffer->getDeviceAddress(),
            particleData.velocityBuffer->getDeviceAddress()
        };

        cmdbuf->bindComputePipeline(m_accumulateWeightsPipeline.get());
        cmdbuf->pushConstants(m_accumulateWeightsPipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0, 2 * sizeof(uint64_t), bufferAddr);
        cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_accumulateWeightsPipeline->getLayout(), 1, 1, &m_accumulateWeightsDs.get());
        cmdbuf->dispatch(WorkgroupCountParticles, 1, 1);

        {
            SMemoryBarrier memBarrier;
            memBarrier.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
            memBarrier.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
            memBarrier.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
            memBarrier.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;
            cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, {.memBarriers = {&memBarrier, 1}});
        }
        
        cmdbuf->bindComputePipeline(m_updateFluidCellsPipeline.get());
        cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_updateFluidCellsPipeline->getLayout(), 1, 1, &m_updateFluidCellsDs.get());
        cmdbuf->dispatch(WorkgroupCountGrid.x, WorkgroupCountGrid.y, WorkgroupCountGrid.z);

        {
            SMemoryBarrier memBarrier;
            memBarrier.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
            memBarrier.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
            memBarrier.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
            memBarrier.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;
            cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, {.memBarriers = {&memBarrier, 1}});
        }

        cmdbuf->bindComputePipeline(m_updateNeighborCellsPipeline.get());
        cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_updateNeighborCellsPipeline->getLayout(), 1, 1, &m_updateNeighborCellsDs.get());
        cmdbuf->dispatch(WorkgroupCountGrid.x, WorkgroupCountGrid.y, WorkgroupCountGrid.z);
    }
    
    void dispatchApplyBodyForces(IGPUCommandBuffer* cmdbuf, bool isFirstSubstep)
    {
        {
            SMemoryBarrier memBarrier;
            memBarrier.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
            memBarrier.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
            memBarrier.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
            memBarrier.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;
            cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, {.memBarriers = {&memBarrier, 1}});
        }

        cmdbuf->bindComputePipeline(m_applyBodyForcesPipeline.get());
        cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_applyBodyForcesPipeline->getLayout(), 1, 1, &m_applyForcesDs.get());
        cmdbuf->dispatch(WorkgroupCountGrid.x, WorkgroupCountGrid.y, WorkgroupCountGrid.z);
    }
    
    void dispatchApplyDiffusion(IGPUCommandBuffer* cmdbuf)
    {
        if (viscosity <= 0.f)
            return;

        {
            SMemoryBarrier memBarrier;
            memBarrier.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
            memBarrier.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
            memBarrier.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
            memBarrier.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;
            cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, {.memBarriers = {&memBarrier, 1}});
        }

        cmdbuf->bindComputePipeline(m_axisCellsPipeline.get());
        cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_axisCellsPipeline->getLayout(), 1, 1, &m_axisCellsDs.get());
        cmdbuf->dispatch(WorkgroupCountGrid.x, WorkgroupCountGrid.y, WorkgroupCountGrid.z);

        {
            SMemoryBarrier memBarrier;
            memBarrier.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
            memBarrier.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
            memBarrier.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
            memBarrier.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;
            cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, {.memBarriers = {&memBarrier, 1}});
        }

        cmdbuf->bindComputePipeline(m_neighborAxisCellsPipeline.get());
        cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_neighborAxisCellsPipeline->getLayout(), 1, 1, &m_neighborAxisCellsDs.get());
        cmdbuf->dispatch(WorkgroupCountGrid.x, WorkgroupCountGrid.y, WorkgroupCountGrid.z);

        float a = viscosity * deltaTime;
        float32_t3 b = float32_t3(m_gridData.gridInvCellSize * m_gridData.gridInvCellSize);
        float c = 1.f / (1.f + 2.f *(b.x + b.y + b.z) * a);
        float32_t4 diffParam = {};	// as push constant
        diffParam.xyz = a * b * c;
        diffParam.w = c;

        cmdbuf->bindComputePipeline(m_iterateDiffusionPipeline.get());
        cmdbuf->pushConstants(m_iterateDiffusionPipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0, sizeof(float32_t4), &diffParam);
        cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_iterateDiffusionPipeline->getLayout(), 1, 1, &m_diffusionDs.get());
        for (int i = 0; i < diffusionIterations; i++)
        {
            {
                SMemoryBarrier memBarrier;
                memBarrier.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
                memBarrier.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
                memBarrier.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
                memBarrier.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;
                cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, {.memBarriers = {&memBarrier, 1}});
            }

            cmdbuf->dispatch(WorkgroupCountGrid.x, WorkgroupCountGrid.y, WorkgroupCountGrid.z);
        }
        
        {
            SMemoryBarrier memBarrier;
            memBarrier.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
            memBarrier.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
            memBarrier.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
            memBarrier.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;
            cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, {.memBarriers = {&memBarrier, 1}});
        }

        cmdbuf->bindComputePipeline(m_diffusionPipeline.get());
        cmdbuf->pushConstants(m_diffusionPipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0, sizeof(float32_t4), &diffParam);
        cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_diffusionPipeline->getLayout(), 1, 1, &m_diffusionDs.get());
        cmdbuf->dispatch(WorkgroupCountGrid.x, WorkgroupCountGrid.y, WorkgroupCountGrid.z);
    }
    
    void dispatchApplyPressure(IGPUCommandBuffer* cmdbuf)
    {
        {
            SMemoryBarrier memBarrier;
            memBarrier.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
            memBarrier.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
            memBarrier.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
            memBarrier.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;
            cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, {.memBarriers = {&memBarrier, 1}});
        }

        cmdbuf->bindComputePipeline(m_calcDivergencePipeline.get());
        cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_calcDivergencePipeline->getLayout(), 1, 1, &m_calcDivergenceDs.get());
        cmdbuf->dispatch(WorkgroupCountGrid.x, WorkgroupCountGrid.y, WorkgroupCountGrid.z);

        cmdbuf->bindComputePipeline(m_iteratePressurePipeline.get());
        cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_iteratePressurePipeline->getLayout(), 1, 1, &m_iteratePressureDs.get());
        for (int i = 0; i < pressureSolverIterations; i++)
        {
            {
                SMemoryBarrier memBarrier;
                memBarrier.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
                memBarrier.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
                memBarrier.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
                memBarrier.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;
                cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, {.memBarriers = {&memBarrier, 1}});
            }

            cmdbuf->dispatch(WorkgroupCountGrid.x, WorkgroupCountGrid.y, WorkgroupCountGrid.z);
        }

        {
            SMemoryBarrier memBarrier;
            memBarrier.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
            memBarrier.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
            memBarrier.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
            memBarrier.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;
            cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, {.memBarriers = {&memBarrier, 1}});
        }

        cmdbuf->bindComputePipeline(m_updateVelPsPipeline.get());
        cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_updateVelPsPipeline->getLayout(), 1, 1, &m_updateVelPsDs.get());
        cmdbuf->dispatch(WorkgroupCountGrid.x, WorkgroupCountGrid.y, WorkgroupCountGrid.z);
    }
            
    void dispatchAdvection(IGPUCommandBuffer* cmdbuf)
    {
        {
            SMemoryBarrier memBarrier;
            memBarrier.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
            memBarrier.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
            memBarrier.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
            memBarrier.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;
            cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, {.memBarriers = {&memBarrier, 1}});
        }

        const uint64_t bufferAddr[2] = {
            particleData.positionBuffer->getDeviceAddress(),
            particleData.velocityBuffer->getDeviceAddress()
        };

        cmdbuf->bindComputePipeline(m_advectParticlesPipeline.get());
        cmdbuf->pushConstants(m_advectParticlesPipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0, 2 * sizeof(uint64_t), bufferAddr);
        cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_advectParticlesPipeline->getLayout(), 1, 1, &m_advectParticlesDs.get());
        cmdbuf->dispatch(WorkgroupCountParticles, 1, 1);
    }

private:
    void usePreset(SimPresets preset)
    {
        m_gridData.gridCellSize = 0.25f;
        m_gridData.gridInvCellSize = 1.f / m_gridData.gridCellSize;

        switch (preset)
        {
        case LONG_BOX:
            m_gridData.gridSize = int32_t4{48, 24, 24, 0};
            m_gridData.particleInitMin = int32_t4{4, 4, 4, 0};
            m_gridData.particleInitMax = int32_t4{20, 20, 20, 0};
            break;
        case CENTER_DROP:
        default:
            m_gridData.gridSize = int32_t4{32, 32, 32, 0};
            m_gridData.particleInitMin = int32_t4{4, 12, 4, 0};
            m_gridData.particleInitMax = int32_t4{28, 28, 28, 0};
            break;
        }
        
        fillGridData();
    }

    void fillGridData()
    {
        m_gridData.particleInitSize = m_gridData.particleInitMax - m_gridData.particleInitMin;
        float32_t4 simAreaSize = m_gridData.gridSize;
        simAreaSize *= m_gridData.gridCellSize;
        m_gridData.worldMin = float32_t4(0.f);
        m_gridData.worldMax = simAreaSize;
        numGridCells = m_gridData.gridSize.x * m_gridData.gridSize.y * m_gridData.gridSize.z;
        numParticles = m_gridData.particleInitSize.x * m_gridData.particleInitSize.y * m_gridData.particleInitSize.z * particlesPerCell;
    }

    template<core::StringLiteral ShaderKey>
    smart_refctd_ptr<IShader> loadPrecompiledShader()
    {
        IAssetLoader::SAssetLoadParams lparams = {};
        lparams.logger = m_logger.get();
        lparams.workingDirectory = "app_resources";
        auto key = nbl::this_example::builtin::build::get_spirv_key<ShaderKey>(m_device.get());
        auto bundle = m_assetMgr->getAsset(key.data(), lparams);
        if (bundle.getContents().empty() || bundle.getAssetType() != IAsset::ET_SHADER)
        {
            m_logger->log("Failed to find shader with key '%s'.", ILogger::ELL_ERROR, ShaderKey);
            exit(-1);
        }
        
        const auto assets = bundle.getContents();
        assert(assets.size() == 1);
        smart_refctd_ptr<IShader> shader = IAsset::castDown<IShader>(assets[0]);

        return shader;
    }

    // TODO: there's a method in IUtilities for this
    bool createBuffer(smart_refctd_ptr<IGPUBuffer>& buffer, video::IGPUBuffer::SCreationParams& params,
        core::bitflag<IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS> allocFlags = IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS::EMAF_NONE)
    {
        buffer = m_device->createBuffer(std::move(params));
        if (!buffer)
            return logFail("Failed to create GPU buffer of size %d!\n", params.size);

        video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = buffer->getMemoryReqs();
        reqs.memoryTypeBits &= m_physicalDevice->getDeviceLocalMemoryTypeBits();

        auto bufMem = m_device->allocate(reqs, buffer.get(), allocFlags);
        if (!bufMem.isValid())
            return logFail("Failed to allocate device memory compatible with gpu buffer!\n");

        return true;
    }

    bool createGridTexture(smart_refctd_ptr<IGPUImageView>& imageView, asset::E_FORMAT format, asset::VkExtent3D extent,
        core::bitflag<asset::IImage::E_USAGE_FLAGS> usage, const std::string& debugName = "",
        core::bitflag<asset::IImage::E_CREATE_FLAGS> flags = asset::IImage::E_CREATE_FLAGS::ECF_NONE)
    {
        IGPUImage::SCreationParams imgInfo;
        imgInfo.format = format;
        imgInfo.type = IGPUImage::ET_3D;
        imgInfo.extent = extent;
        imgInfo.mipLevels = 1u;
        imgInfo.arrayLayers = 1u;
        imgInfo.samples = asset::ICPUImage::ESCF_1_BIT;
        imgInfo.flags = flags;
        imgInfo.usage = usage;
        imgInfo.tiling = IGPUImage::TILING::OPTIMAL;

        auto image = m_device->createImage(std::move(imgInfo));
        auto imageMemReqs = image->getMemoryReqs();
        imageMemReqs.memoryTypeBits &= m_physicalDevice->getDeviceLocalMemoryTypeBits();
        m_device->allocate(imageMemReqs, image.get());

        if (!debugName.empty())
            image->setObjectDebugName(debugName.c_str());

        IGPUImageView::SCreationParams imgViewInfo;
        imgViewInfo.image = std::move(image);
        imgViewInfo.format = format;
        imgViewInfo.viewType = IGPUImageView::ET_3D;
        imgViewInfo.flags = IGPUImageView::E_CREATE_FLAGS::ECF_NONE;
        imgViewInfo.subresourceRange.aspectMask = asset::IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
        imgViewInfo.subresourceRange.baseArrayLayer = 0u;
        imgViewInfo.subresourceRange.baseMipLevel = 0u;
        imgViewInfo.subresourceRange.layerCount = 1u;
        imgViewInfo.subresourceRange.levelCount = 1u;

        imageView = m_device->createImageView(std::move(imgViewInfo));

        return true;
    }

    bool initGraphicsPipeline()
    {
        m_renderSemaphore = m_device->createSemaphore(m_realFrameIx);
        if (!m_renderSemaphore)
            return logFail("Failed to create render semaphore!\n");
            
        ISwapchain::SCreationParams swapchainParams{
            .surface = m_surface->getSurface()
        };
        if (!swapchainParams.deduceFormat(m_physicalDevice))
            return logFail("Could not choose a surface format for the swapchain!\n");

        const static IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] = {
            {
                .srcSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
                .dstSubpass = 0,
                .memoryBarrier = {
                    .srcStageMask = PIPELINE_STAGE_FLAGS::LATE_FRAGMENT_TESTS_BIT,
                    .srcAccessMask = ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                    .dstStageMask = PIPELINE_STAGE_FLAGS::EARLY_FRAGMENT_TESTS_BIT,
                    .dstAccessMask = ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_READ_BIT
                }
            },
            // color from ATTACHMENT_OPTIMAL to PRESENT_SRC
            {
                .srcSubpass = 0,
                .dstSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
                .memoryBarrier = {
                    .srcStageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
                    .srcAccessMask = ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
                }
            },
            IGPURenderpass::SCreationParams::DependenciesEnd
        };

        auto scResources = std::make_unique<CSwapchainFramebuffersAndDepth>(m_device.get(), EF_D16_UNORM, swapchainParams.surfaceFormat.format, dependencies);
        auto* renderpass = scResources->getRenderpass();
        if (!renderpass)
            return logFail("Failed to create renderpass!\n");

        auto queue = getGraphicsQueue();
        if (!m_surface || !m_surface->init(queue, std::move(scResources), swapchainParams.sharedParams))
            return logFail("Could not create window & surface or initialize surface\n");

        m_cmdPool = m_device->createCommandPool(queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
        for (auto i = 0u; i < MaxFramesInFlight; i++)
        {
            if (!m_cmdPool)
                return logFail("Couldn't create command pool\n");

            if (!m_cmdPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { m_cmdBufs.data() + i, 1 }))
                return logFail("Couldn't create command buffer\n");
        }

        m_winMgr->setWindowSize(m_window.get(), WIN_WIDTH, WIN_HEIGHT);
        m_surface->recreateSwapchain();

        // init shaders and pipeline

        auto loadPrecompiledShader = [&]<core::StringLiteral ShaderKey>() -> smart_refctd_ptr<IShader>
        {
            IAssetLoader::SAssetLoadParams lparams = {};
            lparams.logger = m_logger.get();
            lparams.workingDirectory = "app_resources";
            auto key = nbl::this_example::builtin::build::get_spirv_key<ShaderKey>(m_device.get());
            auto bundle = m_assetMgr->getAsset(key.data(), lparams);
            if (bundle.getContents().empty() || bundle.getAssetType() != IAsset::ET_SHADER)
            {
                m_logger->log("Failed to find shader with key '%s'.", ILogger::ELL_ERROR, ShaderKey);
                exit(-1);
            }
        
            const auto assets = bundle.getContents();
            assert(assets.size() == 1);
            smart_refctd_ptr<IShader> shader = IAsset::castDown<IShader>(assets[0]);

            return shader;
        };
        auto vs = loadPrecompiledShader.operator()<"fluidParticles_vertex">(); // "app_resources/fluidParticles.vertex.hlsl"
        auto fs = loadPrecompiledShader.operator()<"fluidParticles_fragment">(); // "app_resources/fluidParticles.fragment.hlsl"

        smart_refctd_ptr<video::IGPUDescriptorSetLayout> descriptorSetLayout1;
        {
            // init descriptors
            video::IGPUDescriptorSetLayout::SBinding bindingsSet1[] = {
                {
                    .binding = 0u,
                    .type = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
                    .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
                    .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_FRAGMENT,
                    .count = 1u,
                }
            };
            descriptorSetLayout1 = m_device->createDescriptorSetLayout(bindingsSet1);
            if (!descriptorSetLayout1)
                return logFail("Failed to Create Render Descriptor Layout 1");

            const auto maxDescriptorSets = ICPUPipelineLayout::DESCRIPTOR_SET_COUNT;
            const std::array<IGPUDescriptorSetLayout*, maxDescriptorSets> dscLayoutPtrs = {
                nullptr,
                descriptorSetLayout1.get(),
                nullptr,
                nullptr
            };
            m_renderDsPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_UPDATE_AFTER_BIND_BIT, std::span(dscLayoutPtrs.begin(), dscLayoutPtrs.end()));
            m_renderDs = m_renderDsPool->createDescriptorSet(descriptorSetLayout1);
        }

        // write descriptors
        {
            IGPUDescriptorSet::SDescriptorInfo camInfo;
            camInfo.desc = smart_refctd_ptr(cameraBuffer);
            camInfo.info.buffer = {.offset = 0, .size = cameraBuffer->getSize()};
            IGPUDescriptorSet::SWriteDescriptorSet writes[1] = {
                {.dstSet = m_renderDs.get(), .binding = 0, .arrayElement = 0, .count = 1, .info = &camInfo}
            };
            m_device->updateDescriptorSets(std::span(writes, 1), {});
        }

        SBlendParams blendParams = {};
        blendParams.logicOp = ELO_NO_OP;
        blendParams.blendParams[0u].colorWriteMask = (1u << 0u) | (1u << 1u) | (1u << 2u) | (1u << 3u);

        {
            const asset::SPushConstantRange pcRange = { .stageFlags = IShader::E_SHADER_STAGE::ESS_VERTEX, .offset = 0, .size = sizeof(uint64_t) };
            const auto pipelineLayout = m_device->createPipelineLayout({ &pcRange , 1 }, nullptr, smart_refctd_ptr(descriptorSetLayout1), nullptr, nullptr);

            SRasterizationParams rasterizationParams{};
            rasterizationParams.faceCullingMode = EFCM_NONE;
            rasterizationParams.depthWriteEnable = true;

            IGPUGraphicsPipeline::SCreationParams params[1] = {};
            params[0].layout = pipelineLayout.get();
            params[0].vertexShader = { .shader = vs.get(), .entryPoint = "main", };
            params[0].fragmentShader = { .shader = fs.get(), .entryPoint = "main", };
            params[0].cached = {
                .vertexInput = {
                },
                .primitiveAssembly = {
                    .primitiveType = E_PRIMITIVE_TOPOLOGY::EPT_TRIANGLE_LIST,
                },
                .rasterization = rasterizationParams,
                .blend = blendParams,
            };
            params[0].renderpass = renderpass;

            if (!m_device->createGraphicsPipelines(nullptr, params, &m_graphicsPipeline))
                return logFail("Graphics pipeline creation failed");
        }

        return true;
    }

    void mouseProcess(const nbl::ui::IMouseEventChannel::range_t& events)
    {
        for (auto eventIt = events.begin(); eventIt != events.end(); eventIt++)
        {
            auto ev = *eventIt;

            // do nothing
        }
    }


    // in-loop functions
    void initializeParticles(IGPUCommandBuffer* cmdbuf)
    {
        {
            SMemoryBarrier memBarrier;
            memBarrier.srcStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS;
            memBarrier.srcAccessMask = ACCESS_FLAGS::MEMORY_READ_BITS | ACCESS_FLAGS::MEMORY_WRITE_BITS;;
            memBarrier.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
            memBarrier.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;
            cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, {.memBarriers = {&memBarrier, 1}});
        }

        const uint64_t bufferAddr[2] = { 
            particleData.positionBuffer->getDeviceAddress(),
            particleData.velocityBuffer->getDeviceAddress()
        };
        
        cmdbuf->bindComputePipeline(m_initParticlePipeline.get());
        cmdbuf->pushConstants(m_initParticlePipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0, 2 * sizeof(uint64_t), bufferAddr);
        cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_initParticlePipeline->getLayout(), 1, 1, &m_initParticleDs.get());
        cmdbuf->dispatch(WorkgroupCountParticles, 1, 1);

        m_shouldInitParticles = false;
    }

    void transitionGridImageLayouts(IGPUCommandBuffer* cmdbuf)
    {
        // transition layouts, only after after initialization
        auto fillGridBarrierInfo = [&](IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t& barrier,
            smart_refctd_ptr<IGPUImageView>& imageView) -> void
            {
                barrier.barrier = {
                    .dep = {
                        .srcStageMask = PIPELINE_STAGE_FLAGS::NONE,
                        .srcAccessMask = ACCESS_FLAGS::NONE,
                        .dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
                        .dstAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS
                }
                };
                barrier.image = imageView->getCreationParameters().image.get();
                barrier.subresourceRange = {
                    .aspectMask = IImage::EAF_COLOR_BIT,
                    .baseMipLevel = 0u,
                    .levelCount = 1u,
                    .baseArrayLayer = 0u,
                    .layerCount = 1u
                };
                barrier.oldLayout = IImage::LAYOUT::UNDEFINED;
                barrier.newLayout = IImage::LAYOUT::GENERAL;
            };

        uint32_t count = 0;
        IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t imageBarriers[14];
        for (uint32_t i = 0; i < 3; i++)
        {
            fillGridBarrierInfo(imageBarriers[count++], velocityFieldImageViews[i]);
            fillGridBarrierInfo(imageBarriers[count++], prevVelocityFieldImageViews[i]);
        }
        fillGridBarrierInfo(imageBarriers[count++], gridParticleCountImageView);

        fillGridBarrierInfo(imageBarriers[count++], gridCellMaterialImageView);
        fillGridBarrierInfo(imageBarriers[count++], tempCellMaterialImageView);

        fillGridBarrierInfo(imageBarriers[count++], gridAxisCellMaterialImageView);
        fillGridBarrierInfo(imageBarriers[count++], tempAxisCellMaterialImageView);

        fillGridBarrierInfo(imageBarriers[count++], gridDiffusionImageView);

        fillGridBarrierInfo(imageBarriers[count++], pressureImageView);
        fillGridBarrierInfo(imageBarriers[count++], divergenceImageView);

        cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = imageBarriers });
    }


    smart_refctd_ptr<IWindow> m_window;
    smart_refctd_ptr<CSimpleResizeSurface<CSwapchainFramebuffersAndDepth>> m_surface;
    smart_refctd_ptr<IGPUGraphicsPipeline> m_graphicsPipeline;
    smart_refctd_ptr<ISemaphore> m_renderSemaphore;
    smart_refctd_ptr<IGPUCommandPool> m_cmdPool;
    std::array<smart_refctd_ptr<IGPUCommandBuffer>, MaxFramesInFlight> m_cmdBufs;
    uint64_t m_realFrameIx : 59 = 0;
    ISimpleManagedSurface::SAcquireResult m_currentImageAcquire = {};

    smart_refctd_ptr<video::IDescriptorPool> m_renderDsPool;
    smart_refctd_ptr<IGPUDescriptorSet> m_renderDs;

    // simulation compute shaders
    // TODO: unsure many of the axis-cell material pipelines need to exist
    smart_refctd_ptr<IGPUComputePipeline> m_initParticlePipeline;

    smart_refctd_ptr<IGPUComputePipeline> m_accumulateWeightsPipeline;
    smart_refctd_ptr<IGPUComputePipeline> m_updateFluidCellsPipeline;
    smart_refctd_ptr<IGPUComputePipeline> m_updateNeighborCellsPipeline;

    smart_refctd_ptr<IGPUComputePipeline> m_applyBodyForcesPipeline;
    
    smart_refctd_ptr<IGPUComputePipeline> m_axisCellsPipeline;
    smart_refctd_ptr<IGPUComputePipeline> m_neighborAxisCellsPipeline;
    smart_refctd_ptr<IGPUComputePipeline> m_iterateDiffusionPipeline;
    smart_refctd_ptr<IGPUComputePipeline> m_diffusionPipeline;

    smart_refctd_ptr<IGPUComputePipeline> m_calcDivergencePipeline;
    smart_refctd_ptr<IGPUComputePipeline> m_iteratePressurePipeline;
    smart_refctd_ptr<IGPUComputePipeline> m_updateVelPsPipeline;

    smart_refctd_ptr<IGPUComputePipeline> m_advectParticlesPipeline;
    smart_refctd_ptr<IGPUComputePipeline> m_genParticleVerticesPipeline;

    // descriptors
    // TODO: why does every single descriptor set have its own pool!?
    smart_refctd_ptr<video::IDescriptorPool> m_initParticlePool;
    smart_refctd_ptr<IGPUDescriptorSet> m_initParticleDs;

    smart_refctd_ptr<video::IDescriptorPool> m_accumulateWeightsPool;
    smart_refctd_ptr<IGPUDescriptorSet> m_accumulateWeightsDs;
    smart_refctd_ptr<video::IDescriptorPool> m_updateFluidCellsPool;
    smart_refctd_ptr<IGPUDescriptorSet> m_updateFluidCellsDs;
    smart_refctd_ptr<video::IDescriptorPool> m_updateNeighborCellsPool;
    smart_refctd_ptr<IGPUDescriptorSet> m_updateNeighborCellsDs;

    smart_refctd_ptr<video::IDescriptorPool> m_applyForcesPool;
    smart_refctd_ptr<IGPUDescriptorSet> m_applyForcesDs;

    smart_refctd_ptr<video::IDescriptorPool> m_axisCellsPool;
    smart_refctd_ptr<IGPUDescriptorSet> m_axisCellsDs;
    smart_refctd_ptr<video::IDescriptorPool> m_neighborAxisCellsPool;
    smart_refctd_ptr<IGPUDescriptorSet> m_neighborAxisCellsDs;
    smart_refctd_ptr<video::IDescriptorPool> m_diffusionPool;
    smart_refctd_ptr<IGPUDescriptorSet> m_diffusionDs;

    smart_refctd_ptr<video::IDescriptorPool> m_calcDivergencePool;
    smart_refctd_ptr<IGPUDescriptorSet> m_calcDivergenceDs;
    smart_refctd_ptr<video::IDescriptorPool> m_iteratePressurePool;
    smart_refctd_ptr<IGPUDescriptorSet> m_iteratePressureDs;
    smart_refctd_ptr<video::IDescriptorPool> m_updateVelPsPool;
    smart_refctd_ptr<IGPUDescriptorSet> m_updateVelPsDs;
    
    smart_refctd_ptr<video::IDescriptorPool> m_advectParticlesPool;
    smart_refctd_ptr<IGPUDescriptorSet> m_advectParticlesDs;
    smart_refctd_ptr<video::IDescriptorPool> m_genVerticesPool;
    smart_refctd_ptr<IGPUDescriptorSet> m_genVerticesDs;

    // input system
    smart_refctd_ptr<InputSystem> m_inputSystem;
    InputSystem::ChannelReader<IMouseEventChannel> mouse;
    InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

    Camera camera = Camera(core::vectorSIMDf(0,0,0), core::vectorSIMDf(0,0,0), hlsl::float32_t4x4());
    video::CDumbPresentationOracle oracle;

    bool m_shouldInitParticles = true;

    // simulation constants
    size_t WorkgroupCountParticles;
    uint32_t3 WorkgroupCountGrid;
    uint32_t m_substepsPerFrame = 1;
    SGridData m_gridData;
    SParticleRenderParams m_pRenderParams;
    uint32_t particlesPerCell = 8;
    uint32_t numParticles;
    uint32_t numGridCells;

    const float viscosity = 0.f;
    const uint32_t diffusionIterations = 5;
    const uint32_t pressureSolverIterations = 5;

    // buffers
    smart_refctd_ptr<IGPUBuffer> cameraBuffer;

    struct ParticleData
    {
        smart_refctd_ptr<IGPUBuffer> positionBuffer;
        smart_refctd_ptr<IGPUBuffer> velocityBuffer;
    };
    ParticleData particleData;

    smart_refctd_ptr<IGPUBuffer> pParamsBuffer;			            // SParticleRenderParams
    // TODO: remove!
    smart_refctd_ptr<IGPUBuffer> particleVertexBuffer;	            // VertexInfo * 6 vertices

    smart_refctd_ptr<IGPUBuffer> gridDataBuffer;		            // SGridData
    smart_refctd_ptr<IGPUBuffer> pressureParamsBuffer;	            // SPressureSolverParams
    smart_refctd_ptr<IGPUImageView> gridParticleCountImageView;	    // uint
    smart_refctd_ptr<IGPUImageView> gridCellMaterialImageView;	    // uint, fluid or solid

    std::array<smart_refctd_ptr<IGPUImageView>, 3> velocityFieldImageViews;		// float * 3 (per axis)
    std::array<smart_refctd_ptr<IGPUImageView>, 3> prevVelocityFieldImageViews;	// float * 3
    smart_refctd_ptr<IGPUSampler> velocityFieldSampler;

    std::array<smart_refctd_ptr<IGPUImageView>, 3> velocityFieldUintViews;
    std::array<smart_refctd_ptr<IGPUImageView>, 3> prevVelocityFieldUintViews;

    smart_refctd_ptr<IGPUImageView> gridDiffusionImageView;	        // float4
    smart_refctd_ptr<IGPUImageView> gridAxisCellMaterialImageView;	// uint4
    smart_refctd_ptr<IGPUImageView> divergenceImageView;		    // float
    smart_refctd_ptr<IGPUImageView> pressureImageView;		        // float

    smart_refctd_ptr<IGPUImageView> tempCellMaterialImageView;	    // uint, fluid or solid
    smart_refctd_ptr<IGPUImageView> tempAxisCellMaterialImageView;	// uint4
};

NBL_MAIN_FUNC(FLIPFluidsApp)