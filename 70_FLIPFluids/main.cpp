#include <nabla.h>

#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"
#include "SimpleWindowedApplication.hpp"
#include "InputSystem.hpp"
#include "CCamera.hpp"

//#include "gpuRadixSort.h"

#include "glm/glm/glm.hpp"
#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>

#include "app_resources/common.hlsl"
#include "app_resources/descriptor_bindings.hlsl"

using namespace nbl::hlsl;
using namespace nbl;
using namespace core;
using namespace hlsl;
using namespace system;
using namespace asset;
using namespace ui;
using namespace video;


enum SimPresets
{
    CENTER_DROP,
    LONG_BOX
};

struct Particle
{
    float32_t4 position;
    float32_t4 velocity;

    uint32_t id;
    uint32_t pad[3];
};

struct SGridData
{
    float gridCellSize;
    float gridInvCellSize;
    float pad0[2];

    int32_t4 particleInitMin;
    int32_t4 particleInitMax;
    int32_t4 particleInitSize;

    float32_t4 worldMin;
    float32_t4 worldMax;
    int32_t4 gridSize;
};

struct SMVPParams
{
    float cameraPosition[4];

    float MVP[4*4];
    float M[4*4];
    float V[4*4];
    float P[4*4];
};

struct SParticleRenderParams
{
    float radius;
    float zNear;
    float zFar;
    float pad;
};

struct VertexInfo
{
    float32_t4 position;
    float32_t4 vsSpherePos;

    float radius;
    float pad;

    float32_t4 color;
    float32_t2 uv;
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

class FLIPFluidsApp final : public examples::SimpleWindowedApplication, public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
    using device_base_t = examples::SimpleWindowedApplication;
    using asset_base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;
    using clock_t = std::chrono::steady_clock;

    _NBL_STATIC_INLINE_CONSTEXPR uint32_t WIN_WIDTH = 1280, WIN_HEIGHT = 720, SC_IMG_COUNT = 3u, FRAMES_IN_FLIGHT = 5u;
    static_assert(FRAMES_IN_FLIGHT > SC_IMG_COUNT);

    constexpr static inline clock_t::duration DisplayImageDuration = std::chrono::milliseconds(900);

    using IGPUDescriptorSetLayoutArray = std::array<core::smart_refctd_ptr<IGPUDescriptorSetLayout>, ICPUPipelineLayout::DESCRIPTOR_SET_COUNT>;

public:
    inline FLIPFluidsApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
        : IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

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
            matrix4SIMD projectionMatrix = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(60.0f), float(WIN_WIDTH) / WIN_HEIGHT, zNear, zFar);
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

        params.size = numParticles * sizeof(Particle);
        params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_SRC_BIT;
        createBuffer(particleBuffer, params);
        params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT;
        createBuffer(tempParticleBuffer, params);

        params.size = numParticles * 6 * sizeof(VertexInfo);
        createBuffer(particleVertexBuffer, params);

        asset::VkExtent3D gridExtent = { m_gridData.gridSize.x, m_gridData.gridSize.y, m_gridData.gridSize.z };

        // cell materials
        createGridTexture(gridCellMaterialImageView, asset::EF_R32_UINT, gridExtent, asset::IImage::EUF_STORAGE_BIT, "cell material0");
        createGridTexture(tempCellMaterialImageView, asset::EF_R32_UINT, gridExtent, asset::IImage::EUF_STORAGE_BIT, "cell material1");

        createGridTexture(gridAxisCellMaterialImageView, asset::EF_R32G32B32A32_UINT, gridExtent, asset::IImage::EUF_STORAGE_BIT, "axis cell material0");
        createGridTexture(tempAxisCellMaterialImageView, asset::EF_R32G32B32A32_UINT, gridExtent, asset::IImage::EUF_STORAGE_BIT, "axis cell material1");

        createGridTexture(gridParticleCountImageView, asset::EF_R32_UINT, gridExtent,
            asset::IImage::EUF_STORAGE_BIT | asset::IImage::EUF_TRANSFER_DST_BIT, "particle counts");

        // diffusion grids
        createGridTexture(gridDiffusionImageView, asset::EF_R32G32B32A32_SFLOAT, gridExtent, asset::IImage::EUF_STORAGE_BIT, "diffusion0");
        createGridTexture(tempDiffusionImageView, asset::EF_R32G32B32A32_SFLOAT, gridExtent, asset::IImage::EUF_STORAGE_BIT, "diffusion1");

        // pressure grids
        createGridTexture(pressureImageView, asset::EF_R32_SFLOAT, gridExtent, asset::IImage::EUF_STORAGE_BIT, "pressure0");
        createGridTexture(tempPressureImageView, asset::EF_R32_SFLOAT, gridExtent, asset::IImage::EUF_STORAGE_BIT, "pressure1");
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
        samplerParams.TextureWrapU = IGPUSampler::ETC_CLAMP_TO_BORDER;
        samplerParams.TextureWrapV = IGPUSampler::ETC_CLAMP_TO_BORDER;
        samplerParams.TextureWrapW = IGPUSampler::ETC_CLAMP_TO_BORDER;
        samplerParams.BorderColor  = IGPUSampler::ETBC_FLOAT_TRANSPARENT_BLACK;
        samplerParams.MinFilter		= IGPUSampler::ETF_LINEAR;
        samplerParams.MaxFilter		= IGPUSampler::ETF_LINEAR;
        samplerParams.MipmapMode	= IGPUSampler::ESMM_LINEAR;
        samplerParams.AnisotropicFilter = 3;
        samplerParams.CompareEnable = false;
        velocityFieldSampler = m_device->createSampler(samplerParams);

        // init render pipeline
        if (!initGraphicsPipeline())
            return logFail("Failed to initialize render pipeline!\n");

        auto createComputePipeline = [&](smart_refctd_ptr<IGPUComputePipeline>& pipeline, smart_refctd_ptr<IDescriptorPool>& pool,
            smart_refctd_ptr<IGPUDescriptorSet>& set, const std::string& shaderPath, const std::string& entryPoint,
            const std::span<const IGPUDescriptorSetLayout::SBinding> bindings, const asset::SPushConstantRange& pcRange = {}) -> void
            {
                auto shader = compileShader(shaderPath, entryPoint);

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
                
                m_device->createComputePipelines(nullptr, { &params,1 }, &pipeline);
            };

        {
            // init particles pipeline
            createComputePipeline(m_initParticlePipeline, m_initParticlePool, m_initParticleDs,
                "app_resources/compute/particlesInit.comp.hlsl", "main", piParticlesInit_bs1);

            {
                IGPUDescriptorSet::SDescriptorInfo infos[2];
                infos[0].desc = smart_refctd_ptr(gridDataBuffer);
                infos[0].info.buffer = {.offset = 0, .size = gridDataBuffer->getSize()};
                infos[1].desc = smart_refctd_ptr(particleBuffer);
                infos[1].info.buffer = {.offset = 0, .size = particleBuffer->getSize()};
                IGPUDescriptorSet::SWriteDescriptorSet writes[2] = {
                    {.dstSet = m_initParticleDs.get(), .binding = b_piGridData, .arrayElement = 0, .count = 1, .info = &infos[0]},
                    {.dstSet = m_initParticleDs.get(), .binding = b_piPBuffer, .arrayElement = 0, .count = 1, .info = &infos[1]},
                };
                m_device->updateDescriptorSets(std::span(writes, 2), {});
            }
        }
        {
            // generate particle vertex pipeline
            createComputePipeline(m_genParticleVerticesPipeline, m_genVerticesPool, m_genVerticesDs,
                "app_resources/compute/genParticleVertices.comp.hlsl", "main", gpvGenVertices_bs1);

            {
                IGPUDescriptorSet::SDescriptorInfo infos[4];
                infos[0].desc = smart_refctd_ptr(cameraBuffer);
                infos[0].info.buffer = {.offset = 0, .size = cameraBuffer->getSize()};
                infos[1].desc = smart_refctd_ptr(pParamsBuffer);
                infos[1].info.buffer = {.offset = 0, .size = pParamsBuffer->getSize()};
                infos[2].desc = smart_refctd_ptr(particleBuffer);
                infos[2].info.buffer = {.offset = 0, .size = particleBuffer->getSize()};
                infos[3].desc = smart_refctd_ptr(particleVertexBuffer);
                infos[3].info.buffer = {.offset = 0, .size = particleVertexBuffer->getSize()};
                IGPUDescriptorSet::SWriteDescriptorSet writes[4] = {
                    {.dstSet = m_genVerticesDs.get(), .binding = b_gpvCamData, .arrayElement = 0, .count = 1, .info = &infos[0]},
                    {.dstSet = m_genVerticesDs.get(), .binding = b_gpvPParams, .arrayElement = 0, .count = 1, .info = &infos[1]},
                    {.dstSet = m_genVerticesDs.get(), .binding = b_gpvPBuffer, .arrayElement = 0, .count = 1, .info = &infos[2]},
                    {.dstSet = m_genVerticesDs.get(), .binding = b_gpvPVertBuffer, .arrayElement = 0, .count = 1, .info = &infos[3]}
                };
                m_device->updateDescriptorSets(std::span(writes, 4), {});
            }
        }
        // update fluid cells pipelines
        {
            createComputePipeline(m_accumulateWeightsPipeline, m_accumulateWeightsPool, m_accumulateWeightsDs,
                "app_resources/compute/prepareCellUpdate.comp.hlsl", "main", ufcAccWeights_bs1);

            {
                IGPUDescriptorSet::SDescriptorInfo infos[3];
                infos[0].desc = smart_refctd_ptr(gridDataBuffer);
                infos[0].info.buffer = { .offset = 0, .size = gridDataBuffer->getSize() };
                infos[1].desc = smart_refctd_ptr(particleBuffer);
                infos[1].info.buffer = { .offset = 0, .size = particleBuffer->getSize() };
                infos[2].desc = gridParticleCountImageView;
                infos[2].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                infos[2].info.combinedImageSampler.sampler = nullptr;

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

                IGPUDescriptorSet::SWriteDescriptorSet writes[5] = {
                    {.dstSet = m_accumulateWeightsDs.get(), .binding = b_ufcGridData, .arrayElement = 0, .count = 1, .info = &infos[0]},
                    {.dstSet = m_accumulateWeightsDs.get(), .binding = b_ufcPBuffer, .arrayElement = 0, .count = 1, .info = &infos[1]},
                    {.dstSet = m_accumulateWeightsDs.get(), .binding = b_ufcGridPCountBuffer, .arrayElement = 0, .count = 1, .info = &infos[2]},
                    {.dstSet = m_accumulateWeightsDs.get(), .binding = b_ufcVelBuffer, .arrayElement = 0, .count = 3, .info = imgInfosVel0},
                    {.dstSet = m_accumulateWeightsDs.get(), .binding = b_ufcPrevVelBuffer, .arrayElement = 0, .count = 3, .info = imgInfosVel1},
                };
                m_device->updateDescriptorSets(std::span(writes, 5), {});
            }
        }
        {
            createComputePipeline(m_updateFluidCellsPipeline, m_updateFluidCellsPool, m_updateFluidCellsDs,
                "app_resources/compute/updateFluidCells.comp.hlsl", "updateFluidCells", ufcFluidCell_bs1);

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
                    {.dstSet = m_updateFluidCellsDs.get(), .binding = b_ufcGridPCountBuffer, .arrayElement = 0, .count = 1, .info = &infos[1]},
                    {.dstSet = m_updateFluidCellsDs.get(), .binding = b_ufcCMOutBuffer, .arrayElement = 0, .count = 1, .info = &infos[2]},
                };
                m_device->updateDescriptorSets(std::span(writes, 3), {});
            }
        }
        {
            createComputePipeline(m_updateNeighborCellsPipeline, m_updateNeighborCellsPool, m_updateNeighborCellsDs,
                "app_resources/compute/updateFluidCells.comp.hlsl", "updateNeighborFluidCells", ufcNeighborCell_bs1);

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
                    {.dstSet = m_updateNeighborCellsDs.get(), .binding = b_ufcCMInBuffer, .arrayElement = 0, .count = 1, .info = &infos[1]},
                    {.dstSet = m_updateNeighborCellsDs.get(), .binding = b_ufcCMOutBuffer, .arrayElement = 0, .count = 1, .info = &infos[2]},
                    {.dstSet = m_updateNeighborCellsDs.get(), .binding = b_ufcVelBuffer, .arrayElement = 0, .count = 3, .info = imgInfosVel0},
                    {.dstSet = m_updateNeighborCellsDs.get(), .binding = b_ufcPrevVelBuffer, .arrayElement = 0, .count = 3, .info = imgInfosVel1},
                };
                m_device->updateDescriptorSets(std::span(writes, 5), {});
            }
        }
        {
            // apply forces pipeline
            createComputePipeline(m_applyBodyForcesPipeline, m_applyForcesPool, m_applyForcesDs, 
                "app_resources/compute/applyBodyForces.comp.hlsl", "main", abfApplyForces_bs1);

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
                    {.dstSet = m_applyForcesDs.get(), .binding = b_abfVelFieldBuffer, .arrayElement = 0, .count = 3, .info = imgInfosVel0},
                    {.dstSet = m_applyForcesDs.get(), .binding = b_abfCMBuffer, .arrayElement = 0, .count = 1, .info = &infos[1]},
                };
                m_device->updateDescriptorSets(std::span(writes, 3), {});
            }
        }
        // apply diffusion pipelines
        {
            createComputePipeline(m_axisCellsPipeline, m_axisCellsPool, m_axisCellsDs, 
                "app_resources/compute/diffusion.comp.hlsl", "setAxisCellMaterial", dAxisCM_bs1);

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
                    {.dstSet = m_axisCellsDs.get(), .binding = b_dCMBuffer, .arrayElement = 0, .count = 1, .info = &infos[1]},
                    {.dstSet = m_axisCellsDs.get(), .binding = b_dAxisOutBuffer, .arrayElement = 0, .count = 1, .info = &infos[2]},
                };
                m_device->updateDescriptorSets(std::span(writes, 3), {});
            }
        }
        {
            createComputePipeline(m_neighborAxisCellsPipeline, m_neighborAxisCellsPool, m_neighborAxisCellsDs, 
                "app_resources/compute/diffusion.comp.hlsl", "setNeighborAxisCellMaterial", dNeighborAxisCM_bs1);

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
                    {.dstSet = m_neighborAxisCellsDs.get(), .binding = b_dAxisInBuffer, .arrayElement = 0, .count = 1, .info = &infos[1]},
                    {.dstSet = m_neighborAxisCellsDs.get(), .binding = b_dAxisOutBuffer, .arrayElement = 0, .count = 1, .info = &infos[2]},
                };
                m_device->updateDescriptorSets(std::span(writes, 3), {});
            }
        }
        {
            const std::string entryPoint = "applyDiffusion";
            auto shader = compileShader("app_resources/compute/diffusion.comp.hlsl", entryPoint);

            auto descriptorSetLayout1 = m_device->createDescriptorSetLayout(dDiffuse_bs1);

            const std::array<IGPUDescriptorSetLayout*, ICPUPipelineLayout::DESCRIPTOR_SET_COUNT> dscLayoutPtrs = {
                nullptr,
                descriptorSetLayout1.get()
            };
            const uint32_t setCounts[2u] = { 0u, 2u };
            m_diffusionPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_UPDATE_AFTER_BIND_BIT, std::span(dscLayoutPtrs.begin(), dscLayoutPtrs.end()), setCounts);
            m_diffusionDs[0] = m_diffusionPool->createDescriptorSet(descriptorSetLayout1);
            m_diffusionDs[1] = m_diffusionPool->createDescriptorSet(descriptorSetLayout1);

            const asset::SPushConstantRange pcRange = { .stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE, .offset = 0, .size = 4 * sizeof(uint32_t) };

            smart_refctd_ptr<nbl::video::IGPUPipelineLayout> pipelineLayout = m_device->createPipelineLayout({ &pcRange, 1 }, nullptr, smart_refctd_ptr(descriptorSetLayout1), nullptr, nullptr);

            IGPUComputePipeline::SCreationParams params = {};
            params.layout = pipelineLayout.get();
            params.shader.entryPoint = entryPoint;
            params.shader.shader = shader.get();
                
            m_device->createComputePipelines(nullptr, { &params,1 }, &m_diffusionPipeline);

            {
                IGPUDescriptorSet::SDescriptorInfo infos[5];
                infos[0].desc = smart_refctd_ptr(gridDataBuffer);
                infos[0].info.buffer = {.offset = 0, .size = gridDataBuffer->getSize()};				
                infos[1].desc = gridCellMaterialImageView;
                infos[1].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                infos[1].info.combinedImageSampler.sampler = nullptr;
                infos[2].desc = gridAxisCellMaterialImageView;
                infos[2].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                infos[2].info.combinedImageSampler.sampler = nullptr;
                infos[3].desc = tempDiffusionImageView;
                infos[3].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                infos[3].info.combinedImageSampler.sampler = nullptr;
                infos[4].desc = gridDiffusionImageView;
                infos[4].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                infos[4].info.combinedImageSampler.sampler = nullptr;

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

                IGPUDescriptorSet::SWriteDescriptorSet writes[6] = {
                    {.dstSet = m_diffusionDs[0].get(), .binding = b_dGridData, .arrayElement = 0, .count = 1, .info = &infos[0]},
                    {.dstSet = m_diffusionDs[0].get(), .binding = b_dCMBuffer, .arrayElement = 0, .count = 1, .info = &infos[1]},
                    {.dstSet = m_diffusionDs[0].get(), .binding = b_dVelBuffer, .arrayElement = 0, .count = 3, .info = imgInfosVel0},
                    {.dstSet = m_diffusionDs[0].get(), .binding = b_dAxisInBuffer, .arrayElement = 0, .count = 1, .info = &infos[2]},
                    {.dstSet = m_diffusionDs[0].get(), .binding = b_dDiffInBuffer, .arrayElement = 0, .count = 1, .info = &infos[3]},
                    {.dstSet = m_diffusionDs[0].get(), .binding = b_dDiffOutBuffer, .arrayElement = 0, .count = 1, .info = &infos[4]},
                };
                m_device->updateDescriptorSets(std::span(writes, 6), {});
            }
            {
                IGPUDescriptorSet::SDescriptorInfo infos[5];
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
                infos[4].desc = tempDiffusionImageView;
                infos[4].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                infos[4].info.combinedImageSampler.sampler = nullptr;

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

                IGPUDescriptorSet::SWriteDescriptorSet writes[6] = {
                    {.dstSet = m_diffusionDs[1].get(), .binding = b_dGridData, .arrayElement = 0, .count = 1, .info = &infos[0]},
                    {.dstSet = m_diffusionDs[1].get(), .binding = b_dCMBuffer, .arrayElement = 0, .count = 1, .info = &infos[1]},
                    {.dstSet = m_diffusionDs[1].get(), .binding = b_dVelBuffer, .arrayElement = 0, .count = 3, .info = imgInfosVel0},
                    {.dstSet = m_diffusionDs[1].get(), .binding = b_dAxisInBuffer, .arrayElement = 0, .count = 1, .info = &infos[2]},
                    {.dstSet = m_diffusionDs[1].get(), .binding = b_dDiffInBuffer, .arrayElement = 0, .count = 1, .info = &infos[3]},
                    {.dstSet = m_diffusionDs[1].get(), .binding = b_dDiffOutBuffer, .arrayElement = 0, .count = 1, .info = &infos[4]},
                };
                m_device->updateDescriptorSets(std::span(writes, 6), {});
            }
        }
        {
            createComputePipeline(m_updateVelDPipeline, m_updateVelDPool, m_updateVelDDs, 
                "app_resources/compute/diffusion.comp.hlsl", "updateVelocity", dUpdateVelD_bs1);

            {
                IGPUDescriptorSet::SDescriptorInfo infos[3];
                infos[0].desc = smart_refctd_ptr(gridDataBuffer);
                infos[0].info.buffer = {.offset = 0, .size = gridDataBuffer->getSize()};				
                infos[1].desc = gridCellMaterialImageView;
                infos[1].info.image.imageLayout = IImage::LAYOUT::GENERAL;
                infos[1].info.combinedImageSampler.sampler = nullptr;
                infos[2].desc = gridDiffusionImageView;
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
                    {.dstSet = m_updateVelDDs.get(), .binding = b_dGridData, .arrayElement = 0, .count = 1, .info = &infos[0]},
                    {.dstSet = m_updateVelDDs.get(), .binding = b_dCMBuffer, .arrayElement = 0, .count = 1, .info = &infos[1]},
                    {.dstSet = m_updateVelDDs.get(), .binding = b_dVelBuffer, .arrayElement = 0, .count = 3, .info = imgInfosVel0},
                    {.dstSet = m_updateVelDDs.get(), .binding = b_dDiffInBuffer, .arrayElement = 0, .count = 1, .info = &infos[2]},
                };
                m_device->updateDescriptorSets(std::span(writes, 4), {});
            }
        }
        // solve pressure system pipelines
        {
            createComputePipeline(m_calcDivergencePipeline, m_calcDivergencePool, m_calcDivergenceDs, 
                "app_resources/compute/pressureSolver.comp.hlsl", "calculateNegativeDivergence", psDivergence_bs1);

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
                    {.dstSet = m_calcDivergenceDs.get(), .binding = b_psCMBuffer, .arrayElement = 0, .count = 1, .info = &infos[1]},
                    {.dstSet = m_calcDivergenceDs.get(), .binding = b_psVelBuffer, .arrayElement = 0, .count = 3, .info = imgInfosVel0},
                    {.dstSet = m_calcDivergenceDs.get(), .binding = b_psDivBuffer, .arrayElement = 0, .count = 1, .info = &infos[2]},
                };
                m_device->updateDescriptorSets(std::span(writes, 4), {});
            }
        }
        {
            createComputePipeline(m_iteratePressurePipeline, m_iteratePressurePool, m_iteratePressureDs,
                "app_resources/compute/pressureSolver.comp.hlsl", "iteratePressureSystem", psIteratePressure_bs1);

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
                    {.dstSet = m_iteratePressureDs.get(), .binding = b_psCMBuffer, .arrayElement = 0, .count = 1, .info = &infos[2]},
                    {.dstSet = m_iteratePressureDs.get(), .binding = b_psDivBuffer, .arrayElement = 0, .count = 1, .info = &infos[3]},
                    {.dstSet = m_iteratePressureDs.get(), .binding = b_psPresInBuffer, .arrayElement = 0, .count = 1, .info = &infos[4]},
                };
                m_device->updateDescriptorSets(std::span(writes, 5), {});
            }
        }
        //{
        //    const std::string entryPoint = "solvePressureSystem";
        //    auto shader = compileShader("app_resources/compute/pressureSolver.comp.hlsl", entryPoint);

        //    auto descriptorSetLayout1 = m_device->createDescriptorSetLayout(psSolvePressure_bs1);

        //    const std::array<IGPUDescriptorSetLayout*, ICPUPipelineLayout::DESCRIPTOR_SET_COUNT> dscLayoutPtrs = {
        //        nullptr,
        //        descriptorSetLayout1.get()
        //    };
        //    const uint32_t setCounts[2u] = { 0u, 2u };
        //    m_solvePressurePool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_UPDATE_AFTER_BIND_BIT, std::span(dscLayoutPtrs.begin(), dscLayoutPtrs.end()), setCounts);
        //    m_solvePressureDs[0] = m_solvePressurePool->createDescriptorSet(descriptorSetLayout1);
        //    m_solvePressureDs[1] = m_solvePressurePool->createDescriptorSet(descriptorSetLayout1);

        //    smart_refctd_ptr<nbl::video::IGPUPipelineLayout> pipelineLayout = m_device->createPipelineLayout({}, nullptr, smart_refctd_ptr(descriptorSetLayout1), nullptr, nullptr);

        //    IGPUComputePipeline::SCreationParams params = {};
        //    params.layout = pipelineLayout.get();
        //    params.shader.entryPoint = entryPoint;
        //    params.shader.shader = shader.get();
        //        
        //    m_device->createComputePipelines(nullptr, { &params,1 }, &m_solvePressurePipeline);

        //    {
        //        IGPUDescriptorSet::SDescriptorInfo infos[6];
        //        infos[0].desc = smart_refctd_ptr(gridDataBuffer);
        //        infos[0].info.buffer = {.offset = 0, .size = gridDataBuffer->getSize()};
        //        infos[1].desc = smart_refctd_ptr(pressureParamsBuffer);
        //        infos[1].info.buffer = {.offset = 0, .size = pressureParamsBuffer->getSize()};
        //        infos[2].desc = gridCellMaterialImageView;
        //        infos[2].info.image.imageLayout = IImage::LAYOUT::GENERAL;
        //        infos[2].info.combinedImageSampler.sampler = nullptr;
        //        infos[3].desc = divergenceImageView;
        //        infos[3].info.image.imageLayout = IImage::LAYOUT::GENERAL;
        //        infos[3].info.combinedImageSampler.sampler = nullptr;
        //        infos[4].desc = tempPressureImageView;
        //        infos[4].info.image.imageLayout = IImage::LAYOUT::GENERAL;
        //        infos[4].info.combinedImageSampler.sampler = nullptr;
        //        infos[5].desc = pressureImageView;
        //        infos[5].info.image.imageLayout = IImage::LAYOUT::GENERAL;
        //        infos[5].info.combinedImageSampler.sampler = nullptr;
        //        IGPUDescriptorSet::SWriteDescriptorSet writes[6] = {
        //            {.dstSet = m_solvePressureDs[0].get(), .binding = b_psGridData, .arrayElement = 0, .count = 1, .info = &infos[0]},
        //            {.dstSet = m_solvePressureDs[0].get(), .binding = b_psParams, .arrayElement = 0, .count = 1, .info = &infos[1]},
        //            {.dstSet = m_solvePressureDs[0].get(), .binding = b_psCMBuffer, .arrayElement = 0, .count = 1, .info = &infos[2]},
        //            {.dstSet = m_solvePressureDs[0].get(), .binding = b_psDivBuffer, .arrayElement = 0, .count = 1, .info = &infos[3]},
        //            {.dstSet = m_solvePressureDs[0].get(), .binding = b_psPresInBuffer, .arrayElement = 0, .count = 1, .info = &infos[4]},
        //            {.dstSet = m_solvePressureDs[0].get(), .binding = b_psPresOutBuffer, .arrayElement = 0, .count = 1, .info = &infos[5]},
        //        };
        //        m_device->updateDescriptorSets(std::span(writes, 6), {});
        //    }
        //    {
        //        IGPUDescriptorSet::SDescriptorInfo infos[6];
        //        infos[0].desc = smart_refctd_ptr(gridDataBuffer);
        //        infos[0].info.buffer = {.offset = 0, .size = gridDataBuffer->getSize()};
        //        infos[1].desc = smart_refctd_ptr(pressureParamsBuffer);
        //        infos[1].info.buffer = {.offset = 0, .size = pressureParamsBuffer->getSize()};
        //        infos[2].desc = gridCellMaterialImageView;
        //        infos[2].info.image.imageLayout = IImage::LAYOUT::GENERAL;
        //        infos[2].info.combinedImageSampler.sampler = nullptr;
        //        infos[3].desc = divergenceImageView;
        //        infos[3].info.image.imageLayout = IImage::LAYOUT::GENERAL;
        //        infos[3].info.combinedImageSampler.sampler = nullptr;
        //        infos[4].desc = pressureImageView;
        //        infos[4].info.image.imageLayout = IImage::LAYOUT::GENERAL;
        //        infos[4].info.combinedImageSampler.sampler = nullptr;
        //        infos[5].desc = tempPressureImageView;
        //        infos[5].info.image.imageLayout = IImage::LAYOUT::GENERAL;
        //        infos[5].info.combinedImageSampler.sampler = nullptr;
        //        IGPUDescriptorSet::SWriteDescriptorSet writes[6] = {
        //            {.dstSet = m_solvePressureDs[1].get(), .binding = b_psGridData, .arrayElement = 0, .count = 1, .info = &infos[0]},
        //            {.dstSet = m_solvePressureDs[1].get(), .binding = b_psParams, .arrayElement = 0, .count = 1, .info = &infos[1]},
        //            {.dstSet = m_solvePressureDs[1].get(), .binding = b_psCMBuffer, .arrayElement = 0, .count = 1, .info = &infos[2]},
        //            {.dstSet = m_solvePressureDs[1].get(), .binding = b_psDivBuffer, .arrayElement = 0, .count = 1, .info = &infos[3]},
        //            {.dstSet = m_solvePressureDs[1].get(), .binding = b_psPresInBuffer, .arrayElement = 0, .count = 1, .info = &infos[4]},
        //            {.dstSet = m_solvePressureDs[1].get(), .binding = b_psPresOutBuffer, .arrayElement = 0, .count = 1, .info = &infos[5]},
        //        };
        //        m_device->updateDescriptorSets(std::span(writes, 6), {});
        //    }
        //}
        {
            createComputePipeline(m_updateVelPsPipeline, m_updateVelPsPool, m_updateVelPsDs, 
                "app_resources/compute/pressureSolver.comp.hlsl", "updateVelocities", psUpdateVelPs_bs1);

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
                    {.dstSet = m_updateVelPsDs.get(), .binding = b_psCMBuffer, .arrayElement = 0, .count = 1, .info = &infos[2]},
                    {.dstSet = m_updateVelPsDs.get(), .binding = b_psVelBuffer, .arrayElement = 0, .count = 3, .info = imgInfosVel0},
                    {.dstSet = m_updateVelPsDs.get(), .binding = b_psPresInBuffer, .arrayElement = 0, .count = 1, .info = &infos[3]},
                };
                m_device->updateDescriptorSets(std::span(writes, 5), {});
            }
        }
        {
            // extrapolate velocities pipeline
            createComputePipeline(m_extrapolateVelPipeline, m_extrapolateVelPool, m_extrapolateVelDs, 
                "app_resources/compute/extrapolateVelocities.comp.hlsl", "main", evExtrapolateVel_bs1);

            {
                IGPUDescriptorSet::SDescriptorInfo infos[3];
                infos[0].desc = smart_refctd_ptr(gridDataBuffer);
                infos[0].info.buffer = {.offset = 0, .size = gridDataBuffer->getSize()};
                infos[1].desc = smart_refctd_ptr(particleBuffer);
                infos[1].info.buffer = {.offset = 0, .size = particleBuffer->getSize()};
                infos[2].desc = velocityFieldSampler;

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
                    {.dstSet = m_extrapolateVelDs.get(), .binding = b_evGridData, .arrayElement = 0, .count = 1, .info = &infos[0]},
                    {.dstSet = m_extrapolateVelDs.get(), .binding = b_evPBuffer, .arrayElement = 0, .count = 1, .info = &infos[1]},
                    {.dstSet = m_extrapolateVelDs.get(), .binding = b_evVelFieldBuffer, .arrayElement = 0, .count = 3, .info = imgInfosVel0},
                    {.dstSet = m_extrapolateVelDs.get(), .binding = b_evPrevVelFieldBuffer, .arrayElement = 0, .count = 3, .info = imgInfosVel1},
                    {.dstSet = m_extrapolateVelDs.get(), .binding = b_evVelSampler, .arrayElement = 0, .count = 1, .info = &infos[2]}
                };
                m_device->updateDescriptorSets(std::span(writes, 5), {});
            }
        }
        {
            // advect particles pipeline
            createComputePipeline(m_advectParticlesPipeline, m_advectParticlesPool, m_advectParticlesDs, "app_resources/compute/advectParticles.comp.hlsl", "main", apAdvectParticles_bs1);

            {
                IGPUDescriptorSet::SDescriptorInfo infos[3];
                infos[0].desc = smart_refctd_ptr(gridDataBuffer);
                infos[0].info.buffer = {.offset = 0, .size = gridDataBuffer->getSize()};
                infos[1].desc = smart_refctd_ptr(particleBuffer);
                infos[1].info.buffer = {.offset = 0, .size = particleBuffer->getSize()};
                infos[2].desc = velocityFieldSampler;

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
                    {.dstSet = m_advectParticlesDs.get(), .binding = b_apGridData, .arrayElement = 0, .count = 1, .info = &infos[0]},
                    {.dstSet = m_advectParticlesDs.get(), .binding = b_apPBuffer, .arrayElement = 0, .count = 1, .info = &infos[1]},
                    {.dstSet = m_advectParticlesDs.get(), .binding = b_apVelFieldBuffer, .arrayElement = 0, .count = 3, .info = imgInfosVel0},
                    {.dstSet = m_advectParticlesDs.get(), .binding = b_apVelSampler, .arrayElement = 0, .count = 1, .info = &infos[2]}
                };
                m_device->updateDescriptorSets(std::span(writes, 4), {});
            }
        }

        m_winMgr->show(m_window.get());

        return true;
    }

    inline void workLoopBody() override
    {
        const auto resourceIx = m_realFrameIx % m_maxFramesInFlight;

        if (m_realFrameIx >= m_maxFramesInFlight)
        {
            const ISemaphore::SWaitInfo cbDonePending[] =
            {
                {
                    .semaphore = m_renderSemaphore.get(),
                    .value = m_realFrameIx + 1 - m_maxFramesInFlight
                }
            };
            if (m_device->blockForSemaphores(cbDonePending) != ISemaphore::WAIT_RESULT::SUCCESS)
                return;
        }

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

        SMVPParams camData;
        SBufferRange<IGPUBuffer> camDataRange;
        {
            const auto viewMatrix = camera.getViewMatrix();
            const auto projectionMatrix = camera.getProjectionMatrix();
            const auto viewProjectionMatrix = camera.getConcatenatedMatrix();

            core::matrix3x4SIMD modelMatrix;
            modelMatrix.setTranslation(nbl::core::vectorSIMDf(0, 0, 0, 0));
            modelMatrix.setRotation(quaternion(0, 0, 0));

            core::matrix3x4SIMD modelViewMatrix = core::concatenateBFollowedByA(viewMatrix, modelMatrix);
            core::matrix4SIMD modelViewProjectionMatrix = core::concatenateBFollowedByA(viewProjectionMatrix, modelMatrix);

            auto modelMat = core::concatenateBFollowedByA(core::matrix4SIMD(), modelMatrix);

            const core::vector3df camPos = camera.getPosition().getAsVector3df();

            camPos.getAs4Values(camData.cameraPosition);
            memcpy(camData.MVP, modelViewProjectionMatrix.pointer(), sizeof(camData.MVP));
            memcpy(camData.M, modelMat.pointer(), sizeof(camData.M));
            memcpy(camData.V, viewMatrix.pointer(), sizeof(camData.V));
            memcpy(camData.P, projectionMatrix.pointer(), sizeof(camData.P));
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
        if (m_shouldInitParticles)
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
        }

        {
            SMemoryBarrier memBarrier;
            memBarrier.srcStageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS;
            memBarrier.srcAccessMask = ACCESS_FLAGS::MEMORY_WRITE_BITS;
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
            dispatchExtrapolateVelocities(cmdbuf);	// grid -> particle vel
            dispatchAdvection(cmdbuf);				// update/advect fluid
        }

        {
            SMemoryBarrier memBarrier;
            memBarrier.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
            memBarrier.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
            memBarrier.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
            memBarrier.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;
            cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, {.memBarriers = {&memBarrier, 1}});
        }

        // prepare particle vertices for render
        cmdbuf->bindComputePipeline(m_genParticleVerticesPipeline.get());
        cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_genParticleVerticesPipeline->getLayout(), 1, 1, &m_genVerticesDs.get());
        cmdbuf->dispatch(WorkgroupCountParticles, 1, 1);

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

        cmdbuf->bindGraphicsPipeline(m_graphicsPipeline.get());
        cmdbuf->bindDescriptorSets(EPBP_GRAPHICS, m_graphicsPipeline->getLayout(), 1, 1, &m_renderDs.get());

        cmdbuf->draw(numParticles * 6, 1, 0, 0);

        cmdbuf->endRenderPass();

        {
            SMemoryBarrier memBarrier;
            memBarrier.srcStageMask = PIPELINE_STAGE_FLAGS::ALL_GRAPHICS_BITS;
            memBarrier.srcAccessMask = ACCESS_FLAGS::MEMORY_WRITE_BITS;
            memBarrier.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
            memBarrier.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;
            cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, {.memBarriers = {&memBarrier, 1}});
        }

        cmdbuf->endDebugMarker();
        cmdbuf->end();

        // submit
        const IQueue::SSubmitInfo::SSemaphoreInfo rendered[1] = {{
            .semaphore = m_renderSemaphore.get(),
            .value = ++m_submitIx,
            .stageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT
        }};

        {
            {
                const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[1] = {{
                        .cmdbuf = cmdbuf
                    }};
                const IQueue::SSubmitInfo::SSemaphoreInfo acquired[1] = {{
                        .semaphore = m_currentImageAcquire.semaphore,
                        .value = m_currentImageAcquire.acquireCount,
                        .stageMask = PIPELINE_STAGE_FLAGS::NONE
                    }};
                const IQueue::SSubmitInfo infos[1] = {{
                    .waitSemaphores = acquired,
                    .commandBuffers = commandBuffers,
                    .signalSemaphores = rendered
                }};
                if (bCaptureTestInitParticles)
                    queue->startCapture();
                if (queue->submit(infos)!=IQueue::RESULT::SUCCESS)
                    m_submitIx--;
                if (bCaptureTestInitParticles)
                    queue->endCapture();
            }
        }

        m_surface->present(m_currentImageAcquire.imageIndex, rendered);
        m_realFrameIx++;
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

        cmdbuf->bindComputePipeline(m_accumulateWeightsPipeline.get());
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

        cmdbuf->bindComputePipeline(m_diffusionPipeline.get());
        cmdbuf->pushConstants(m_diffusionPipeline->getLayout(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0, sizeof(float32_t4), &diffParam);
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

            cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_diffusionPipeline->getLayout(), 1, 1, &m_diffusionDs[i % 2].get());
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

        cmdbuf->bindComputePipeline(m_updateVelDPipeline.get());
        cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_updateVelDPipeline->getLayout(), 1, 1, &m_updateVelDDs.get());
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
        for (int i = 0; i < 5; i++)
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
    
    void dispatchExtrapolateVelocities(IGPUCommandBuffer* cmdbuf)
    {
        {
            SMemoryBarrier memBarrier;
            memBarrier.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
            memBarrier.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
            memBarrier.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
            memBarrier.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;
            cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, {.memBarriers = {&memBarrier, 1}});
        }

        cmdbuf->bindComputePipeline(m_extrapolateVelPipeline.get());
        cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_extrapolateVelPipeline->getLayout(), 1, 1, &m_extrapolateVelDs.get());
        cmdbuf->dispatch(WorkgroupCountParticles, 1, 1);
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

        cmdbuf->bindComputePipeline(m_advectParticlesPipeline.get());
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

    smart_refctd_ptr<IGPUShader> compileShader(const std::string& filePath, const std::string& entryPoint = "main")
    {
        IAssetLoader::SAssetLoadParams lparams = {};
        lparams.logger = m_logger.get();
        lparams.workingDirectory = "";
        auto bundle = m_assetMgr->getAsset(filePath, lparams);
        if (bundle.getContents().empty() || bundle.getAssetType() != IAsset::ET_SHADER)
        {
            m_logger->log("Shader %s not found!", ILogger::ELL_ERROR, filePath);
            exit(-1);
        }
        
        const auto assets = bundle.getContents();
        assert(assets.size() == 1);
        smart_refctd_ptr<ICPUShader> shaderSrc = IAsset::castDown<ICPUShader>(assets[0]);

        smart_refctd_ptr<ICPUShader> shader = shaderSrc;
        if (entryPoint != "main")
        {
            auto compiler = make_smart_refctd_ptr<asset::CHLSLCompiler>(smart_refctd_ptr(m_system));
            CHLSLCompiler::SOptions options = {};
            options.stage = shaderSrc->getStage();
            if (!(options.stage == IShader::E_SHADER_STAGE::ESS_COMPUTE || options.stage == IShader::E_SHADER_STAGE::ESS_FRAGMENT))
                options.stage = IShader::E_SHADER_STAGE::ESS_VERTEX;
            options.targetSpirvVersion = m_device->getPhysicalDevice()->getLimits().spirvVersion;
            options.spirvOptimizer = nullptr;
            options.debugInfoFlags |= IShaderCompiler::E_DEBUG_INFO_FLAGS::EDIF_SOURCE_BIT;
            options.preprocessorOptions.sourceIdentifier = shaderSrc->getFilepathHint();
            options.preprocessorOptions.logger = m_logger.get();
            options.preprocessorOptions.includeFinder = compiler->getDefaultIncludeFinder();

            std::string dxcOptionStr[] = {"-E " + entryPoint};
            options.dxcOptions = std::span(dxcOptionStr);

            shader = compiler->compileToSPIRV((const char*)shaderSrc->getContent()->getPointer(), options);
        }

        return m_device->createShader(shader.get());
    }

    bool createBuffer(smart_refctd_ptr<IGPUBuffer>& buffer, video::IGPUBuffer::SCreationParams& params)
    {
        buffer = m_device->createBuffer(std::move(params));
        if (!buffer)
            return logFail("Failed to create GPU buffer of size %d!\n", params.size);

        video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = buffer->getMemoryReqs();
        reqs.memoryTypeBits &= m_physicalDevice->getDeviceLocalMemoryTypeBits();

        auto bufMem = m_device->allocate(reqs, buffer.get());
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
        m_renderSemaphore = m_device->createSemaphore(m_submitIx);
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

        m_maxFramesInFlight = m_surface->getMaxFramesInFlight();
        if (FRAMES_IN_FLIGHT < m_maxFramesInFlight)
        {
            m_logger->log("Lowering frames in flight!\n", ILogger::ELL_WARNING);
            m_maxFramesInFlight = FRAMES_IN_FLIGHT;
        }

        m_cmdPool = m_device->createCommandPool(queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
        for (auto i = 0u; i < m_maxFramesInFlight; i++)
        {
            if (!m_cmdPool)
                return logFail("Couldn't create command pool\n");

            if (!m_cmdPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { m_cmdBufs.data() + i, 1 }))
                return logFail("Couldn't create command buffer\n");
        }

        m_winMgr->setWindowSize(m_window.get(), WIN_WIDTH, WIN_HEIGHT);
        m_surface->recreateSwapchain();

        // init shaders and pipeline

        auto compileShader = [&](const std::string& filePath, IShader::E_SHADER_STAGE stage) -> smart_refctd_ptr<IGPUShader>
            {
                IAssetLoader::SAssetLoadParams lparams = {};
                lparams.logger = m_logger.get();
                lparams.workingDirectory = "";
                auto bundle = m_assetMgr->getAsset(filePath, lparams);
                if (bundle.getContents().empty() || bundle.getAssetType() != IAsset::ET_SHADER)
                {
                    m_logger->log("Shader %s not found!", ILogger::ELL_ERROR, filePath);
                    exit(-1);
                }
        
                const auto assets = bundle.getContents();
                assert(assets.size() == 1);
                smart_refctd_ptr<ICPUShader> shaderSrc = IAsset::castDown<ICPUShader>(assets[0]);
                shaderSrc->setShaderStage(stage);
                if (!shaderSrc)
                    return nullptr;

                return m_device->createShader(shaderSrc.get());
            };
        auto vs = compileShader("app_resources/fluidParticles.vertex.hlsl", IShader::E_SHADER_STAGE::ESS_VERTEX);
        auto fs = compileShader("app_resources/fluidParticles.fragment.hlsl", IShader::E_SHADER_STAGE::ESS_FRAGMENT);

        smart_refctd_ptr<video::IGPUDescriptorSetLayout> descriptorSetLayout1, descriptorSetLayout2;
        {
            // init descriptors
            video::IGPUDescriptorSetLayout::SBinding bindingsSet1[] = {
                {
                    .binding = 0u,
                    .type = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
                    .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
                    .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_FRAGMENT,
                    .count = 1u,
                },
                {
                    .binding = 1u,
                    .type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
                    .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
                    .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_VERTEX,
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
            IGPUDescriptorSet::SDescriptorInfo verticesInfo;
            verticesInfo.desc = smart_refctd_ptr(particleVertexBuffer);
            verticesInfo.info.buffer = {.offset = 0, .size = particleVertexBuffer->getSize()};
            IGPUDescriptorSet::SWriteDescriptorSet writes[2] = {
                {.dstSet = m_renderDs.get(), .binding = 0, .arrayElement = 0, .count = 1, .info = &camInfo},
                {.dstSet = m_renderDs.get(), .binding = 1, .arrayElement = 0, .count = 1, .info = &verticesInfo},
            };
            m_device->updateDescriptorSets(std::span(writes, 2), {});
        }

        SBlendParams blendParams = {};
        blendParams.logicOp = ELO_NO_OP;
        blendParams.blendParams[0u].srcColorFactor = asset::EBF_SRC_ALPHA;
        blendParams.blendParams[0u].dstColorFactor = asset::EBF_ONE_MINUS_SRC_ALPHA;
        blendParams.blendParams[0u].colorBlendOp = asset::EBO_ADD;
        blendParams.blendParams[0u].srcAlphaFactor = asset::EBF_ONE_MINUS_SRC_ALPHA;
        blendParams.blendParams[0u].dstAlphaFactor = asset::EBF_ZERO;
        blendParams.blendParams[0u].alphaBlendOp = asset::EBO_ADD;
        blendParams.blendParams[0u].colorWriteMask = (1u << 0u) | (1u << 1u) | (1u << 2u) | (1u << 3u);

        {
            IGPUShader::SSpecInfo specInfo[3] = {
                {.shader = vs.get()},
                {.shader = fs.get()},
            };

            const auto pipelineLayout = m_device->createPipelineLayout({}, nullptr, smart_refctd_ptr(descriptorSetLayout1), smart_refctd_ptr(descriptorSetLayout2), nullptr);

            SRasterizationParams rasterizationParams{};
            rasterizationParams.faceCullingMode = EFCM_NONE;
            rasterizationParams.depthWriteEnable = true;

            IGPUGraphicsPipeline::SCreationParams params[1] = {};
            params[0].layout = pipelineLayout.get();
            params[0].shaders = specInfo;
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
        
        cmdbuf->bindComputePipeline(m_initParticlePipeline.get());
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
        IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t imageBarriers[16];
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
        fillGridBarrierInfo(imageBarriers[count++], tempDiffusionImageView);

        fillGridBarrierInfo(imageBarriers[count++], pressureImageView);
        fillGridBarrierInfo(imageBarriers[count++], tempPressureImageView);
        fillGridBarrierInfo(imageBarriers[count++], divergenceImageView);

        cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = imageBarriers });
    }


    smart_refctd_ptr<IWindow> m_window;
    smart_refctd_ptr<CSimpleResizeSurface<CSwapchainFramebuffersAndDepth>> m_surface;
    smart_refctd_ptr<IGPUGraphicsPipeline> m_graphicsPipeline;
    smart_refctd_ptr<ISemaphore> m_renderSemaphore;
    smart_refctd_ptr<IGPUCommandPool> m_cmdPool;
    std::array<smart_refctd_ptr<IGPUCommandBuffer>, ISwapchain::MaxImages> m_cmdBufs;
    uint64_t m_realFrameIx : 59 = 0;
    uint64_t m_submitIx : 59 = 0;
    uint64_t m_maxFramesInFlight : 5;
    ISimpleManagedSurface::SAcquireResult m_currentImageAcquire = {};

    smart_refctd_ptr<video::IDescriptorPool> m_renderDsPool;
    smart_refctd_ptr<IGPUDescriptorSet> m_renderDs;

    // simulation compute shaders
    smart_refctd_ptr<IGPUComputePipeline> m_initParticlePipeline;

    smart_refctd_ptr<IGPUComputePipeline> m_accumulateWeightsPipeline;
    smart_refctd_ptr<IGPUComputePipeline> m_updateFluidCellsPipeline;
    smart_refctd_ptr<IGPUComputePipeline> m_updateNeighborCellsPipeline;

    smart_refctd_ptr<IGPUComputePipeline> m_applyBodyForcesPipeline;
    
    smart_refctd_ptr<IGPUComputePipeline> m_axisCellsPipeline;
    smart_refctd_ptr<IGPUComputePipeline> m_neighborAxisCellsPipeline;
    smart_refctd_ptr<IGPUComputePipeline> m_diffusionPipeline;
    smart_refctd_ptr<IGPUComputePipeline> m_updateVelDPipeline;

    smart_refctd_ptr<IGPUComputePipeline> m_calcDivergencePipeline;
    smart_refctd_ptr<IGPUComputePipeline> m_iteratePressurePipeline;
    // smart_refctd_ptr<IGPUComputePipeline> m_solvePressurePipeline;
    smart_refctd_ptr<IGPUComputePipeline> m_updateVelPsPipeline;

    smart_refctd_ptr<IGPUComputePipeline> m_extrapolateVelPipeline;
    smart_refctd_ptr<IGPUComputePipeline> m_advectParticlesPipeline;
    smart_refctd_ptr<IGPUComputePipeline> m_genParticleVerticesPipeline;

    // descriptors
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
    std::array<smart_refctd_ptr<IGPUDescriptorSet>, 2> m_diffusionDs;
    smart_refctd_ptr<video::IDescriptorPool> m_updateVelDPool;
    smart_refctd_ptr<IGPUDescriptorSet> m_updateVelDDs;

    smart_refctd_ptr<video::IDescriptorPool> m_calcDivergencePool;
    smart_refctd_ptr<IGPUDescriptorSet> m_calcDivergenceDs;
    smart_refctd_ptr<video::IDescriptorPool> m_iteratePressurePool;
    smart_refctd_ptr<IGPUDescriptorSet> m_iteratePressureDs;
    // smart_refctd_ptr<video::IDescriptorPool> m_solvePressurePool;
    // std::array<smart_refctd_ptr<IGPUDescriptorSet>, 2> m_solvePressureDs;
    smart_refctd_ptr<video::IDescriptorPool> m_updateVelPsPool;
    smart_refctd_ptr<IGPUDescriptorSet> m_updateVelPsDs;
    
    smart_refctd_ptr<video::IDescriptorPool> m_extrapolateVelPool;
    smart_refctd_ptr<IGPUDescriptorSet> m_extrapolateVelDs;
    smart_refctd_ptr<video::IDescriptorPool> m_advectParticlesPool;
    smart_refctd_ptr<IGPUDescriptorSet> m_advectParticlesDs;
    smart_refctd_ptr<video::IDescriptorPool> m_genVerticesPool;
    smart_refctd_ptr<IGPUDescriptorSet> m_genVerticesDs;

    // input system
    smart_refctd_ptr<InputSystem> m_inputSystem;
    InputSystem::ChannelReader<IMouseEventChannel> mouse;
    InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;

    Camera camera = Camera(core::vectorSIMDf(0,0,0), core::vectorSIMDf(0,0,0), core::matrix4SIMD());
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
    const uint32_t diffusionIterations = 15;
    const uint32_t pressureSolverIterations = 15;

    // buffers
    smart_refctd_ptr<IGPUBuffer> cameraBuffer;

    smart_refctd_ptr<IGPUBuffer> particleBuffer;		// Particle
    smart_refctd_ptr<IGPUBuffer> tempParticleBuffer;	// Particle
    smart_refctd_ptr<IGPUBuffer> pParamsBuffer;			// SParticleRenderParams
    smart_refctd_ptr<IGPUBuffer> particleVertexBuffer;	// VertexInfo * 6 vertices

    smart_refctd_ptr<IGPUBuffer> gridDataBuffer;		// SGridData
    smart_refctd_ptr<IGPUBuffer> pressureParamsBuffer;	// SPressureSolverParams
    smart_refctd_ptr<IGPUImageView> gridParticleCountImageView;	// uint
    smart_refctd_ptr<IGPUImageView> gridCellMaterialImageView;	// uint, fluid or solid

    std::array<smart_refctd_ptr<IGPUImageView>, 3> velocityFieldImageViews;		// float * 3 (per axis)
    std::array<smart_refctd_ptr<IGPUImageView>, 3> prevVelocityFieldImageViews;	// float * 3
    smart_refctd_ptr<IGPUSampler> velocityFieldSampler;

    std::array<smart_refctd_ptr<IGPUImageView>, 3> velocityFieldUintViews;
    std::array<smart_refctd_ptr<IGPUImageView>, 3> prevVelocityFieldUintViews;

    smart_refctd_ptr<IGPUImageView> gridDiffusionImageView;	// float4
    smart_refctd_ptr<IGPUImageView> gridAxisCellMaterialImageView;	// uint4
    smart_refctd_ptr<IGPUImageView> divergenceImageView;		// float
    smart_refctd_ptr<IGPUImageView> pressureImageView;		// float

    smart_refctd_ptr<IGPUImageView> tempCellMaterialImageView;	// uint, fluid or solid
    smart_refctd_ptr<IGPUImageView> tempDiffusionImageView;	// float4
    smart_refctd_ptr<IGPUImageView> tempAxisCellMaterialImageView;	// uint4
    smart_refctd_ptr<IGPUImageView> tempPressureImageView;	// float
};

NBL_MAIN_FUNC(FLIPFluidsApp)