// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include <nabla.h>
#include "nbl/ext/ScreenShot/ScreenShot.h"
#include "compute/common.h"

using namespace nbl;
using namespace core;

class IESCompute
{
    public:
        IESCompute(video::IVideoDriver* _driver, asset::IAssetManager* _assetManager, const asset::CIESProfile& _profile)
            : profile(_profile), driver(_driver), pushConstant({ (float)profile.getMaxValue(), 0.f})
        {
            createGPUEnvironment(_assetManager); 

            fbo = createFBO<asset::EF_R16G16B16A16_SFLOAT>(driver->getScreenSize().Width, driver->getScreenSize().Height);
        }
        ~IESCompute() {}

        enum E_MODE : uint32_t
        {
            EM_CDC,         //! Candlepower Distribution Curve
            EM_IES_C,       //! IES Candela
            EM_SPERICAL_C,  //! Sperical coordinates
            EM_DIRECTION,   //! Sample direction
            EM_PASS_T_MASK, //! Test mask
            EM_SIZE
        };

        enum E_BINDINGS
        {
            EB_IMAGE_IES_C, //! Image with IES Candela data
            EB_IMAGE_S,     //! Image with spehircal coordinates data
            EB_IMAGE_D,     //! Image with direction data
            EB_IMAGE_T_MASK,//! Image with test mask data
            EB_SSBO_HA,     //! IES Profile SSBO Horizontal Angles 
            EB_SSBO_VA,     //! IES Profile SSBO Vertical Angles
            EB_SSBO_D,      //! IES Profile SSBO Data
            EB_SIZE
        };

        void begin()
        {
            driver->setRenderTarget(fbo);
            const float clear[4]{ 0.f,0.f,0.f,1.f };
            driver->clearColorBuffer(video::EFAP_COLOR_ATTACHMENT0, clear);
            driver->beginScene(true, false, video::SColor(255, 0, 0, 0));
        }

        void dispatch()
        {
            auto& gpue = m_gpue;

            driver->bindComputePipeline(gpue.cPipeline.get());
            driver->bindDescriptorSets(EPBP_COMPUTE, gpue.cPipeline->getLayout(), 0u, 1u, &gpue.cDescriptorSet.get(), nullptr);
            driver->pushConstants(gpue.cPipeline->getLayout(), asset::ISpecializedShader::ESS_COMPUTE, 0u, sizeof(PushConstant), &pushConstant);

            _NBL_STATIC_INLINE_CONSTEXPR auto xGroups = (TEXTURE_SIZE - 1u) / WORKGROUP_DIMENSION + 1u;
            driver->dispatch(xGroups, xGroups, 1u);

            COpenGLExtensionHandler::extGlMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        }

        void renderpass()
        {
            auto& gpue = m_gpue;

            driver->bindGraphicsPipeline(gpue.gPipeline.get());
            driver->bindDescriptorSets(video::EPBP_GRAPHICS, gpue.gPipeline->getLayout(), 3u, 1u, &gpue.gDescriptorSet.get(), nullptr);
            driver->pushConstants(gpue.gPipeline->getLayout(), asset::ISpecializedShader::ESS_FRAGMENT, 0u, sizeof(PushConstant), &pushConstant);
            driver->drawMeshBuffer(gpue.mBuffer.get());
        }

        void end()
        {
            driver->blitRenderTargets(fbo, nullptr, false, false);
            driver->endScene();
        }

        void updateZDegree(const asset::CIESProfile::IES_STORAGE_FORMAT& degreeOffset)
        {
            //const auto newDegreeRotation = std::clamp<float>(pushConstant.zAngleDegreeRotation + degreeOffset, profile.getHoriAngles().front(), profile.getHoriAngles().back());
            // TMP

            const auto newDegreeRotation = std::clamp<float>(pushConstant.zAngleDegreeRotation + degreeOffset, 0, 360);
            pushConstant.zAngleDegreeRotation = newDegreeRotation;
        }

        const auto& getZDegree()
        {
            return pushConstant.zAngleDegreeRotation;
        }

        void updateMode(const E_MODE& mode)
        {
            pushConstant.mode = static_cast<decltype(pushConstant.mode)>(mode);
        }

        const auto& getMode()
        {
            return pushConstant.mode;
        }

    private:

        void createGPUEnvironment(asset::IAssetManager* _assetManager)
        {
            auto gpuSpecializedShaderFromFile = [&](const char* path)
            {
                auto bundle = _assetManager->getAsset(path, {});
                auto shader = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*bundle.getContents().begin());

                return driver->getGPUObjectsFromAssets<asset::ICPUSpecializedShader>(&shader, &shader + 1u)->operator[](0); // omg
            };

            auto& gpue = m_gpue;

            gpue.dImageIESC = std::move(createGPUImageView<asset::EF_R16_UNORM>(TEXTURE_SIZE, TEXTURE_SIZE));
            gpue.dImageS = std::move(createGPUImageView<asset::EF_R32G32_SFLOAT>(TEXTURE_SIZE, TEXTURE_SIZE));
            gpue.dImageD = std::move(createGPUImageView<asset::EF_R32G32B32A32_SFLOAT>(TEXTURE_SIZE, TEXTURE_SIZE));
            gpue.dImageTMask = std::move(createGPUImageView<asset::EF_R8G8_UNORM>(TEXTURE_SIZE, TEXTURE_SIZE));

            // Compute
            {
                const std::vector<IGPUDescriptorSetLayout::SBinding> bindings =
                {
                    { EB_IMAGE_IES_C, asset::EDT_STORAGE_IMAGE, 1, asset::ISpecializedShader::ESS_COMPUTE, nullptr },
                    { EB_IMAGE_S, asset::EDT_STORAGE_IMAGE, 1, asset::ISpecializedShader::ESS_COMPUTE, nullptr },
                    { EB_IMAGE_D, asset::EDT_STORAGE_IMAGE, 1, asset::ISpecializedShader::ESS_COMPUTE, nullptr },
                    { EB_IMAGE_T_MASK, asset::EDT_STORAGE_IMAGE, 1, asset::ISpecializedShader::ESS_COMPUTE, nullptr },
                    { EB_SSBO_HA, asset::EDT_STORAGE_BUFFER, 1, asset::ISpecializedShader::ESS_COMPUTE, nullptr },
                    { EB_SSBO_VA, asset::EDT_STORAGE_BUFFER, 1, asset::ISpecializedShader::ESS_COMPUTE, nullptr },
                    { EB_SSBO_D, asset::EDT_STORAGE_BUFFER, 1, asset::ISpecializedShader::ESS_COMPUTE, nullptr }
                };

                {
                    auto descriptorSetLayout = driver->createGPUDescriptorSetLayout(bindings.data(), bindings.data() + bindings.size());
                    asset::SPushConstantRange range = { asset::ISpecializedShader::ESS_COMPUTE, 0u, sizeof(PushConstant) };

                    gpue.cPipeline = driver->createGPUComputePipeline(nullptr, driver->createGPUPipelineLayout(&range, &range + 1u, core::smart_refctd_ptr(descriptorSetLayout)), gpuSpecializedShaderFromFile(getShaderPath().data()));
                    gpue.cDescriptorSet = driver->createGPUDescriptorSet(std::move(descriptorSetLayout));
                }

                {
                    IGPUDescriptorSet::SDescriptorInfo infos[EB_SIZE];
                    {
                        createSSBODescriptorInfo<EB_SSBO_HA>(profile, infos[EB_SSBO_HA]);
                        createSSBODescriptorInfo<EB_SSBO_VA>(profile, infos[EB_SSBO_VA]);
                        createSSBODescriptorInfo<EB_SSBO_D>(profile, infos[EB_SSBO_D]);

                        {
                            infos[EB_IMAGE_IES_C].desc = core::smart_refctd_ptr(gpue.dImageIESC);
                            infos[EB_IMAGE_IES_C].image = { nullptr, asset::EIL_GENERAL };

                            infos[EB_IMAGE_S].desc = core::smart_refctd_ptr(gpue.dImageS);
                            infos[EB_IMAGE_S].image = { nullptr, asset::EIL_GENERAL };

                            infos[EB_IMAGE_D].desc = core::smart_refctd_ptr(gpue.dImageD);
                            infos[EB_IMAGE_D].image = { nullptr, asset::EIL_GENERAL };

                            infos[EB_IMAGE_T_MASK].desc = core::smart_refctd_ptr(gpue.dImageTMask);
                            infos[EB_IMAGE_T_MASK].image = { nullptr, asset::EIL_GENERAL };
                        }
                    }

                    IGPUDescriptorSet::SWriteDescriptorSet writes[EB_SIZE];
                    for (auto i = 0; i < EB_SIZE; i++)
                    {
                        writes[i].dstSet = gpue.cDescriptorSet.get();
                        writes[i].binding = i;
                        writes[i].arrayElement = 0u;
                        writes[i].count = 1u;
                        writes[i].info = &infos[i];
                    }

                    writes[EB_IMAGE_IES_C].descriptorType = asset::EDT_STORAGE_IMAGE;
                    writes[EB_IMAGE_S].descriptorType = asset::EDT_STORAGE_IMAGE;
                    writes[EB_IMAGE_D].descriptorType = asset::EDT_STORAGE_IMAGE;
                    writes[EB_IMAGE_T_MASK].descriptorType = asset::EDT_STORAGE_IMAGE;
                    writes[EB_SSBO_HA].descriptorType = asset::EDT_STORAGE_BUFFER;
                    writes[EB_SSBO_VA].descriptorType = asset::EDT_STORAGE_BUFFER;
                    writes[EB_SSBO_D].descriptorType = asset::EDT_STORAGE_BUFFER;

                    driver->updateDescriptorSets(EB_SIZE, writes, 0u, nullptr);
                }
            }

            // Graphics
            {
                const std::vector<IGPUDescriptorSetLayout::SBinding> bindings =
                {
                    { EB_IMAGE_IES_C, asset::EDT_COMBINED_IMAGE_SAMPLER, 1, asset::ISpecializedShader::ESS_FRAGMENT, nullptr },
                    { EB_IMAGE_S, asset::EDT_COMBINED_IMAGE_SAMPLER, 1, asset::ISpecializedShader::ESS_FRAGMENT, nullptr },
                    { EB_IMAGE_D, asset::EDT_COMBINED_IMAGE_SAMPLER, 1, asset::ISpecializedShader::ESS_FRAGMENT, nullptr },
                    { EB_IMAGE_T_MASK, asset::EDT_COMBINED_IMAGE_SAMPLER, 1, asset::ISpecializedShader::ESS_FRAGMENT, nullptr }
                };

                {
                    auto descriptorSetLayout = driver->createGPUDescriptorSetLayout(bindings.data(), bindings.data() + bindings.size());

                    auto mesh = _assetManager->getGeometryCreator()->createRectangleMesh(vector2df_SIMD(1.0, 1.0));
                    auto cpusphere = core::make_smart_refctd_ptr<asset::ICPUMeshBuffer>(nullptr, nullptr, mesh.bindings, std::move(mesh.indexBuffer));

                    cpusphere->setBoundingBox(mesh.bbox);
                    cpusphere->setIndexType(mesh.indexType);
                    cpusphere->setIndexCount(mesh.indexCount);

                    auto vShader = gpuSpecializedShaderFromFile("../shader.vert");
                    auto fShader = gpuSpecializedShaderFromFile("../shader.frag");

                    video::IGPUSpecializedShader* shaders[] = { vShader.get(), fShader.get() };
                    asset::SRasterizationParams raster;

                    asset::SPushConstantRange range = { asset::ISpecializedShader::ESS_FRAGMENT, 0u, sizeof(PushConstant) };
                    gpue.gPipeline = driver->createGPURenderpassIndependentPipeline(nullptr, driver->createGPUPipelineLayout(&range, &range + 1u, nullptr, nullptr, nullptr, core::smart_refctd_ptr(descriptorSetLayout)), shaders, shaders + 2, mesh.inputParams, asset::SBlendParams{}, mesh.assemblyParams, raster);
                    gpue.gDescriptorSet = driver->createGPUDescriptorSet(core::smart_refctd_ptr(descriptorSetLayout));
                    gpue.mBuffer = driver->getGPUObjectsFromAssets(&cpusphere.get(), &cpusphere.get() + 1)->front();
                }
                

                auto createSampler = [&]()
                {
                    return driver->createGPUSampler({ asset::ISampler::ETC_CLAMP_TO_EDGE,asset::ISampler::ETC_CLAMP_TO_EDGE,asset::ISampler::ETC_CLAMP_TO_EDGE,asset::ISampler::ETBC_FLOAT_OPAQUE_BLACK,asset::ISampler::ETF_LINEAR,asset::ISampler::ETF_LINEAR,asset::ISampler::ESMM_LINEAR,0u,false,asset::ECO_ALWAYS });
                };

                _NBL_STATIC_INLINE_CONSTEXPR uint8_t NBL_D_IMAGES_AMOUNT = 4u;

                IGPUDescriptorSet::SDescriptorInfo infos[NBL_D_IMAGES_AMOUNT];
                {
                    infos[EB_IMAGE_IES_C].desc = core::smart_refctd_ptr(gpue.dImageIESC);
                    infos[EB_IMAGE_IES_C].image = { createSampler(),asset::EIL_SHADER_READ_ONLY_OPTIMAL};

                    infos[EB_IMAGE_S].desc = core::smart_refctd_ptr(gpue.dImageS);
                    infos[EB_IMAGE_S].image = { createSampler(),asset::EIL_SHADER_READ_ONLY_OPTIMAL };

                    infos[EB_IMAGE_D].desc = core::smart_refctd_ptr(gpue.dImageD);
                    infos[EB_IMAGE_D].image = { createSampler(),asset::EIL_SHADER_READ_ONLY_OPTIMAL };

                    infos[EB_IMAGE_T_MASK].desc = core::smart_refctd_ptr(gpue.dImageTMask);
                    infos[EB_IMAGE_T_MASK].image = { createSampler(),asset::EIL_SHADER_READ_ONLY_OPTIMAL };
                }

                video::IGPUDescriptorSet::SWriteDescriptorSet writes[NBL_D_IMAGES_AMOUNT];
                for (auto i = 0; i < NBL_D_IMAGES_AMOUNT; i++)
                {
                    writes[i].dstSet = gpue.gDescriptorSet.get();
                    writes[i].binding = i;
                    writes[i].count = 1u;
                    writes[i].arrayElement = 0u;
                    writes[i].descriptorType = asset::EDT_COMBINED_IMAGE_SAMPLER;
                    writes[i].info = &infos[i];
                }

                driver->updateDescriptorSets(NBL_D_IMAGES_AMOUNT, writes, 0u, nullptr);
            }
        }

        template<E_BINDINGS binding>
        auto createSSBODescriptorInfo(const asset::CIESProfile& profile, IGPUDescriptorSet::SDescriptorInfo& info)
        {
            static_assert(binding == EB_SSBO_HA || binding == EB_SSBO_VA || binding == EB_SSBO_D);

            auto createBuffer = [&](const auto& pInput)
            {
                core::smart_refctd_ptr<asset::ICPUBuffer> buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(asset::CIESProfile::IES_STORAGE_FORMAT) * pInput.size());
                memcpy(buffer->getPointer(), pInput.data(), buffer->getSize());

                return buffer;
            };

            core::smart_refctd_ptr<asset::ICPUBuffer> buffer;

            if constexpr (binding == EB_SSBO_HA)
                buffer = createBuffer(profile.getHoriAngles());
            else if (binding == EB_SSBO_VA)
                buffer = createBuffer(profile.getVertAngles());
            else
                buffer = createBuffer(profile.getData());

            auto ssbo = driver->createFilledDeviceLocalGPUBufferOnDedMem(buffer->getSize(), buffer->getPointer());;
            info.desc = core::smart_refctd_ptr(ssbo);
            info.buffer = { 0, ssbo->getSize() };
        }

        template<asset::E_FORMAT format>
        auto createGPUImageView(const size_t& width, const size_t& height)
        {
            IGPUImage::SCreationParams imageInfo;
            imageInfo.format = format;
            imageInfo.type = IGPUImage::ET_2D;
            imageInfo.extent.width = width;
            imageInfo.extent.height = height;
            imageInfo.extent.depth = 1u;

            imageInfo.mipLevels = 1u;
            imageInfo.arrayLayers = 1u;
            imageInfo.samples = asset::ICPUImage::ESCF_1_BIT;
            imageInfo.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);

            auto image = driver->createGPUImageOnDedMem(std::move(imageInfo), driver->getDeviceLocalGPUMemoryReqs());

            IGPUImageView::SCreationParams imgViewInfo;
            imgViewInfo.image = std::move(image);
            imgViewInfo.format = format;
            imgViewInfo.viewType = IGPUImageView::ET_2D;
            imgViewInfo.flags = static_cast<IGPUImageView::E_CREATE_FLAGS>(0u);
            imgViewInfo.subresourceRange.baseArrayLayer = 0u;
            imgViewInfo.subresourceRange.baseMipLevel = 0u;
            imgViewInfo.subresourceRange.layerCount = 1u;
            imgViewInfo.subresourceRange.levelCount = 1u;

            return driver->createGPUImageView(std::move(imgViewInfo));
        }

        _NBL_STATIC_INLINE_CONSTEXPR std::string_view getShaderPath()
        {
            return "../compute/cdc.comp";
        }

        template<asset::E_FORMAT format>
        video::IFrameBuffer* createFBO(const size_t& width, const size_t& height)
        {
            auto* fbo = driver->addFrameBuffer();

            bBuffer = createGPUImageView<format>(width, height);
            fbo->attach(video::EFAP_COLOR_ATTACHMENT0, core::smart_refctd_ptr(bBuffer));
          
            return fbo;
        }

        const asset::CIESProfile& profile;
        video::IVideoDriver* const driver;

        struct GPUE
        {
            // Compute
            core::smart_refctd_ptr<video::IGPUComputePipeline> cPipeline;
            core::smart_refctd_ptr<video::IGPUDescriptorSet> cDescriptorSet;

            // Graphics
            core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> gPipeline;
            core::smart_refctd_ptr<video::IGPUDescriptorSet> gDescriptorSet;
            core::smart_refctd_ptr<video::IGPUMeshBuffer> mBuffer;

            // Shared data
            core::smart_refctd_ptr<video::IGPUImageView> dImageIESC;
            core::smart_refctd_ptr<video::IGPUImageView> dImageS;
            core::smart_refctd_ptr<video::IGPUImageView> dImageD;
            core::smart_refctd_ptr<video::IGPUImageView> dImageTMask;
        } m_gpue;

        #include "nbl/nblpack.h"
        struct PushConstant
        {
            float maxIValueReciprocal;
            float zAngleDegreeRotation;
            IESCompute::E_MODE mode = IESCompute::EM_CDC;
            uint32_t dummy;
        } PACK_STRUCT;
        #include "nbl/nblunpack.h"
        
        PushConstant pushConstant;

        video::IFrameBuffer* fbo = nullptr;
        core::smart_refctd_ptr<video::IGPUImageView> bBuffer;
};

class IESExampleEventReceiver : public nbl::IEventReceiver
{
public:
    IESExampleEventReceiver() {}

    bool OnEvent(const nbl::SEvent& event)
    {
        if (event.EventType == nbl::EET_MOUSE_INPUT_EVENT)
        {
            zDegreeOffset = event.MouseInput.Wheel;

            return true;
        }

        if (event.EventType == nbl::EET_KEY_INPUT_EVENT && !event.KeyInput.PressedDown)
        {
            switch (event.KeyInput.Key)
            {
                case nbl::KEY_KEY_C:
                {
                    mode = IESCompute::EM_CDC;
                    return true;
                }
                case nbl::KEY_KEY_V:
                {
                    mode = IESCompute::EM_IES_C;
                    return true;
                }
                case nbl::KEY_KEY_S:
                {
                    mode = IESCompute::EM_SPERICAL_C;
                    return true;
                }
                case nbl::KEY_KEY_D:
                {
                    mode = IESCompute::EM_DIRECTION;
                    return true;
                }
                case nbl::KEY_KEY_M:
                {
                    mode = IESCompute::EM_PASS_T_MASK;
                    return true;
                }
                case nbl::KEY_KEY_Q:
                {
                    running = false;
                    return true;
                }
            }
        }

        return false;
    }

    void reset() { zDegreeOffset = 0; }
    inline const auto& isRunning() const { return running; }
    inline const auto& getMode() const { return mode; }
    template<typename T = double>
    inline const auto& getZDegreeOffset() const { return static_cast<T>(zDegreeOffset); }

private:
    _NBL_STATIC_INLINE_CONSTEXPR size_t DEGREE_SHIFT = 5u;
    double zDegreeOffset = 0.0;
    IESCompute::E_MODE mode = IESCompute::EM_CDC;
    bool running = true;
};

int main()
{
    nbl::SIrrlichtCreationParameters params;
    params.Bits = 24;
    params.ZBufferBits = 24;
    params.DriverType = video::EDT_OPENGL;
    params.WindowSize = dimension2d<uint32_t>(640, 640);
    params.Fullscreen = false;
    params.Vsync = true;
    params.Doublebuffer = true;
    params.Stencilbuffer = false;

    auto device = createDeviceEx(params);

    if (!device)
        return 1;

    auto* driver = device->getVideoDriver();
    auto* am = device->getAssetManager();

    asset::IAssetLoader::SAssetLoadParams lparams;
    lparams.loaderFlags = asset::IAssetLoader::E_LOADER_PARAMETER_FLAGS::ELPF_LOAD_METADATA_ONLY;

    auto assetLoaded = device->getAssetManager()->getAsset("../../media/mitsuba/aniso_ies/028e97564391140b1476695ae7a46fa4.ies", lparams);
    const auto* meta = assetLoaded.getMetadata();

    if (!meta)
        return 2;

    const auto* iesProfileMeta = meta->selfCast<const asset::CIESProfileMetadata>();
    IESCompute iesComputeEnvironment(driver, am, iesProfileMeta->profile);

    // temporary, to remove later
    {
        auto iOld = iesProfileMeta->profile.getIntegral();
        iesProfileMeta->profile.createCDCTexture();
        auto iNew = iesProfileMeta->profile.getIntegralFromGrid();
        const auto error = std::abs((iOld - iNew) / iOld);
        printf("integral error: %s", std::to_string(error).c_str());
    }
    
    IESExampleEventReceiver receiver;
    device->setEventReceiver(&receiver);
        
    while (device->run() && receiver.isRunning())
    {
        iesComputeEnvironment.updateZDegree(receiver.getZDegreeOffset());
        iesComputeEnvironment.updateMode(receiver.getMode());

        iesComputeEnvironment.begin();
        iesComputeEnvironment.dispatch();
        iesComputeEnvironment.renderpass();
        iesComputeEnvironment.end();

        std::wostringstream windowCaption;
        {
            const wchar_t* const mode = [&]()
            {
                switch (iesComputeEnvironment.getMode())
                {
                    case IESCompute::EM_CDC:
                        return L"CDC";
                    case IESCompute::EM_IES_C:
                        return L"IES Candela";
                    case IESCompute::EM_SPERICAL_C:
                        return L"Spherical Coordinates";
                    case IESCompute::EM_DIRECTION:
                        return L"Direction sample";
                    case IESCompute::EM_PASS_T_MASK:
                        return L"Pass Mask";
                    default:
                        return L"ERROR";
                }
            }();
            windowCaption << L"IES Demo - Nabla Engine - Degrees: : " << iesComputeEnvironment.getZDegree() << L" - Mode: " << mode;
            device->setWindowCaption(windowCaption.str());
        }
        receiver.reset();
    }

    return 0;
}