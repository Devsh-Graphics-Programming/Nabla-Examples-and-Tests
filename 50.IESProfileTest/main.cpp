// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include <nabla.h>
#include "nbl/ext/ScreenShot/ScreenShot.h"
#include "compute/common.h"

using namespace nbl;
using namespace core;

class IESExampleEventReceiver : public nbl::IEventReceiver
{
    public:
        IESExampleEventReceiver() {}

        bool OnEvent(const nbl::SEvent& event)
        {
            if (event.EventType == nbl::EET_MOUSE_INPUT_EVENT)
            {
                const auto& newDegree = std::clamp<double>(zDegree + event.MouseInput.Wheel, 0.0, 360.0);

                if (zDegree != newDegree)
                    regenerateCDC = true;

                zDegree = newDegree;

                return true;
            }

            if (event.EventType == nbl::EET_KEY_INPUT_EVENT && !event.KeyInput.PressedDown)
            {
                switch (event.KeyInput.Key)
                {
                    case nbl::KEY_KEY_C:
                    {
                        cdcMode = true;
                        return true;
                    }
                    case nbl::KEY_KEY_R:
                    {
                        cdcMode = false;
                        return true;
                    }
                    case nbl::KEY_KEY_D:
                    {
                        debug = !debug;

                        if(debug)
                            printf("[INFO] Debug mode turned ON, verbose logs will be generated to stdout");
                        else
                            printf("[INFO] Debug mode turned OFF, verbose logs will be stopped");

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

        inline const auto& isRunning() const { return running; }
        inline const auto& isInCDCMode() const { return cdcMode; }
        inline const auto& isDebug() const { return debug; }
        inline const auto& needToRegenerateCDC() const { return regenerateCDC; }
        template<typename T = double>
        inline const auto& getZDegree() const { return static_cast<T>(zDegree); }
        inline void resetRequests() { regenerateCDC = false; }

    private:
        _NBL_STATIC_INLINE_CONSTEXPR size_t DEGREE_SHIFT = 5u;
        double zDegree = 0.0;
        bool cdcMode = true, running = true, debug = false, regenerateCDC = false;
};

class IESCompute
{
    public:
        IESCompute(video::IVideoDriver* _driver, asset::IAssetManager* _assetManager, const asset::CIESProfile& _profile)
            : profile(_profile), driver(_driver), pushConstant({ (float)profile.getMaxValue(), 0.0 })
        {
            createGPUEnvironment<EM_CDC>(_assetManager);
            // createGPUEnvironment<EM_RENDER>(_assetManager); // TODO

            fbo = createFBO<asset::EF_R16G16B16A16_SFLOAT>(driver->getScreenSize().Width, driver->getScreenSize().Height);
        }
        ~IESCompute() {}

        enum E_MODE
        {
            EM_CDC,     //! Candlepower Distribution Curve
            EM_RENDER,  //! 3D render of an IES light
            EM_SIZE
        };

        enum E_BINDINGS
        {
            EB_IMAGE,    //! Image with data depending on E_MODE
            EB_SSBO_HA,  //! IES Profile SSBO Horizontal Angles 
            EB_SSBO_VA,  //! IES Profile SSBO Vertical Angles
            EB_SSBO_D,   //! IES Profile SSBO Data
            EB_SIZE
        };

        void begin()
        {
            driver->setRenderTarget(fbo);
            const float clear[4]{ 0.f,0.f,0.f,1.f };
            driver->clearColorBuffer(video::EFAP_COLOR_ATTACHMENT0, clear);
            driver->beginScene(true, false, video::SColor(255, 0, 0, 0));
        }

        template<E_MODE mode>
        void dispatch()
        {
            static_assert(mode != EM_SIZE);
            auto& gpue = getGPUE<mode>();

            driver->bindComputePipeline(gpue.cPipeline.get());
            driver->bindDescriptorSets(EPBP_COMPUTE, gpue.cPipeline->getLayout(), 0u, 1u, &gpue.cDescriptorSet.get(), nullptr);
            driver->pushConstants(gpue.cPipeline->getLayout(), asset::ISpecializedShader::ESS_COMPUTE, 0u, sizeof(PushConstant), &pushConstant);

            _NBL_STATIC_INLINE_CONSTEXPR auto xGroups = (TEXTURE_SIZE - 1u) / WORKGROUP_DIMENSION + 1u;
            driver->dispatch(xGroups, xGroups, 1u);

            COpenGLExtensionHandler::extGlMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        }

        template<E_MODE mode>
        void renderpass()
        {
            static_assert(mode != EM_SIZE);
            auto& gpue = getGPUE<mode>();

            driver->bindGraphicsPipeline(gpue.gPipeline.get());
            driver->bindDescriptorSets(video::EPBP_GRAPHICS, gpue.gPipeline->getLayout(), 3u, 1u, &gpue.gDescriptorSet.get(), nullptr);
            driver->drawMeshBuffer(gpue.mBuffer.get());
        }

        void end()
        {
            driver->blitRenderTargets(fbo, nullptr, false, false);
            driver->endScene();
        }

        void updateZDegree(const asset::CIESProfile::IES_STORAGE_FORMAT& degree)
        {
            pushConstant.zAngleDegreeRotation = degree;
        }

        const auto& getZDegree()
        {
            return pushConstant.zAngleDegreeRotation;
        }

    private:

        template<E_MODE _mode>
        void createGPUEnvironment(asset::IAssetManager* _assetManager)
        {
            static_assert(_mode != EM_SIZE);

            auto gpuSpecializedShaderFromFile = [&](const char* path)
            {
                auto bundle = _assetManager->getAsset(path, {});
                auto shader = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*bundle.getContents().begin());

                return driver->getGPUObjectsFromAssets<asset::ICPUSpecializedShader>(&shader, &shader + 1u)->operator[](0); // omg
            };

            if (_mode == EM_SIZE)
                return;

            auto& gpue = m_gpue[_mode];

            gpue.dImageV = std::move(createGPUImageView<asset::EF_R16_UNORM>(TEXTURE_SIZE, TEXTURE_SIZE));

            // Compute
            {
                const std::vector<IGPUDescriptorSetLayout::SBinding> bindings =
                {
                    { EB_IMAGE, asset::EDT_STORAGE_IMAGE, 1, asset::ISpecializedShader::ESS_COMPUTE, nullptr },
                    { EB_SSBO_HA, asset::EDT_STORAGE_BUFFER, 1, asset::ISpecializedShader::ESS_COMPUTE, nullptr },
                    { EB_SSBO_VA, asset::EDT_STORAGE_BUFFER, 1, asset::ISpecializedShader::ESS_COMPUTE, nullptr },
                    { EB_SSBO_D, asset::EDT_STORAGE_BUFFER, 1, asset::ISpecializedShader::ESS_COMPUTE, nullptr }
                };

                {
                    auto descriptorSetLayout = driver->createGPUDescriptorSetLayout(bindings.data(), bindings.data() + bindings.size());
                    asset::SPushConstantRange range = { asset::ISpecializedShader::ESS_COMPUTE, 0u, sizeof(PushConstant) };

                    gpue.cPipeline = driver->createGPUComputePipeline(nullptr, driver->createGPUPipelineLayout(&range, &range + 1u, core::smart_refctd_ptr(descriptorSetLayout)), gpuSpecializedShaderFromFile(getShaderPath<_mode>().data()));
                    gpue.cDescriptorSet = driver->createGPUDescriptorSet(std::move(descriptorSetLayout));
                }

                {
                    IGPUDescriptorSet::SDescriptorInfo infos[EB_SIZE];
                    {
                        createSSBODescriptorInfo<EB_SSBO_HA>(profile, infos[EB_SSBO_HA]);
                        createSSBODescriptorInfo<EB_SSBO_VA>(profile, infos[EB_SSBO_VA]);
                        createSSBODescriptorInfo<EB_SSBO_D>(profile, infos[EB_SSBO_D]);

                        {
                            infos[EB_IMAGE].desc = core::smart_refctd_ptr(gpue.dImageV);
                            infos[EB_IMAGE].image = { nullptr, asset::EIL_GENERAL };
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

                    writes[EB_IMAGE].descriptorType = asset::EDT_STORAGE_IMAGE;
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
                    { EB_IMAGE, asset::EDT_COMBINED_IMAGE_SAMPLER, 1, asset::ISpecializedShader::ESS_FRAGMENT, nullptr }
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

                    gpue.gPipeline = driver->createGPURenderpassIndependentPipeline(nullptr, driver->createGPUPipelineLayout(nullptr, nullptr, nullptr, nullptr, nullptr, core::smart_refctd_ptr(descriptorSetLayout)), shaders, shaders + 2, mesh.inputParams, asset::SBlendParams{}, mesh.assemblyParams, raster);
                    gpue.gDescriptorSet = driver->createGPUDescriptorSet(core::smart_refctd_ptr(descriptorSetLayout));
                    gpue.mBuffer = driver->getGPUObjectsFromAssets(&cpusphere.get(), &cpusphere.get() + 1)->front();
                }
                
                auto sampler = driver->createGPUSampler({ asset::ISampler::ETC_CLAMP_TO_EDGE,asset::ISampler::ETC_CLAMP_TO_EDGE,asset::ISampler::ETC_CLAMP_TO_EDGE,asset::ISampler::ETBC_FLOAT_OPAQUE_BLACK,asset::ISampler::ETF_LINEAR,asset::ISampler::ETF_LINEAR,asset::ISampler::ESMM_LINEAR,0u,false,asset::ECO_ALWAYS });

                video::IGPUDescriptorSet::SWriteDescriptorSet write;
                {
                    write.dstSet = gpue.gDescriptorSet.get();
                    write.binding = EB_IMAGE;
                    write.count = 1u;
                    write.arrayElement = 0u;
                    write.descriptorType = asset::EDT_COMBINED_IMAGE_SAMPLER;
                }
                
                IGPUDescriptorSet::SDescriptorInfo info;
                {
                    info.desc = core::smart_refctd_ptr(gpue.dImageV);
                    info.image = { core::smart_refctd_ptr(sampler),asset::EIL_SHADER_READ_ONLY_OPTIMAL };
                }

                write.info = &info;
                driver->updateDescriptorSets(1u, &write, 0u, nullptr);   
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

        template<E_MODE mode>
        _NBL_STATIC_INLINE_CONSTEXPR std::string_view getShaderPath()
        {
            if constexpr (mode == EM_CDC)
                return "../compute/cdc.comp";
            else if (mode == EM_RENDER)
                return "../compute/render.comp";
            else
                return "";

            static_assert(mode != EM_SIZE);
        }

        template<E_MODE mode>
        auto& getGPUE()
        {
            static_assert(mode != EM_SIZE);
            return m_gpue[mode];
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
            core::smart_refctd_ptr<video::IGPUImageView> dImageV;
        } m_gpue[EM_SIZE];

        #include "nbl/nblpack.h"
        struct PushConstant
        {
            float maxIValueReciprocal;
            float zAngleDegreeRotation;
        } PACK_STRUCT;
        #include "nbl/nblunpack.h"
        
        PushConstant pushConstant;

        video::IFrameBuffer* fbo = nullptr;
        core::smart_refctd_ptr<video::IGPUImageView> bBuffer;
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

    auto assetLoaded = device->getAssetManager()->getAsset("../028e97564391140b1476695ae7a46fa4.ies", lparams);
    const auto* meta = assetLoaded.getMetadata();

    if (!meta)
        return 2;

    const auto* iesProfileMeta = meta->selfCast<const asset::CIESProfileMetadata>();
    IESCompute iesComputeEnvironment(driver, am, iesProfileMeta->profile);

    IESExampleEventReceiver receiver;
    device->setEventReceiver(&receiver);
        
    while (device->run() && receiver.isRunning())
    {
        iesComputeEnvironment.updateZDegree(receiver.getZDegree());

        iesComputeEnvironment.begin();
        iesComputeEnvironment.dispatch<IESCompute::EM_CDC>();
        iesComputeEnvironment.renderpass<IESCompute::EM_CDC>();
        iesComputeEnvironment.end();

        std::wostringstream windowCaption;
        {
            const wchar_t* const mode = receiver.isInCDCMode() ? L"CDC" : L"3D Render";
            windowCaption << L"IES Demo - Nabla Engine - Degrees: : " << receiver.getZDegree() << L" - Mode: " << mode;
            device->setWindowCaption(windowCaption.str());
        } receiver.resetRequests();
    }

    return 0;
}