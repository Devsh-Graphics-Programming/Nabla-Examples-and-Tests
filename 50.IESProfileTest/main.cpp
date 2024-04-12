// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include <nabla.h>
#include <chrono>
#include <filesystem>
#include "nbl/ext/ScreenShot/ScreenShot.h"
#include "compute/common.h"

using namespace nbl;
using namespace core;

class IESCompute
{
    public:
        IESCompute(video::IVideoDriver* _driver, asset::IAssetManager* _assetManager, const std::vector<asset::SAssetBundle>& _assets)
            : assets(_assets), driver(_driver), generalPurposeOffset(0), pushConstant({(float)getProfile(0).getMaxCandelaValue(), 0.f})
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

        const asset::CIESProfile& getProfile(const size_t& assetIndex)
        {
            return assets[assetIndex].getMetadata()->selfCast<const asset::CIESProfileMetadata>()->profile;
        }

        const asset::CIESProfile& getActiveProfile()
        {
            return getProfile(generalPurposeOffset);
        }

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
            const auto& profile = getProfile(generalPurposeOffset);
            const auto newDegreeRotation = std::clamp<float>(pushConstant.zAngleDegreeRotation + degreeOffset, profile.getHoriAngles().front(), profile.getHoriAngles().back());
            pushConstant.zAngleDegreeRotation = newDegreeRotation;
        }

        void updateGeneralPurposeOffset(const int8_t& offset)
        {
            const auto newOffset = std::clamp<size_t>(int64_t(generalPurposeOffset) + int64_t(core::sign(offset)), int64_t(0), int64_t(assets.size() - 1));

            if (newOffset != generalPurposeOffset)
            {
                generalPurposeOffset = newOffset;

                // not elegant way to do it here but lets leave it as it is
                updateCDescriptorSets();
                pushConstant.maxIValueReciprocal = (float)getActiveProfile().getMaxCandelaValue();
            }
        }

        const asset::CIESProfile::IES_STORAGE_FORMAT getZDegree()
        {
            const auto& profile = getProfile(generalPurposeOffset);
            return pushConstant.zAngleDegreeRotation + (profile.getSymmetry() == asset::CIESProfile::OTHER_HALF_SYMMETRIC ? 90.0 : 0.0); // real IES horizontal angle has 90.0 degress offset if OTHER_HALF_SYMMETRY, we handle it because of legacy IES 1995 specification case
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

            createSSBOBuffers();

            // Compute
            {
                const std::vector<IGPUDescriptorSetLayout::SBinding> bindings = getCBindings();

                {
                    auto descriptorSetLayout = driver->createGPUDescriptorSetLayout(bindings.data(), bindings.data() + bindings.size());
                    asset::SPushConstantRange range = { asset::ISpecializedShader::ESS_COMPUTE, 0u, sizeof(PushConstant) };

                    gpue.cPipeline = driver->createGPUComputePipeline(nullptr, driver->createGPUPipelineLayout(&range, &range + 1u, core::smart_refctd_ptr(descriptorSetLayout)), gpuSpecializedShaderFromFile("../compute/cdc.comp"));
                    gpue.cDescriptorSet = driver->createGPUDescriptorSet(std::move(descriptorSetLayout));
                }

                {
                    {
                        {
                            gpue.cinfos[EB_IMAGE_IES_C].desc = core::smart_refctd_ptr(gpue.dImageIESC);
                            gpue.cinfos[EB_IMAGE_IES_C].image = { nullptr, asset::EIL_GENERAL };

                            gpue.cinfos[EB_IMAGE_S].desc = core::smart_refctd_ptr(gpue.dImageS);
                            gpue.cinfos[EB_IMAGE_S].image = { nullptr, asset::EIL_GENERAL };

                            gpue.cinfos[EB_IMAGE_D].desc = core::smart_refctd_ptr(gpue.dImageD);
                            gpue.cinfos[EB_IMAGE_D].image = { nullptr, asset::EIL_GENERAL };

                            gpue.cinfos[EB_IMAGE_T_MASK].desc = core::smart_refctd_ptr(gpue.dImageTMask);
                            gpue.cinfos[EB_IMAGE_T_MASK].image = { nullptr, asset::EIL_GENERAL };
                        }
                    }

                    for (auto i = 0; i < EB_SIZE; i++)
                    {
                        gpue.cwrites[i].dstSet = gpue.cDescriptorSet.get();
                        gpue.cwrites[i].binding = i;
                        gpue.cwrites[i].arrayElement = 0u;
                        gpue.cwrites[i].count = 1u;
                        gpue.cwrites[i].info = &gpue.cinfos[i];
                    }

                    gpue.cwrites[EB_IMAGE_IES_C].descriptorType = asset::EDT_STORAGE_IMAGE;
                    gpue.cwrites[EB_IMAGE_S].descriptorType = asset::EDT_STORAGE_IMAGE;
                    gpue.cwrites[EB_IMAGE_D].descriptorType = asset::EDT_STORAGE_IMAGE;
                    gpue.cwrites[EB_IMAGE_T_MASK].descriptorType = asset::EDT_STORAGE_IMAGE;
                    gpue.cwrites[EB_SSBO_HA].descriptorType = asset::EDT_STORAGE_BUFFER;
                    gpue.cwrites[EB_SSBO_VA].descriptorType = asset::EDT_STORAGE_BUFFER;
                    gpue.cwrites[EB_SSBO_D].descriptorType = asset::EDT_STORAGE_BUFFER;

                    updateCDescriptorSets();
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

        void createSSBOBuffers()
        {
            auto createCPUBuffer = [&](const auto& pInput)
            {
                core::smart_refctd_ptr<asset::ICPUBuffer> buffer = core::make_smart_refctd_ptr<asset::ICPUBuffer>(sizeof(asset::CIESProfile::IES_STORAGE_FORMAT) * pInput.size());
                memcpy(buffer->getPointer(), pInput.data(), buffer->getSize());

                return buffer;
            };

            for(size_t i = 0; i < assets.size(); ++i)
            {
                const auto& profile = getProfile(i);
                auto& cssbod = m_gpue.CSSBOD.emplace_back();

                auto createGPUBuffer = [&](const auto& cpuBuffer)
                {
                    return driver->createFilledDeviceLocalGPUBufferOnDedMem(cpuBuffer->getSize(), cpuBuffer->getPointer());
                };

                cssbod.hAngles = createGPUBuffer(createCPUBuffer(profile.getHoriAngles()));
                cssbod.vAngles = createGPUBuffer(createCPUBuffer(profile.getVertAngles()));
                cssbod.data = createGPUBuffer(createCPUBuffer(profile.getData()));
            }
        }

        void updateCDescriptorSets()
        {
            fillSSBODescriptorInfo<EB_SSBO_HA>(generalPurposeOffset, m_gpue.cinfos[EB_SSBO_HA]);
            fillSSBODescriptorInfo<EB_SSBO_VA>(generalPurposeOffset, m_gpue.cinfos[EB_SSBO_VA]);
            fillSSBODescriptorInfo<EB_SSBO_D>(generalPurposeOffset, m_gpue.cinfos[EB_SSBO_D]);

            const std::vector<IGPUDescriptorSetLayout::SBinding> bindings = getCBindings();
            {
                auto descriptorSetLayout = driver->createGPUDescriptorSetLayout(bindings.data(), bindings.data() + bindings.size());
                asset::SPushConstantRange range = { asset::ISpecializedShader::ESS_COMPUTE, 0u, sizeof(PushConstant) };
                m_gpue.cDescriptorSet = driver->createGPUDescriptorSet(std::move(descriptorSetLayout)); // I guess it can be done better
            }

            const core::smart_refctd_ptr<const video::IGPUDescriptorSetLayout> proxy(m_gpue.cPipeline->getLayout()->getDescriptorSetLayout(0));
            m_gpue.cDescriptorSet = core::smart_refctd_ptr(driver->createGPUDescriptorSet(core::smart_refctd_ptr(proxy)));

            for (auto i = 0; i < EB_SIZE; i++)
                m_gpue.cwrites[i].dstSet = m_gpue.cDescriptorSet.get();

            driver->updateDescriptorSets(EB_SIZE, m_gpue.cwrites, 0u, nullptr);
        }

        template<E_BINDINGS binding>
        void fillSSBODescriptorInfo(const size_t assetIndex, IGPUDescriptorSet::SDescriptorInfo& info)
        {
            static_assert(binding == EB_SSBO_HA || binding == EB_SSBO_VA || binding == EB_SSBO_D);

            const auto& profile = getProfile(assetIndex);
            auto& cssbod = m_gpue.CSSBOD[assetIndex];

            core::smart_refctd_ptr<video::IGPUBuffer> proxy;

            if constexpr (binding == EB_SSBO_HA)
                proxy = core::smart_refctd_ptr(cssbod.hAngles);
            else if (binding == EB_SSBO_VA)
                proxy = core::smart_refctd_ptr(cssbod.vAngles);
            else
                proxy = core::smart_refctd_ptr(cssbod.data);

            info.desc = core::smart_refctd_ptr(proxy);
            info.buffer = { 0, proxy->getSize() };
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

        std::vector<IGPUDescriptorSetLayout::SBinding> getCBindings()
        {
            std::vector<IGPUDescriptorSetLayout::SBinding> bindings = 
            {
                { EB_IMAGE_IES_C, asset::EDT_STORAGE_IMAGE, 1, asset::ISpecializedShader::ESS_COMPUTE, nullptr },
                { EB_IMAGE_S, asset::EDT_STORAGE_IMAGE, 1, asset::ISpecializedShader::ESS_COMPUTE, nullptr },
                { EB_IMAGE_D, asset::EDT_STORAGE_IMAGE, 1, asset::ISpecializedShader::ESS_COMPUTE, nullptr },
                { EB_IMAGE_T_MASK, asset::EDT_STORAGE_IMAGE, 1, asset::ISpecializedShader::ESS_COMPUTE, nullptr },
                { EB_SSBO_HA, asset::EDT_STORAGE_BUFFER, 1, asset::ISpecializedShader::ESS_COMPUTE, nullptr },
                { EB_SSBO_VA, asset::EDT_STORAGE_BUFFER, 1, asset::ISpecializedShader::ESS_COMPUTE, nullptr },
                { EB_SSBO_D, asset::EDT_STORAGE_BUFFER, 1, asset::ISpecializedShader::ESS_COMPUTE, nullptr }
            };

            return bindings;
        }

        template<asset::E_FORMAT format>
        video::IFrameBuffer* createFBO(const size_t& width, const size_t& height)
        {
            auto* fbo = driver->addFrameBuffer();

            bBuffer = createGPUImageView<format>(width, height);
            fbo->attach(video::EFAP_COLOR_ATTACHMENT0, core::smart_refctd_ptr(bBuffer));
          
            return fbo;
        }

        const std::vector<asset::SAssetBundle> assets;
        video::IVideoDriver* const driver;

        struct GPUE
        {
            // Compute
            core::smart_refctd_ptr<video::IGPUComputePipeline> cPipeline;
            core::smart_refctd_ptr<video::IGPUDescriptorSet> cDescriptorSet;

            IGPUDescriptorSet::SDescriptorInfo cinfos[EB_SIZE];
            IGPUDescriptorSet::SWriteDescriptorSet cwrites[EB_SIZE];

            struct CSSBODescriptor
            {
                core::smart_refctd_ptr<video::IGPUBuffer> vAngles, hAngles, data;
            };

            std::vector<CSSBODescriptor> CSSBOD;

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

        size_t generalPurposeOffset = 0;
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
                case nbl::KEY_UP:
                {
                    generalPurposeOffset = 1;
                    return true;
                }
                case nbl::KEY_DOWN:
                {
                    generalPurposeOffset = -1;
                    return true;
                }
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

    void reset() { zDegreeOffset = 0; generalPurposeOffset = 0; }
    inline const auto& isRunning() const { return running; }
    inline const auto& getMode() const { return mode; }
    template<typename T = double>
    inline const auto& getZDegreeOffset() const { return static_cast<T>(zDegreeOffset); }
    inline const auto& getGeneralPurposeOffset() { return generalPurposeOffset; }
private:
    double zDegreeOffset = 0.0;
    int8_t generalPurposeOffset = 0;
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
    lparams.loaderFlags;
   
    constexpr auto IES_INPUTS = std::array
    { 
        std::string_view("../../media/mitsuba/ies/ISOTROPIC/007cfb11e343e2f42e3b476be4ab684e.ies"),
        std::string_view("../../media/mitsuba/ies/ANIISOTROPIC/QUAD_SYMMETRY/0275171fb664c1b3f024d1e442a68d22.ies"),
        std::string_view("../../media/mitsuba/ies/ANIISOTROPIC/HALF_SYMMETRY/1392a1ba55b67d3e0ae7fd63527f3e78.ies"),
        std::string_view("../../media/mitsuba/ies/ANIISOTROPIC/OTHER_HALF_SYMMETRY/028e97564391140b1476695ae7a46fa4.ies"),
        std::string_view("../../media/mitsuba/ies/NO_LATERAL_SYMMET/4b88bf886b39cfa63094e70e1afa680e.ies"),
    };

    const auto ASSETS = [&]()
    {
        std::vector<asset::SAssetBundle> assets;
        std::vector<std::string> outStems;
            
        for (size_t i = 0; i < IES_INPUTS.size(); ++i)
        {
            auto asset = device->getAssetManager()->getAsset(IES_INPUTS[i].data(), lparams);
            const auto stem = std::filesystem::path(IES_INPUTS[i].data()).stem().string();

            if (asset.getMetadata())
            {
                assets.emplace_back(std::move(asset));
                outStems.push_back(stem);
            }
            else
                printf("Could not load metadata from \"%s\" asset! Skipping..", stem.c_str());
        }

        return std::make_pair(assets, outStems);
    }();

    IESCompute iesComputeEnvironment(driver, am, ASSETS.first);    
    IESExampleEventReceiver receiver;
    device->setEventReceiver(&receiver);

    auto getModeRS = [&]()
    {
        switch (iesComputeEnvironment.getMode())
        {
            case IESCompute::EM_CDC:
                return "CDC";
            case IESCompute::EM_IES_C:
                return "IES Candela";
            case IESCompute::EM_SPERICAL_C:
                return "Spherical Coordinates";
            case IESCompute::EM_DIRECTION:
                return "Direction sample";
            case IESCompute::EM_PASS_T_MASK:
                return "Pass Mask";
            default:
                return "ERROR";
        }
    };

    auto getProfileRS = [&](const asset::CIESProfile& profile)
    {            
        switch (profile.getSymmetry())
        {
            case asset::CIESProfile::ISOTROPIC:
                return "ISOTROPIC";
            case asset::CIESProfile::QUAD_SYMETRIC:
                return "QUAD_SYMETRIC";
            case asset::CIESProfile::HALF_SYMETRIC:
                return "HALF_SYMETRIC";
            case asset::CIESProfile::OTHER_HALF_SYMMETRIC:
                return "OTHER_HALF_SYMMETRIC";
            case asset::CIESProfile::NO_LATERAL_SYMMET:
                return "NO_LATERAL_SYMMET";
            default:
                return "ERROR";
        }
    };
        
    while (device->run() && receiver.isRunning())
    {
        iesComputeEnvironment.updateGeneralPurposeOffset(receiver.getGeneralPurposeOffset());
        iesComputeEnvironment.updateZDegree(receiver.getZDegreeOffset());
        iesComputeEnvironment.updateMode(receiver.getMode());

        iesComputeEnvironment.begin();
        iesComputeEnvironment.dispatch();
        iesComputeEnvironment.renderpass();
        iesComputeEnvironment.end();

        std::wostringstream windowCaption;
        {
            const auto* const mode = getModeRS();
            const auto* const profile = getProfileRS(iesComputeEnvironment.getActiveProfile());

            windowCaption << "IES Demo - Nabla Engine - Profile: " << profile << " - Degrees: " << iesComputeEnvironment.getZDegree() << " - Mode: " << mode;
            device->setWindowCaption(windowCaption.str());
        }
        receiver.reset();
    }

    for (size_t i = 0; i < ASSETS.first.size(); ++i)
    {
        const auto& bundle = ASSETS.first[i];
        const auto& stem = ASSETS.second[i];

        const auto& profile = bundle.getMetadata()->selfCast<const asset::CIESProfileMetadata>()->profile;
        // const std::string out = std::filesystem::absolute("out/cpu/" + std::string(getProfileRS(profile)) + "/" + stem + ".png").string(); TODO (?): why its not working?
        const std::string out = std::filesystem::absolute(std::string(getProfileRS(profile)) + "_" + stem + ".png").string();

        asset::IAssetWriter::SAssetWriteParams wparams(bundle.getContents().begin()->get());

        if (am->writeAsset(out.c_str(), wparams))
            printf("Saved \"%s\"\n", out.c_str());
        else
            printf("Could not write \"%s\"\n", out.c_str());
    }

    return 0;
}