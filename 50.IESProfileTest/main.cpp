// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define BENCHMARK_TILL_FIRST_FRAME

#include <nabla.h>
#include <chrono>
#include <filesystem>
#include "nbl/ext/ScreenShot/ScreenShot.h"
#include "compute/common.h"
#include <stdio.h>

// small hack to compile with the json library
namespace std 
{
    int sprintf_s(char* buffer, size_t size, const char* format, ...) {
        va_list args;
        va_start(args, format);
        int result = ::sprintf_s(buffer, size, format, args);
        va_end(args);
        return result;
    }
}

#include "nlohmann/json.hpp"

using namespace nbl;
using namespace core;
using json = nlohmann::json;

#ifdef BENCHMARK_TILL_FIRST_FRAME
const std::chrono::steady_clock::time_point startBenchmark = std::chrono::high_resolution_clock::now();
bool stopBenchamrkFlag = false;
#endif

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

            const auto xGroups = (getActiveProfile().getOptimalIESResolution().x - 1u) / WORKGROUP_DIMENSION + 1u;
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

            #ifdef BENCHMARK_TILL_FIRST_FRAME
            if (!stopBenchamrkFlag)
            {
                const std::chrono::steady_clock::time_point stopBenchmark = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stopBenchmark - startBenchmark);
                std::cout << "Time taken till first render pass: " << duration.count() << " milliseconds" << std::endl;
                stopBenchamrkFlag = true;
            }
            #endif
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
                updateCDescriptorSets(); // flush descriptor set
                updateGDescriptorSets(); // flush descriptor set

                const auto& profile = getActiveProfile();
                pushConstant.maxIValue = (float)profile.getMaxCandelaValue();
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
            createGPUDescriptors();
            const auto initIdx = generalPurposeOffset;

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
                const std::vector<IGPUDescriptorSetLayout::SBinding> bindings = getGBindings();
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

                gpue.sampler = createSampler();

                for (auto i = 0; i < gpue.NBL_D_IMAGES_AMOUNT; i++)
                {
                    gpue.gwrites[i].dstSet = gpue.gDescriptorSet.get();
                    gpue.gwrites[i].binding = i;
                    gpue.gwrites[i].count = 1u;
                    gpue.gwrites[i].arrayElement = 0u;
                    gpue.gwrites[i].descriptorType = asset::EDT_COMBINED_IMAGE_SAMPLER;
                    gpue.gwrites[i].info = gpue.ginfos + i;
                }

                updateGDescriptorSets();
            }
        }

        void createGPUDescriptors()
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

                const auto optimalResolution = profile.getOptimalIESResolution();

                cssbod.dImageIESC = std::move(createGPUImageView<asset::EF_R16_UNORM>(optimalResolution.x, optimalResolution.y));
                cssbod.dImageS = std::move(createGPUImageView<asset::EF_R32G32_SFLOAT>(optimalResolution.x, optimalResolution.y));
                cssbod.dImageD = std::move(createGPUImageView<asset::EF_R32G32B32A32_SFLOAT>(optimalResolution.x, optimalResolution.y));
                cssbod.dImageTMask = std::move(createGPUImageView<asset::EF_R8G8_UNORM>(optimalResolution.x, optimalResolution.y));
            }
        }

        void updateCDescriptorSets()
        {
            fillImageDescriptorInfo<EB_IMAGE_IES_C>(generalPurposeOffset, m_gpue.cinfos[EB_IMAGE_IES_C]);
            fillImageDescriptorInfo<EB_IMAGE_S>(generalPurposeOffset, m_gpue.cinfos[EB_IMAGE_S]);
            fillImageDescriptorInfo<EB_IMAGE_D>(generalPurposeOffset, m_gpue.cinfos[EB_IMAGE_D]);
            fillImageDescriptorInfo<EB_IMAGE_T_MASK>(generalPurposeOffset, m_gpue.cinfos[EB_IMAGE_T_MASK]);

            fillSSBODescriptorInfo<EB_SSBO_HA>(generalPurposeOffset, m_gpue.cinfos[EB_SSBO_HA]);
            fillSSBODescriptorInfo<EB_SSBO_VA>(generalPurposeOffset, m_gpue.cinfos[EB_SSBO_VA]);
            fillSSBODescriptorInfo<EB_SSBO_D>(generalPurposeOffset, m_gpue.cinfos[EB_SSBO_D]);

            const core::smart_refctd_ptr<const video::IGPUDescriptorSetLayout> proxy(m_gpue.cPipeline->getLayout()->getDescriptorSetLayout(0));
            m_gpue.cDescriptorSet = core::smart_refctd_ptr(driver->createGPUDescriptorSet(core::smart_refctd_ptr(proxy)));

            for (auto i = 0; i < EB_SIZE; i++)
                m_gpue.cwrites[i].dstSet = m_gpue.cDescriptorSet.get();

            driver->updateDescriptorSets(EB_SIZE, m_gpue.cwrites, 0u, nullptr);
        }

        void updateGDescriptorSets()
        {
            fillImageDescriptorInfo<EB_IMAGE_IES_C>(generalPurposeOffset, m_gpue.ginfos[EB_IMAGE_IES_C]);
            fillImageDescriptorInfo<EB_IMAGE_S>(generalPurposeOffset, m_gpue.ginfos[EB_IMAGE_S]);
            fillImageDescriptorInfo<EB_IMAGE_D>(generalPurposeOffset, m_gpue.ginfos[EB_IMAGE_D]);
            fillImageDescriptorInfo<EB_IMAGE_T_MASK>(generalPurposeOffset, m_gpue.ginfos[EB_IMAGE_T_MASK]);

            const core::smart_refctd_ptr<const video::IGPUDescriptorSetLayout> proxy(m_gpue.gPipeline->getLayout()->getDescriptorSetLayout(3));
            m_gpue.gDescriptorSet = core::smart_refctd_ptr(driver->createGPUDescriptorSet(core::smart_refctd_ptr(proxy)));

            for (auto i = 0; i < m_gpue.NBL_D_IMAGES_AMOUNT; i++)
                m_gpue.gwrites[i].dstSet = m_gpue.gDescriptorSet.get();

            driver->updateDescriptorSets(m_gpue.NBL_D_IMAGES_AMOUNT, m_gpue.gwrites, 0u, nullptr);
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

        template<E_BINDINGS binding>
        void fillImageDescriptorInfo(const size_t assetIndex, IGPUDescriptorSet::SDescriptorInfo& info)
        {
            static_assert(binding == EB_IMAGE_IES_C || binding == EB_IMAGE_S || binding == EB_IMAGE_D || binding == EB_IMAGE_T_MASK);

            const auto& profile = getProfile(assetIndex);
            auto& cssbod = m_gpue.CSSBOD[assetIndex];

            core::smart_refctd_ptr<video::IGPUImageView> proxy;

            if constexpr (binding == EB_IMAGE_IES_C)
                proxy = core::smart_refctd_ptr(cssbod.dImageIESC);
            else if (binding == EB_IMAGE_S)
                proxy = core::smart_refctd_ptr(cssbod.dImageS);
            else if (binding == EB_IMAGE_D)
                proxy = core::smart_refctd_ptr(cssbod.dImageD);
            else
                proxy = core::smart_refctd_ptr(cssbod.dImageTMask);

            info.desc = core::smart_refctd_ptr(proxy);
            info.image = { core::smart_refctd_ptr(m_gpue.sampler), asset::EIL_SHADER_READ_ONLY_OPTIMAL };
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

        std::vector<IGPUDescriptorSetLayout::SBinding> getGBindings()
        {
            const std::vector<IGPUDescriptorSetLayout::SBinding> bindings =
            {
                { EB_IMAGE_IES_C, asset::EDT_COMBINED_IMAGE_SAMPLER, 1, asset::ISpecializedShader::ESS_FRAGMENT, nullptr },
                { EB_IMAGE_S, asset::EDT_COMBINED_IMAGE_SAMPLER, 1, asset::ISpecializedShader::ESS_FRAGMENT, nullptr },
                { EB_IMAGE_D, asset::EDT_COMBINED_IMAGE_SAMPLER, 1, asset::ISpecializedShader::ESS_FRAGMENT, nullptr },
                { EB_IMAGE_T_MASK, asset::EDT_COMBINED_IMAGE_SAMPLER, 1, asset::ISpecializedShader::ESS_FRAGMENT, nullptr }
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
            _NBL_STATIC_INLINE_CONSTEXPR uint8_t NBL_D_IMAGES_AMOUNT = 4u;

            // Compute
            core::smart_refctd_ptr<video::IGPUComputePipeline> cPipeline;
            core::smart_refctd_ptr<video::IGPUDescriptorSet> cDescriptorSet;

            IGPUDescriptorSet::SDescriptorInfo cinfos[EB_SIZE];
            IGPUDescriptorSet::SWriteDescriptorSet cwrites[EB_SIZE];

            struct CSSBODescriptor
            {
                core::smart_refctd_ptr<video::IGPUBuffer> vAngles, hAngles, data;
                core::smart_refctd_ptr<video::IGPUImageView> dImageIESC, dImageS, dImageD, dImageTMask;
            };

            std::vector<CSSBODescriptor> CSSBOD;

            // Graphics
            core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> gPipeline;
            core::smart_refctd_ptr<video::IGPUDescriptorSet> gDescriptorSet;
            core::smart_refctd_ptr<video::IGPUMeshBuffer> mBuffer;

            IGPUDescriptorSet::SDescriptorInfo ginfos[NBL_D_IMAGES_AMOUNT];
            IGPUDescriptorSet::SWriteDescriptorSet gwrites[NBL_D_IMAGES_AMOUNT];

            // Shared data
            core::smart_refctd_ptr<video::IGPUSampler> sampler;
        } m_gpue;

        #include "nbl/nblpack.h"
        struct PushConstant
        {
            float maxIValue;
            float zAngleDegreeRotation;
            IESCompute::E_MODE mode = IESCompute::EM_CDC;
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

    auto readJSON = [](const std::string& filePath)
    {
        std::ifstream file(filePath.data());
        if (!file.is_open()) {
            printf("Invalid input json \"%s\" file! Aborting..", filePath.data());
            exit(0x45);
        }

        std::stringstream buffer;
        buffer << file.rdbuf();

        return buffer.str();
    };

    const auto INPUT_JSON_FILE_PATH_FS = std::filesystem::absolute("../inputs.json");
    const auto INPUT_JSON_FILE_PATH = INPUT_JSON_FILE_PATH_FS.string();
    const auto jsonBuffer = readJSON(INPUT_JSON_FILE_PATH);
    if (jsonBuffer.empty()) {
        printf("Read input json \"%s\" file is empty! Aborting..\n", INPUT_JSON_FILE_PATH.c_str());
        exit(0x45);
    }

    const auto jsonMap = json::parse(jsonBuffer.c_str());
    
    if (!jsonMap["directories"].is_array())
    {
        printf("Input json \"%s\" file's field \"directories\" is not an array! Aborting..\n", INPUT_JSON_FILE_PATH.c_str());
        exit(0x45);
    }

    if (!jsonMap["files"].is_array())
    {
        printf("Input json \"%s\" file's field \"files\" is not an array! Aborting..\n", INPUT_JSON_FILE_PATH.c_str());
        exit(0x45);
    }

    if (!jsonMap["writeAssets"].is_boolean())
    {
        printf("Input json \"%s\" file's field \"writeAssets\" is not a boolean! Aborting..\n", INPUT_JSON_FILE_PATH.c_str());
        exit(0x45);
    }

    const auto&& IES_INPUTS = [&]()
    {
        std::vector<std::string> inputFilePaths;

        auto addFile = [&inputFilePaths, &INPUT_JSON_FILE_PATH_FS](const std::string_view filePath) -> void
        {
            auto path = std::filesystem::path(filePath);

            if (!path.is_absolute())
                path = std::filesystem::absolute(INPUT_JSON_FILE_PATH_FS.parent_path() / path);

            if (std::filesystem::exists(path) && std::filesystem::is_regular_file(path) && path.extension() == ".ies")
                inputFilePaths.push_back(path.string());
            else
            {
                printf("Invalid input path \"%s\"! Aborting..\n", path.string().c_str());
                exit(0x45);
            }
        };

        auto addFiles = [&inputFilePaths, &INPUT_JSON_FILE_PATH_FS, &addFile](const std::string_view directoryPath) -> void
        {
            auto directory(std::filesystem::absolute(INPUT_JSON_FILE_PATH_FS.parent_path() / directoryPath));
            if (!std::filesystem::exists(directory) || !std::filesystem::is_directory(directory)) {
                printf("Invalid input directory \"%s\"! Aborting..\n", directoryPath.data());
                exit(0x45);
            }

            for (const auto& entry : std::filesystem::directory_iterator(directory))
                addFile(entry.path().string().c_str());
        };

        // parse json
        {
            std::vector<std::string_view> jDirectories;
            jsonMap["directories"].get_to(jDirectories);

            for (const auto& it : jDirectories)
                addFiles(it);

            std::vector<std::string_view> jFiles;
            jsonMap["files"].get_to(jFiles);

            for (const auto& it : jFiles)
                addFile(it);
        }

        return std::move(inputFilePaths);
    }();

    const bool GUI = [&]()
    {
        bool b = false;
        jsonMap["gui"].get_to(b);

        return b;
    }();

    const bool WRITE_ASSETS = [&]()
    {
        bool b = false;
        jsonMap["writeAssets"].get_to(b);

        return b;
    }();
   
    const auto ASSETS = [&]()
    {
        size_t loaded = {}, total = IES_INPUTS.size();
        std::vector<asset::SAssetBundle> assets;
        std::vector<std::string> outStems;
            
        for (size_t i = 0; i < total; ++i)
        {
            auto asset = device->getAssetManager()->getAsset(IES_INPUTS[i].c_str(), lparams);
            const auto* path = IES_INPUTS[i].c_str();
            const auto stem = std::filesystem::path(IES_INPUTS[i].c_str()).stem().string();

            if (asset.getMetadata())
            {
                assets.emplace_back(std::move(asset));
                outStems.push_back(stem);
                ++loaded;
            }
            else
                printf("Could not load metadata from \"%s\" asset! Skipping..\n", path);
        }
        printf("Loaded [%s/%s] assets! Status: %s\n", std::to_string(loaded).c_str(), std::to_string(total).c_str(), loaded == total ? "PASSING" : "FAILING");

        return std::make_pair(assets, outStems);
    }();

    if (GUI)
        printf("GUI Mode: ON\n");
    else
    {
        printf("GUI Mode: OFF\nExiting...");
        exit(0);
    }

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

    if(WRITE_ASSETS)
        for (size_t i = 0; i < ASSETS.first.size(); ++i)
        {
            const auto& bundle = ASSETS.first[i];
            const auto& stem = ASSETS.second[i];

            const auto& profile = bundle.getMetadata()->selfCast<const asset::CIESProfileMetadata>()->profile;
            // const std::string out = std::filesystem::absolute("out/cpu/" + std::string(getProfileRS(profile)) + "/" + stem + ".png").string(); TODO (?): why its not working? ah touch required probably first
            const std::string out = std::filesystem::absolute(std::string(getProfileRS(profile)) + "_" + stem + ".png").string();

            asset::IAssetWriter::SAssetWriteParams wparams(bundle.getContents().begin()->get());

            if (am->writeAsset(out.c_str(), wparams))
                printf("Saved \"%s\"\n", out.c_str());
            else
                printf("Could not write \"%s\"\n", out.c_str());
        }

    return 0;
}