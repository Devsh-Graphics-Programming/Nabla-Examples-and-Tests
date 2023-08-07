// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <nabla.h>
#include "nbl/ext/ScreenShot/ScreenShot.h"

using namespace nbl;
using namespace core;

int main()
{
    // create device with full flexibility over creation parameters
    // you can add more parameters if desired, check nbl::SIrrlichtCreationParameters
    nbl::SIrrlichtCreationParameters params;
    params.Bits = 24; //may have to set to 32bit for some platforms
    params.ZBufferBits = 24; //we'd like 32bit here
    params.DriverType = video::EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
    params.WindowSize = dimension2d<uint32_t>(640, 640);
    params.Fullscreen = false;
    params.Vsync = true; //! If supported by target platform
    params.Doublebuffer = true;
    params.Stencilbuffer = false; //! This will not even be a choice soon
    auto device = createDeviceEx(params);

    if (!device)
        return 1; // could not create selected driver.

    auto* driver = device->getVideoDriver();
    auto* smgr = device->getSceneManager();
    auto* am = device->getAssetManager();
    auto* fs = am->getFileSystem();
    auto* glslc = am->getGLSLCompiler();
    auto* gc = am->getGeometryCreator();

    struct SPushConsts
    {
        core::matrix4SIMD VP;
        core::vectorSIMDf campos;
    };

    core::smart_refctd_ptr<video::IGPUPipelineLayout> layout;
    {
        asset::SPushConstantRange rng[2];
        rng[0].offset = 0u;
        rng[0].size = sizeof(SPushConsts::VP);
        rng[0].stageFlags = asset::ISpecializedShader::ESS_VERTEX;
        rng[1].offset = offsetof(SPushConsts,campos);
        rng[1].size = sizeof(SPushConsts::campos);
        rng[1].stageFlags = asset::ISpecializedShader::ESS_FRAGMENT;

        layout = driver->createGPUPipelineLayout(rng, rng+2);
    }

    asset::IAssetLoader::SAssetLoadParams lparams;
    auto assetLoaded = device->getAssetManager()->getAsset("../test.ies", lparams);
    auto assetCand = *assetLoaded.getContents().begin();
    auto cpuImage = core::smart_refctd_ptr_static_cast<asset::ICPUImage>(std::move(assetCand));

    core::smart_refctd_ptr<video::IGPUMeshBuffer> quad;
    core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline> pipeline;
    {
        auto dat = gc->createRectangleMesh(vector2df_SIMD(1.0, 1.0));
        
        auto cpusphere = core::make_smart_refctd_ptr<asset::ICPUMeshBuffer>(nullptr, nullptr, dat.bindings, std::move(dat.indexBuffer));
        cpusphere->setBoundingBox(dat.bbox);
        cpusphere->setIndexType(dat.indexType);
        cpusphere->setIndexCount(dat.indexCount);

        io::IReadFile* file = fs->createAndOpenFile("../shader.vert");
        auto cpuvs = glslc->resolveIncludeDirectives(file, asset::ISpecializedShader::ESS_VERTEX, "../shader.vert");
        auto vs = driver->createGPUShader(std::move(cpuvs));
        file->drop();
        asset::ISpecializedShader::SInfo vsinfo(nullptr, nullptr, "main", asset::ISpecializedShader::ESS_VERTEX, "../shader.vert");
        auto vs_spec = driver->createGPUSpecializedShader(vs.get(), vsinfo);

        io::IReadFile* file2 = fs->createAndOpenFile("../shader.frag");
        auto cpufs = glslc->resolveIncludeDirectives(file2, asset::ISpecializedShader::ESS_FRAGMENT, "../shader.frag");
        auto fs = driver->createGPUShader(std::move(cpufs));
        file2->drop();
        asset::ISpecializedShader::SInfo fsinfo(nullptr, nullptr, "main", asset::ISpecializedShader::ESS_FRAGMENT, "../shader.frag");
        auto fs_spec = driver->createGPUSpecializedShader(fs.get(), fsinfo);

        video::IGPUSpecializedShader* shaders[2]{ vs_spec.get(),fs_spec.get()};
        asset::SRasterizationParams raster;
        pipeline = driver->createGPURenderpassIndependentPipeline(nullptr, core::smart_refctd_ptr(layout), shaders, shaders + 2, dat.inputParams, asset::SBlendParams{}, dat.assemblyParams, raster);

        quad = driver->getGPUObjectsFromAssets(&cpusphere.get(), &cpusphere.get()+1)->front();
    }

    //! we want to move around the scene and view it from different angles
    scene::ICameraSceneNode* camera = smgr->addCameraSceneNodeFPS(nullptr , 100.0f, 0.005f);

    video::IGPUImage::SCreationParams imgInfo;
    imgInfo.format = asset::EF_R16G16B16A16_SFLOAT;
    imgInfo.type = asset::ICPUImage::ET_2D;
    imgInfo.extent.width = driver->getScreenSize().Width;
    imgInfo.extent.height = driver->getScreenSize().Height;
    imgInfo.extent.depth = 1u;
    imgInfo.mipLevels = 1u;
    imgInfo.arrayLayers = 1u;
    imgInfo.samples = asset::ICPUImage::ESCF_1_BIT;
    imgInfo.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);

    auto image = driver->createGPUImageOnDedMem(std::move(imgInfo), driver->getDeviceLocalGPUMemoryReqs());
    const auto texelFormatBytesize = getTexelOrBlockBytesize(image->getCreationParameters().format);

    video::IGPUImageView::SCreationParams imgViewInfo;
    imgViewInfo.format = image->getCreationParameters().format;
    imgViewInfo.image = std::move(image);
    imgViewInfo.viewType = asset::IImageView<video::IGPUImage>::ET_2D;
    imgViewInfo.flags = static_cast<video::IGPUImageView::E_CREATE_FLAGS>(0u);
    imgViewInfo.subresourceRange.baseArrayLayer = 0u;
    imgViewInfo.subresourceRange.baseMipLevel = 0u;
    imgViewInfo.subresourceRange.layerCount = imgInfo.arrayLayers;
    imgViewInfo.subresourceRange.levelCount = imgInfo.mipLevels;

    auto imageView = driver->createGPUImageView(std::move(imgViewInfo));

    auto* fbo = driver->addFrameBuffer();
    fbo->attach(video::EFAP_COLOR_ATTACHMENT0, core::smart_refctd_ptr(imageView));

    SPushConsts pc;
    uint32_t ssNum = 0u;
    while (device->run())
    {
        driver->setRenderTarget(fbo);
        const float clear[4] {0.f,0.f,0.f,1.f};
        driver->clearColorBuffer(video::EFAP_COLOR_ATTACHMENT0, clear);
        driver->beginScene(true, false, video::SColor(255, 0, 0, 0));

        driver->bindGraphicsPipeline(pipeline.get()); 
        driver->drawMeshBuffer(quad.get());

        driver->blitRenderTargets(fbo, nullptr, false, false);

        driver->endScene();
    }


    return 0;
}