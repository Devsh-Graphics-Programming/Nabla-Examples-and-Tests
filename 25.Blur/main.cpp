#define _IRR_STATIC_LIB_
#include <irrlicht.h>
#include "../source/Irrlicht/COpenGLExtensionHandler.h"

#include "../source/Irrlicht/CGeometryCreator.h"
#include "../../ext/Blur/CBlurPerformer.h"

using namespace irr;
using namespace core;


//!Same As Last Example
class MyEventReceiver : public IEventReceiver
{
    const uint32_t BLUR_RADIUS_MIN = 2u, BLUR_RADIUS_MAX = 50u;

public:
    ext::Blur::CBlurPerformer* blurPerf = nullptr;

    bool OnEvent(const SEvent& event)
    {
        if (event.EventType == irr::EET_KEY_INPUT_EVENT && !event.KeyInput.PressedDown)
        {
            switch (event.KeyInput.Key)
            {
            case irr::KEY_KEY_Q: // switch wire frame mode
                exit(0);
                return true;
            default:
                break;
            }
        }
        else if (event.EventType == irr::EET_MOUSE_INPUT_EVENT && blurPerf)
        {
            uint32_t r = blurPerf->getRadius() + uint32_t(event.MouseInput.Wheel);
            blurPerf->setRadius(std::max(BLUR_RADIUS_MIN, std::min(r, BLUR_RADIUS_MAX)));
        }

        return false;
    }

private:
};

class SimpleCallBack : public video::IShaderConstantSetCallBack
{
    int32_t mvpUniformLocation;
    int32_t cameraDirUniformLocation;
    video::E_SHADER_CONSTANT_TYPE mvpUniformType;
    video::E_SHADER_CONSTANT_TYPE cameraDirUniformType;

public:
    SimpleCallBack() : cameraDirUniformLocation(-1), cameraDirUniformType(video::ESCT_FLOAT_VEC3) {}

    virtual void PostLink(video::IMaterialRendererServices* services, const video::E_MATERIAL_TYPE& materialType, const core::array<video::SConstantLocationNamePair>& constants)
    {
        for (size_t i = 0; i<constants.size(); i++)
        {
            if (constants[i].name == "MVP")
            {
                mvpUniformLocation = constants[i].location;
                mvpUniformType = constants[i].type;
            }
            else if (constants[i].name == "cameraPos")
            {
                cameraDirUniformLocation = constants[i].location;
                cameraDirUniformType = constants[i].type;
            }
        }
    }

    virtual void OnSetConstants(video::IMaterialRendererServices* services, int32_t userData)
    {
        core::vectorSIMDf modelSpaceCamPos;
        modelSpaceCamPos.set(services->getVideoDriver()->getTransform(video::E4X3TS_WORLD_VIEW_INVERSE).getTranslation());
        services->setShaderConstant(&modelSpaceCamPos, cameraDirUniformLocation, cameraDirUniformType, 1);
        services->setShaderConstant(services->getVideoDriver()->getTransform(video::EPTS_PROJ_VIEW_WORLD).pointer(), mvpUniformLocation, mvpUniformType, 1);
    }

    virtual void OnUnsetMaterial() {}
};


int main()
{
    // create device with full flexibility over creation parameters
    // you can add more parameters if desired, check irr::SIrrlichtCreationParameters
    irr::SIrrlichtCreationParameters params;
    params.Bits = 24; //may have to set to 32bit for some platforms
    params.ZBufferBits = 24; //we'd like 32bit here
    params.DriverType = video::EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
    params.WindowSize = dimension2d<uint32_t>(1280, 720);
    params.Fullscreen = false;
    params.Vsync = true; //! If supported by target platform
    params.Doublebuffer = true;
    params.Stencilbuffer = false; //! This will not even be a choice soon
    IrrlichtDevice* device = createDeviceEx(params);

    if (device == 0)
        return 1; // could not create selected driver.

    video::IVideoDriver* driver = device->getVideoDriver();

    SimpleCallBack* cb = new SimpleCallBack();
    video::E_MATERIAL_TYPE newMaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../cube.vert",
        "", "", "", //! No Geometry or Tessellation Shaders
        "../cube.frag",
        3, video::EMT_SOLID, //! 3 vertices per primitive (this is tessellation shader relevant only
        cb, //! Our Shader Callback
        0); //! No custom user data
    cb->drop();


    scene::ISceneManager* smgr = device->getSceneManager();
    driver->setTextureCreationFlag(video::ETCF_ALWAYS_32_BIT, true);
    scene::ICameraSceneNode* camera =
        smgr->addCameraSceneNodeFPS(0, 100.0f, 0.01f);
    camera->setPosition(core::vector3df(-4, 0, 0));
    camera->setTarget(core::vector3df(0, 0, 0));
    camera->setNearValue(0.01f);
    camera->setFarValue(100.0f);
    smgr->setActiveCamera(camera);
    device->getCursorControl()->setVisible(false);
    MyEventReceiver receiver;
    device->setEventReceiver(&receiver);


    scene::ICPUMesh* cpumesh = smgr->getGeometryCreator()->createCubeMeshCPU();
    video::ITexture* texture = driver->getTexture("../tex.jpg", video::ECF_A16B16G16R16F);

    const core::vector2d<uint32_t> outputSz{ 1000u, 1000u };
    ext::Blur::CBlurPerformer* blur = ext::Blur::CBlurPerformer::instantiate(driver, 4u, outputSz);
    receiver.blurPerf = blur;

    video::ITexture* outputTex = driver->addTexture(video::ITexture::ETT_2D, &outputSz.X, 1u, "blur_output", video::ECF_A16B16G16R16F);

    cpumesh->getMeshBuffer(0)->getMaterial().TextureLayer[0].SamplingParams.TextureWrapU = video::ETC_CLAMP_TO_EDGE;
    cpumesh->getMeshBuffer(0)->getMaterial().TextureLayer[0].SamplingParams.TextureWrapV = video::ETC_CLAMP_TO_EDGE;

    scene::IGPUMesh* gpumesh = driver->createGPUMeshesFromCPU(std::vector<scene::ICPUMesh*>(1,cpumesh))[0];
    video::SMaterial& mutableMaterial = smgr->addMeshSceneNode(gpumesh)->getMaterial(0);
    mutableMaterial.MaterialType = static_cast<video::E_MATERIAL_TYPE>(newMaterialType);
    mutableMaterial.TextureLayer[0].Texture = outputTex;
    gpumesh->drop();

    cpumesh->getMeshBuffer(0)->getMaterial().setTexture(0, texture);
    gpumesh = driver->createGPUMeshesFromCPU(std::vector<scene::ICPUMesh*>(1,cpumesh))[0];
    smgr->addMeshSceneNode(gpumesh, nullptr, -1, vector3df(10.f, 0.f, 0.f))->setMaterialType(newMaterialType);
    gpumesh->drop();

    smgr->getMeshCache()->removeMesh(cpumesh);

    uint64_t lastFPSTime = 0;

    while (device->run())
    {
        driver->beginScene(true, true, video::SColor(255, 0, 0, 255));

        blur->blurTexture(texture, outputTex);

        smgr->drawAll();

        driver->endScene();

        // display frames per second in window title
        uint64_t time = device->getTimer()->getRealTime();
        if (time - lastFPSTime > 1000)
        {
            std::wostringstream sstr;
            sstr << L"Builtin Nodes Demo - Irrlicht Engine FPS:" << driver->getFPS() << " PrimitvesDrawn:" << driver->getPrimitiveCountDrawn();

            device->setWindowCaption(sstr.str().c_str());
            lastFPSTime = time;
        }
    }

    blur->drop();

    //create a screenshot
    video::IImage* screenshot = driver->createImage(video::ECF_A8R8G8B8, params.WindowSize);
    glReadPixels(0, 0, params.WindowSize.Width, params.WindowSize.Height, GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, screenshot->getData());
    {
        // images are horizontally flipped, so we have to fix that here.
        uint8_t* pixels = (uint8_t*)screenshot->getData();

        const int32_t pitch = screenshot->getPitch();
        uint8_t* p2 = pixels + (params.WindowSize.Height - 1) * pitch;
        uint8_t* tmpBuffer = new uint8_t[pitch];
        for (uint32_t i = 0; i < params.WindowSize.Height; i += 2)
        {
            memcpy(tmpBuffer, pixels, pitch);
            memcpy(pixels, p2, pitch);
            memcpy(p2, tmpBuffer, pitch);
            pixels += pitch;
            p2 -= pitch;
        }
        delete[] tmpBuffer;
    }
    driver->writeImageToFile(screenshot, "./screenshot.png");
    screenshot->drop();
    device->sleep(3000);

    device->drop();

    return 0;
}
