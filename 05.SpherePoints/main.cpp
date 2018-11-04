/* GCC compile Flags
-flto
-fuse-linker-plugin
-fno-omit-frame-pointer //for debug
-msse3
-mfpmath=sse
-ggdb3 //for debug
*/
/* Linker Flags
-lIrrlicht
-lXrandr
-lGL
-lX11
-lpthread
-ldl

-fuse-ld=gold
-flto
-fuse-linker-plugin
-msse3
*/
#include <irrlicht.h>
#include "../source/Irrlicht/COpenGLExtensionHandler.h"

/**
This example shows how to:
1) Set up and Use a Simple Shader
2) render triangle buffers to screen in all the different ways
**/
using namespace irr;
using namespace core;



class SimpleCallBack : public video::IShaderConstantSetCallBack
{
    int32_t mvpUniformLocation;
    video::E_SHADER_CONSTANT_TYPE mvpUniformType;
public:
    SimpleCallBack() : mvpUniformLocation(-1), mvpUniformType(video::ESCT_FLOAT_VEC3) {}

    virtual void PostLink(video::IMaterialRendererServices* services, const video::E_MATERIAL_TYPE& materialType, const core::vector<video::SConstantLocationNamePair>& constants)
    {
        //! Normally we'd iterate through the array and check our actual constant names before mapping them to locations but oh well
        mvpUniformLocation = constants[0].location;
        mvpUniformType = constants[0].type;
    }

    virtual void OnSetConstants(video::IMaterialRendererServices* services, int32_t userData)
    {
        services->setShaderConstant(services->getVideoDriver()->getTransform(video::EPTS_PROJ_VIEW_WORLD).pointer(),mvpUniformLocation,mvpUniformType,1);
    }

    virtual void OnUnsetMaterial() {}
};


/*
The start of the main function starts like in most other example. We ask the
user for the desired renderer and start it up.
*/
int main()
{
	// create device with full flexibility over creation parameters
	// you can add more parameters if desired, check irr::SIrrlichtCreationParameters
	irr::SIrrlichtCreationParameters params;
	params.Bits = 24; //may have to set to 32bit for some platforms
	params.ZBufferBits = 24; //we'd like 32bit here
	params.DriverType = video::EDT_OPENGL; //! Only Well functioning driver, software renderer left for sake of 2D image drawing
	params.WindowSize = dimension2d<uint32_t>(1920, 1080);
	params.Fullscreen = false;
	params.Vsync = true; //! If supported by target platform
	params.Doublebuffer = true;
	params.Stencilbuffer = false; //! This will not even be a choice soon
	IrrlichtDevice* device = createDeviceEx(params);

	if (device == 0)
		return 1; // could not create selected driver.


	video::IVideoDriver* driver = device->getVideoDriver();
	SimpleCallBack* callBack = new SimpleCallBack();

    //! First need to make a material other than default to be able to draw with custom shader
    video::SMaterial material;
    //material.BackfaceCulling = false; //! Triangles will be visible from both sides
    material.MaterialType = (video::E_MATERIAL_TYPE)driver->getGPUProgrammingServices()->addHighLevelShaderMaterialFromFiles("../points.vert",
                                                        "","","", //! No Geometry or Tessellation Shaders
                                                        "../points.frag",
                                                        3,video::EMT_SOLID, //! 3 vertices per primitive (this is tessellation shader relevant only
                                                        callBack, //! No Shader Callback (we dont have any constants/uniforms to pass to the shader)
                                                        0); //! No custom user data
    callBack->drop();



	scene::ISceneManager* smgr = device->getSceneManager();


    scene::IGPUMeshBuffer* mb = new scene::IGPUMeshBuffer();
    scene::IGPUMeshDataFormatDesc* desc = driver->createGPUMeshDataFormatDesc();
    mb->setMeshDataAndFormat(desc);
    desc->drop();

    size_t xComps = 0x1u<<9;
    size_t yComps = 0x1u<<9;
    size_t zComps = 0x1u<<9;
    size_t verts = xComps*yComps*zComps;
    uint32_t bufSize = verts*sizeof(uint32_t);
    uint32_t* mem = (uint32_t*)malloc(bufSize);
    for (size_t i=0; i<xComps; i++)
    for (size_t j=0; j<yComps; j++)
    for (size_t k=0; k<zComps; k++)
    {
        mem[i+xComps*(j+yComps*k)] = (i<<20)|(j<<10)|(k);
    }

    video::IGPUBuffer* positionBuf = driver->createDeviceLocalGPUBufferOnDedMem(bufSize);
    //! Buffer we want to upload is too big, so staging buffer must be used multiple times to upload the full data in parts
    auto upStreamBuff = driver->getDefaultUpStreamingBuffer();
    for (uint32_t uploadedSize=0; uploadedSize<bufSize;)
    {
        // the offset upload range by how much has already been uploaded
        const void* dataPtr = reinterpret_cast<uint8_t*>(mem)+uploadedSize;
        //without offset initialized to invalid_address multi_alloc/multi_place will ignore the request for allocation for that particular element
        uint32_t offset = video::StreamingTransientDataBufferMT<>::invalid_address;
        uint32_t alignment = sizeof(uint32_t);
        // max_size gives us the largest allocation we can hope to be able to perform
        uint32_t size = core::alignDown(upStreamBuff->max_size(),alignment);
        // multi_place can fail to allocate if the memory has not been freed yet and it times out on the wait (see comment to multi_free)_
        upStreamBuff->multi_place(std::chrono::milliseconds(50u),1u,(const void* const*)&dataPtr,&offset,&size,&alignment);
        // keep trying again
        if (offset==video::StreamingTransientDataBufferMT<>::invalid_address)
            continue;

        // some platforms expose non-coherent host-visible GPU memory, so writes need to be flushed explicitly
        if (upStreamBuff->needsManualFlushOrInvalidate())
            driver->flushMappedMemoryRanges({{upStreamBuff->getBuffer()->getBoundMemory(),offset,size}});
        // after we make sure writes are in GPU memory (visible to GPU) and not still in a cache, we can copy using the GPU to device-only memory
        driver->copyBuffer(upStreamBuff->getBuffer(),positionBuf,offset,uploadedSize,size);
        // this doesn't actually free the memory, the memory is queued up to be freed only after the GPU fence/event is signalled
        upStreamBuff->multi_free(1u,&offset,&size,driver->placeFence());
        //try upload the next chunk
        uploadedSize += size;
    }
    //positionBuf->updateSubRange(video::IDriverMemoryAllocation::MemoryRange(0,reqs.vulkanReqs.size),mem);
    free(mem);


    //! By mapping we increase/grab() ref counter of positionBuf, any previously mapped buffer will have it's reference dropped
    desc->mapVertexAttrBuffer(positionBuf,
                            scene::EVAI_ATTR0, //! we use first attribute slot (out of a minimum of 16)
                            scene::ECPA_FOUR, //! there are 3 components per vertex
                            scene::ECT_INT_2_10_10_10_REV); //! and they are floats

    /** Since we mapped the buffer, the MeshBuffers will be using it.
        If we drop it, it will be automatically deleted when MeshBuffers are done using it.
    **/
    positionBuf->drop();


    mb->setIndexCount(verts);
    mb->setPrimitiveType(scene::EPT_POINTS);

    scene::ICameraSceneNode* camera = smgr->addCameraSceneNodeFPS(0,80.f,0.001f);
    smgr->setActiveCamera(camera);
    camera->setNearValue(0.001f);
    camera->setFarValue(10.f);

	uint64_t lastFPSTime = 0;

	while(device->run())
	if (device->isWindowActive())
	{
		driver->beginScene(true, true, video::SColor(255,0,0,255) );

        smgr->drawAll();

        driver->setTransform(video::E4X3TS_WORLD,core::matrix4x3());
        driver->setMaterial(material);
        //! draw back to front
        driver->drawMeshBuffer(mb);

		driver->endScene();

		// display frames per second in window title
		uint64_t time = device->getTimer()->getRealTime();
		if (time-lastFPSTime > 1000)
		{
			std::wostringstream str;
			str << L"Sphere Points - Irrlicht Engine  FPS:" << driver->getFPS() << " PrimitvesDrawn:" << driver->getPrimitiveCountDrawn();

			device->setWindowCaption(str.str().c_str());
			lastFPSTime = time;
		}
	}
	mb->drop();

    //create a screenshot
	video::IImage* screenshot = driver->createImage(video::ECF_A8R8G8B8,params.WindowSize);
    glReadPixels(0,0, params.WindowSize.Width,params.WindowSize.Height, GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, screenshot->getData());
    {
        // images are horizontally flipped, so we have to fix that here.
        uint8_t* pixels = (uint8_t*)screenshot->getData();

        const int32_t pitch=screenshot->getPitch();
        uint8_t* p2 = pixels + (params.WindowSize.Height - 1) * pitch;
        uint8_t* tmpBuffer = new uint8_t[pitch];
        for (uint32_t i=0; i < params.WindowSize.Height; i += 2)
        {
            memcpy(tmpBuffer, pixels, pitch);
            memcpy(pixels, p2, pitch);
            memcpy(p2, tmpBuffer, pitch);
            pixels += pitch;
            p2 -= pitch;
        }
        delete [] tmpBuffer;
    }
	driver->writeImageToFile(screenshot,"./screenshot.png");
	screenshot->drop();

	device->drop();

	return 0;
}
