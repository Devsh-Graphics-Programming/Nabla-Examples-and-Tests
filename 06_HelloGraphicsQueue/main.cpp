// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


// I've moved out a tiny part of this example into a shared header for reuse, please open and read it.
#include "../common/MonoDeviceApplication.hpp"
#include "../common/MonoAssetManagerAndBuiltinResourceApplication.hpp"

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;


// Here we showcase the use of Graphics Queue only 
class HelloGraphicsQueueApp final : public examples::MonoDeviceApplication, public examples::MonoAssetManagerAndBuiltinResourceApplication
{
		using device_base_t = examples::MonoDeviceApplication;
		using asset_base_t = examples::MonoAssetManagerAndBuiltinResourceApplication;

	public:
		// Yay thanks to multiple inheritance we cannot forward ctors anymore
		HelloGraphicsQueueApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
			system::IApplicationFramework(_localInputCWD,_localOutputCWD,_sharedInputCWD,_sharedOutputCWD) {}

		// This time we will load images and compute their histograms and output them as CSV
		bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			// Remember to call the base class initialization!
			if (!device_base_t::onAppInitialized(std::move(system)))
				return false;
			if (!asset_base_t::onAppInitialized(std::move(system)))
				return false;

			core::smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
			if (!m_device->createCommandBuffers(nullptr,IGPUCommandBuffer::EL_PRIMARY,1,&cmdbuf))
				return false;

			// TODO: for later
			// - create two EF_R8G8B8A8_SRGB images of 512^2 and 256^2 size
			// - create a buffer large enough to back the output 256^2 image
			// - clear one to red the other to blue
			// - blit the blue in the middle of the red
			// - blit the red back to blue
			// - save image to disk
			// all without using IUtilities

			cmdbuf->begin();
			// clear the image to a given colour
			cmdbuf->clearColorImage();
			// now blit it into the center of the other one
			cmdbuf->blitImage(,NEAREST);
			// blit it again to force downsampling with linear filtering
			cmdbuf->blitImage(,LINEAR);
			// copy the resulting image to a buffer
			cmdbuf->copyImageToBuffer();
			cmdbuf->end();

			// read the buffer back, create an ICPUImage with an adopted buffer over its contents
			// save the image to PNG,JPG and whatever other non-EXR format we support writing

			return true;
		}

		//
		void workLoopBody() override {}

		//
		bool keepRunning() override {return false;}

	protected:
		core::vector<queue_req_t> getQueueRequirements() const override
		{
			return {{.requiredFlags=flags_t::EQF_GRAPHICS_BIT,.disallowedFlags=flags_t::EQF_NONE,.queueCount=1,.maxImageTransferGranularity={1,1,1}}};
		}
};


NBL_MAIN_FUNC(HelloGraphicsQueueApp)