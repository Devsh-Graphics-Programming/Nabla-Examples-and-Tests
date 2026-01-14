// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/examples/examples.hpp"
#include "nbl/video/utilities/CAssetConverter.h"

using namespace nbl;
using namespace core;
using namespace hlsl;
using namespace system;
using namespace asset;
using namespace ui;
using namespace video;
using namespace scene;
using namespace nbl::examples;

// instead of defining our own `int main()` we derive from `nbl::system::IApplicationFramework` to play "nice" wil all platofmrs
class EnvmapImportanceSamplingApp final : public MonoWindowApplication, public BuiltinResourcesApplication
{
	using device_base_t = MonoWindowApplication;
	using asset_base_t = BuiltinResourcesApplication;
public:

  inline EnvmapImportanceSamplingApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) 
    : IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD),
    device_base_t({1280,720}, EF_UNKNOWN, _localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

	bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		if (!asset_base_t::onAppInitialized(smart_refctd_ptr(system)))
			return false;
		if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
			return false;

		m_semaphore = m_device->createSemaphore(m_realFrameIx);
		if (!m_semaphore)
			return logFail("Failed to Create a Semaphore!");

		auto pool = m_device->createCommandPool(getGraphicsQueue()->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
		for (auto i = 0u; i<MaxFramesInFlight; i++)
		{
			if (!pool)
				return logFail("Couldn't create Command Pool!");
			if (!pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,{m_cmdBufs.data()+i,1}))
				return logFail("Couldn't create Command Buffer!");
		}

		constexpr std::string_view defaultImagePath = "../../media/envmap/envmap_0.exr";

		const auto targetFilePath = [&]() -> std::string_view
		{
			const auto argc = argv.size();
			const bool isDefaultImageRequested = argc == 1;

			if (isDefaultImageRequested)
			{
				m_logger->log("No image specified, loading default \"%s\" OpenEXR image from media directory!", ILogger::ELL_INFO, defaultImagePath.data());
				return defaultImagePath;
			}
			else if (argc == 2)
			{
				const std::string_view target(argv[1]);
				m_logger->log("Requested \"%s\"", ILogger::ELL_INFO, target.data());
				return { target };
			}
			else
			{
				m_logger->log("To many arguments! Pass a single filename to an OpenEXR image w.r.t CWD.", ILogger::ELL_ERROR);
				return {};
			}
		}();

		if (targetFilePath.empty())
			return false;
			
		auto assetManager = make_smart_refctd_ptr<nbl::asset::IAssetManager>(smart_refctd_ptr(m_system));

		nbl::asset::IAssetLoader::SAssetLoadParams lp;
		const asset::COpenEXRMetadata* meta;

		auto image_bundle = assetManager->getAsset(targetFilePath.data(), lp);
		auto contents = image_bundle.getContents();
		{
			if (contents.empty())
			{
				m_logger->log("Could not load \"%s\"", ILogger::ELL_ERROR, targetFilePath.data());
				return false;
			}

			meta = image_bundle.getMetadata()->selfCast<const COpenEXRMetadata>();

			if (!meta)
			{
				m_logger->log("Could not selfCast \"%s\" asset's metadata to COpenEXRMetadata, the tool expects valid OpenEXR input image, terminating!", ILogger::ELL_ERROR, targetFilePath.data());
				return false;
			}
		}

		auto asset = contents[0];
		auto image = IAsset::castDown<ICPUImage>(asset);
		const auto* metadata = static_cast<const COpenEXRMetadata::CImage*>(meta->getAssetSpecificMetadata(image.get()));

		ICPUImageView::SCreationParams imgViewParams;
		imgViewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
		imgViewParams.image = std::move(image);
		imgViewParams.format = imgViewParams.image->getCreationParameters().format;
		imgViewParams.viewType = ICPUImageView::ET_2D;
		imgViewParams.subresourceRange = { static_cast<IImage::E_ASPECT_FLAGS>(0u),0u,1u,0u,1u };

		auto imageView = ICPUImageView::create(std::move(imgViewParams));
		auto channelsName = metadata->m_name;


		auto converter = CAssetConverter::create( { .device=m_device.get() });
		
		{
			// Test the provision of a custom patch this time
			CAssetConverter::patch_t<ICPUImageView> patch(imageView.get(),IImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT);

			// We don't want to generate mip-maps for these images (YET), to ensure that we must override the default callbacks.
			struct SInputs final : CAssetConverter::SInputs
			{
				inline uint8_t getMipLevelCount(const size_t groupCopyID, const ICPUImage* image, const CAssetConverter::patch_t<asset::ICPUImage>& patch) const override
				{
					return image->getCreationParameters().mipLevels;
				}
				inline uint16_t needToRecomputeMips(const size_t groupCopyID, const ICPUImage* image, const CAssetConverter::patch_t<asset::ICPUImage>& patch) const override
				{
					return 0b0u;
				}
			} inputs = {};
			std::get<CAssetConverter::SInputs::asset_span_t<ICPUImageView>>(inputs.assets) = { &imageView.get(),1 };
			std::get<CAssetConverter::SInputs::patch_span_t<ICPUImageView>>(inputs.patches) = { &patch,1 };
			inputs.logger = m_logger.get();

			//
			auto reservation = converter->reserve(inputs);

			// get the created image view
			auto gpuView = reservation.getGPUObjects<ICPUImageView>().front().value;
			if (!gpuView)
				return false;
			gpuView->getCreationParameters().image->setObjectDebugName("envmap");
		}

		return true;
	}

	protected:
		const video::IGPURenderpass::SCreationParams::SSubpassDependency* getDefaultSubpassDependencies() const override
		{
			// Subsequent submits don't wait for each other, but they wait for acquire and get waited on by present
			const static IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] = {
				// don't want any writes to be available, we'll clear, only thing to worry about is the layout transition
				{
					.srcSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
					.dstSubpass = 0,
					.memoryBarrier = {
						.srcStageMask = PIPELINE_STAGE_FLAGS::NONE, // should sync against the semaphore wait anyway 
						.srcAccessMask = ACCESS_FLAGS::NONE,
						// layout transition needs to finish before the color write
						.dstStageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
						.dstAccessMask = ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
					}
					// leave view offsets and flags default
				},
				// want layout transition to begin after all color output is done
				{
					.srcSubpass = 0,
					.dstSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
					.memoryBarrier = {
						// last place where the color can get modified, depth is implicitly earlier
						.srcStageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
						// only write ops, reads can't be made available
						.srcAccessMask = ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
						// spec says nothing is needed when presentation is the destination
					}
					// leave view offsets and flags default
				},
				IGPURenderpass::SCreationParams::DependenciesEnd
			};
			return dependencies;
		}

		inline IQueue::SSubmitInfo::SSemaphoreInfo renderFrame(const std::chrono::microseconds nextPresentationTimestamp) override
		{

			const auto resourceIx = m_realFrameIx % MaxFramesInFlight;
			auto* const cb = m_cmdBufs.data()[resourceIx].get();
			cb->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
			cb->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			cb->end();
			IQueue::SSubmitInfo::SSemaphoreInfo retval =
			{
				.semaphore = m_semaphore.get(),
				.value = ++m_realFrameIx,
				.stageMask = PIPELINE_STAGE_FLAGS::ALL_GRAPHICS_BITS
			};
			const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[] =
			{
				{.cmdbuf = cb }
			};
			const IQueue::SSubmitInfo::SSemaphoreInfo acquired[] = {
				{
					.semaphore = device_base_t::getCurrentAcquire().semaphore,
					.value = device_base_t::getCurrentAcquire().acquireCount,
					.stageMask = PIPELINE_STAGE_FLAGS::NONE
				}
			};
			const IQueue::SSubmitInfo infos[] =
			{
				{
					.waitSemaphores = acquired,
					.commandBuffers = commandBuffers,
					.signalSemaphores = {&retval,1}
				}
			};
			
			if (getGraphicsQueue()->submit(infos) != IQueue::RESULT::SUCCESS)
			{
				retval.semaphore = nullptr; // so that we don't wait on semaphore that will never signal
				m_realFrameIx--;
			}


			m_window->setCaption("[Nabla Engine] UI App Test Demo");
			return retval;
		}

	smart_refctd_ptr<ISemaphore> m_semaphore;
	uint64_t m_realFrameIx = 0;
	std::array<smart_refctd_ptr<IGPUCommandBuffer>,MaxFramesInFlight> m_cmdBufs;
};

NBL_MAIN_FUNC(EnvmapImportanceSamplingApp)
