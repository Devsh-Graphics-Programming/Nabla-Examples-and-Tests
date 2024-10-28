// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"
#include "SimpleWindowedApplication.hpp"

#include "nbl/video/surface/CSurfaceVulkan.h"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"

#include "nlohmann/json.hpp"
#include "argparse/argparse.hpp"

using json = nlohmann::json;

using namespace nbl;
using namespace core;
using namespace hlsl;
using namespace system;
using namespace asset;
using namespace ui;
using namespace video;

// defines for sampler tests can be found in the file below
#include "app_resources/push_constants.hlsl"

class ColorSpaceTestSampleApp final : public examples::SimpleWindowedApplication, public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
		using device_base_t = examples::SimpleWindowedApplication;
		using asset_base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;
		using clock_t = std::chrono::steady_clock;
		using perf_clock_resolution_t = std::chrono::milliseconds;

		constexpr static inline clock_t::duration DisplayImageDuration = std::chrono::milliseconds(900);
		constexpr static inline std::string_view DefaultImagePathsFile = "../imagesTestList.txt";

	public:
		// Yay thanks to multiple inheritance we cannot forward ctors anymore
		inline ColorSpaceTestSampleApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
			IApplicationFramework(_localInputCWD,_localOutputCWD,_sharedInputCWD,_sharedOutputCWD) {}
		
		// Will get called mid-initialization, via `filterDevices` between when the API Connection is created and Physical Device is chosen
		inline core::vector<video::SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const override
		{
			// So let's create our Window and Surface then!
			if (!m_surface)
			{
				{
					IWindow::SCreationParams params = {};
					params.callback = core::make_smart_refctd_ptr<nbl::video::ISimpleManagedSurface::ICallback>();
					params.width = 256;
					params.height = 256;
					params.x = 32;
					params.y = 32;
					// Don't want to have a window lingering about before we're ready so create it hidden.
					// Only programmatic resize, not regular.
					params.flags = ui::IWindow::ECF_HIDDEN|IWindow::ECF_BORDERLESS|IWindow::ECF_RESIZABLE;
					params.windowCaption = "ColorSpaceTestSampleApp";
					const_cast<std::remove_const_t<decltype(m_window)>&>(m_window) = m_winMgr->createWindow(std::move(params));
				}
				auto surface = CSurfaceVulkanWin32::create(smart_refctd_ptr(m_api),smart_refctd_ptr_static_cast<IWindowWin32>(m_window));
				const_cast<std::remove_const_t<decltype(m_surface)>&>(m_surface) = nbl::video::CSimpleResizeSurface<nbl::video::CDefaultSwapchainFramebuffers>::create(std::move(surface));
			}
			if (m_surface)
				return {{m_surface->getSurface()/*,EQF_NONE*/}};
			return {};
		}
		
		inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			argparse::ArgumentParser program("Color Space");

			program.add_argument("--verbose")
				.default_value(false)
				.implicit_value(true)
				.help("Print detailed logs.");

			program.add_argument("--test")
				.help("Perform tests for given mode.");

			program.add_argument("--input-list")
				.help("File path to override input list with image file paths to execute this program with.");

			program.add_argument("--update-references")
				.default_value(false)
				.implicit_value(true)
				.help("Update test result references with current test result data.");

			try
			{
				program.parse_args({ argv.data(), argv.data() + argv.size() });
			}
			catch (const std::exception& err)
			{
				std::cerr << err.what() << std::endl << program;
				return 1;
			}

			options.verbose = program.get<bool>("--verbose");
			{
				const auto test = program.present("--test");

				if (test)
				{
					options.tests.enabled = true;
					options.tests.mode = *test;
					options.tests.updateReferences = program.get<bool>("--update-references");
				}
			}

			if (!options.tests.enabled)
			{
				// Remember to call the base class initialization!
				if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
					return false;
			}

			if (!asset_base_t::onAppInitialized(std::move(system)))
				return false;

			if (options.tests.enabled)
			{
				// validate, do not move before asset_base_t::onAppInitialized
				if (options.tests.mode != "hash")
				{
					logFail("Invalid test mode \"%s\", current only \"hash\" is supported for this example!", options.tests.mode.c_str());
					exit(0x45);
				}
			}

			// get custom input list of files to execute the program with
			system::path m_loadCWD = DefaultImagePathsFile;
			{
				const auto hook = program.present("--input-list");

				if (hook)
				{
					const auto inputList = *hook;

					m_testPathsFile = std::ifstream(inputList);
					if (m_testPathsFile.is_open())
						m_loadCWD = inputList;
					else
						m_logger->log("Couldn't open test file given by argument --input-list \"%s\", falling back to default list!", ILogger::ELL_ERROR, inputList.c_str());
				}
			}

			if (!m_testPathsFile.is_open())
				m_testPathsFile = std::ifstream(m_loadCWD);

			if (!m_testPathsFile.is_open())
				return logFail("Could not open the test paths file");

			m_logger->log("Connected \"%s\" input test list!", ILogger::ELL_INFO, m_loadCWD.string().c_str());
			m_loadCWD = m_loadCWD.parent_path();

			if (!options.tests.enabled)
			{
				// Load FSTri Shader
				ext::FullScreenTriangle::ProtoPipeline fsTriProtoPPln(m_assetMgr.get(), m_device.get(), m_logger.get());
				if (!fsTriProtoPPln)
					return logFail("Failed to create Full Screen Triangle protopipeline or load its vertex shader!");

				// Load Custom Shader
				auto loadCompileAndCreateShader = [&](const std::string& relPath) -> smart_refctd_ptr<IGPUShader>
					{
						IAssetLoader::SAssetLoadParams lp = {};
						lp.logger = m_logger.get();
						lp.workingDirectory = ""; // virtual root
						auto assetBundle = m_assetMgr->getAsset(relPath, lp);
						const auto assets = assetBundle.getContents();
						if (assets.empty())
							return nullptr;

						// lets go straight from ICPUSpecializedShader to IGPUSpecializedShader
						auto source = IAsset::castDown<ICPUShader>(assets[0]);
						if (!source)
							return nullptr;

						return m_device->createShader(source.get());
					};
				auto fragmentShader = loadCompileAndCreateShader("app_resources/present.frag.hlsl");
				if (!fragmentShader)
					return logFail("Failed to Load and Compile Fragment Shader!");

				// Now surface indep resources
				m_semaphore = m_device->createSemaphore(m_submitIx);
				if (!m_semaphore)
					return logFail("Failed to Create a Semaphore!");

				/*
				* We'll be using a combined image sampler for this example, which lets us assign both a sampled image and a sampler to the same binding.
				* In this example we provide a sampler at descriptor set creation time, via the SBinding struct below. This specifies that the sampler for this binding is immutable,
				* as evidenced by the name of the field in the SBinding.
				* Samplers for combined image samplers can also be mutable, which for a binding of a descriptor set is specified also at creation time by leaving the immutableSamplers
				* field set to its default (nullptr).
				*/
				smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout;
				{
					auto defaultSampler = m_device->createSampler({
						.AnisotropicFilter = 0
						});

					const IGPUDescriptorSetLayout::SBinding bindings[1] = { {
						.binding = 0,
						.type = IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
						.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
						.stageFlags = IShader::E_SHADER_STAGE::ESS_FRAGMENT,
						.count = 1,
						.immutableSamplers = &defaultSampler
					}
					};
					dsLayout = m_device->createDescriptorSetLayout(bindings);
					if (!dsLayout)
						return logFail("Failed to Create Descriptor Layout");

				}

				ISwapchain::SCreationParams swapchainParams = { .surface = m_surface->getSurface() };
				// Need to choose a surface format
				if (!swapchainParams.deduceFormat(m_physicalDevice))
					return logFail("Could not choose a Surface Format for the Swapchain!");
				// We actually need external dependencies to ensure ordering of the Implicit Layout Transitions relative to the semaphore signals
				const static IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] = {
					// wipe-transition to ATTACHMENT_OPTIMAL
					{
						.srcSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
						.dstSubpass = 0,
						.memoryBarrier = {
						// since we're uploading the image data we're about to draw 
						.srcStageMask = asset::PIPELINE_STAGE_FLAGS::COPY_BIT,
						.srcAccessMask = asset::ACCESS_FLAGS::TRANSFER_WRITE_BIT,
						.dstStageMask = asset::PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
						// because we clear and don't blend
						.dstAccessMask = asset::ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
					}
					// leave view offsets and flags default
				},
					// ATTACHMENT_OPTIMAL to PRESENT_SRC
					{
						.srcSubpass = 0,
						.dstSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
						.memoryBarrier = {
							.srcStageMask = asset::PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
							.srcAccessMask = asset::ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
							// we can have NONE as the Destinations because the spec says so about presents
						}
					// leave view offsets and flags default
				},
				IGPURenderpass::SCreationParams::DependenciesEnd
				};
				auto scResources = std::make_unique<CDefaultSwapchainFramebuffers>(m_device.get(), swapchainParams.surfaceFormat.format, dependencies);
				if (!scResources->getRenderpass())
					return logFail("Failed to create Renderpass!");

				// Now create the pipeline
				{
					const asset::SPushConstantRange range = {
						.stageFlags = IShader::E_SHADER_STAGE::ESS_FRAGMENT,
						.offset = 0,
						.size = sizeof(push_constants_t)
					};
					auto layout = m_device->createPipelineLayout({ &range,1 }, nullptr, nullptr, nullptr, core::smart_refctd_ptr(dsLayout));
					const IGPUShader::SSpecInfo fragSpec = {
						.entryPoint = "main",
						.shader = fragmentShader.get()
					};
					m_pipeline = fsTriProtoPPln.createPipeline(fragSpec, layout.get(), scResources->getRenderpass()/*,default is subpass 0*/);
					if (!m_pipeline)
						return logFail("Could not create Graphics Pipeline!");
				}

				auto queue = getGraphicsQueue();
				// Let's just use the same queue since there's no need for async present
				if (!m_surface || !m_surface->init(queue, std::move(scResources), swapchainParams.sharedParams))
					return logFail("Could not create Window & Surface or initialize the Surface!");
				m_maxFramesInFlight = m_surface->getMaxFramesInFlight();

				// create the descriptor sets, 1 per FIF and with enough room for one image sampler
				{
					const uint32_t setCount = m_maxFramesInFlight;
					auto pool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::E_CREATE_FLAGS::ECF_NONE, { &dsLayout.get(),1 }, &setCount);
					if (!pool)
						return logFail("Failed to Create Descriptor Pool");

					for (auto i = 0u; i < m_maxFramesInFlight; i++)
					{
						m_descriptorSets[i] = pool->createDescriptorSet(core::smart_refctd_ptr(dsLayout));
						if (!m_descriptorSets[i])
							return logFail("Could not create Descriptor Set!");
					}
				}

				// need resetttable commandbuffers for the upload utility
				m_cmdPool = m_device->createCommandPool(queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
				// create the commandbuffers
				for (auto i = 0u; i < m_maxFramesInFlight; i++)
				{
					if (!m_cmdPool)
						return logFail("Couldn't create Command Pool!");
					if (!m_cmdPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { m_cmdBufs.data() + i,1 }))
						return logFail("Couldn't create Command Buffer!");
				}

				// things for IUtilities
				m_scratchSemaphore = m_device->createSemaphore(0);
				if (!m_scratchSemaphore)
					return logFail("Could not create Scratch Semaphore");
				m_scratchSemaphore->setObjectDebugName("Scratch Semaphore");
				// we don't want to overcomplicate the example with multi-queue
				m_intendedSubmit.queue = queue;
				// wait for nothing before upload
				m_intendedSubmit.waitSemaphores = {};
				m_intendedSubmit.prevCommandBuffers = {};
				// fill later
				m_intendedSubmit.scratchCommandBuffers = {};
				m_intendedSubmit.scratchSemaphore = {
					.semaphore = m_scratchSemaphore.get(),
					.value = 0,
					.stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
				};

				// Allocate and Leave 1/4 for image uploads, to test image copy with small memory remaining 
				{
					uint32_t localOffset = video::StreamingTransientDataBufferMT<>::invalid_value;
					uint32_t maxFreeBlock = m_utils->getDefaultUpStreamingBuffer()->max_size();
					const uint32_t allocationAlignment = 64u;
					const uint32_t allocationSize = (maxFreeBlock / 4) * 3;
					m_utils->getDefaultUpStreamingBuffer()->multi_allocate(std::chrono::steady_clock::now() + std::chrono::microseconds(500u), 1u, &localOffset, &allocationSize, &allocationAlignment);
				}
			}

			return true;
		}

		// We do a very simple thing, display an image and wait `DisplayImageMs` to show it
		inline void workLoopBody() override
		{
			std::array<core::blake3_hash_t, E_IMAGE_REGIONS::EIR_COUNT> hashes;
			// load the image view
			system::path filename, extension;
			const auto asset = getImageView(m_nextPath, filename, extension);
			auto inView = std::move(*asset);

			auto execute = [&](E_IMAGE_REGIONS mode) -> bool
			{
				const auto modeAsString = [mode]() -> std::string
				{
					switch (mode)
					{
					case EIR_FLATTEN_FULL_EXTENT:
						return "EIR_FLATTEN_FULL_EXTENT";
					case EIR_FLATTEN_MULTI_OFFSET:
						return "EIR_FLATTEN_MULTI_OFFSET";
					case EIR_MULTI_OVERLAPPING_FULL_EXTENT:
						return "EIR_MULTI_OVERLAPPING_FULL_EXTENT";
					default:
						assert(false);
						return "";
					}
				}();

				auto getRespecifedView = [&]() -> core::smart_refctd_ptr<ICPUImageView>
				{
					if (!asset)
					{
						options.tests.passed = false;
						return nullptr;
					}

					// we always need to re-create the view because KTX image views load as 2D instead of 2D_ARRAY type
					auto outViewParams = inView->getCreationParameters();
					outViewParams.viewType = IImageView<ICPUImage>::E_TYPE::ET_2D_ARRAY;
					const auto* inImage = outViewParams.image.get();

					const auto inImageParams = inImage->getCreationParameters();
					smart_refctd_ptr<ICPUBuffer> inBuffer = core::make_smart_refctd_ptr<asset::CCustomAllocatorCPUBuffer<core::null_allocator<uint8_t>, true> >(inImage->getBuffer()->getSize(), (uint8_t*)inImage->getBuffer()->getPointer(), core::adopt_memory); // adopt memory & don't free it on exit
					const auto inRegions = inImage->getRegionArray();
					const auto inAmountOfRegions = inRegions->size();

					/*
						patterns to copy only the regions marked by X (for 0th mip level only we respecify)
					*/
					switch (mode)
					{
						/*
							+----+----+
							|         |
							+    X    +
							|         |
							+----+----+
						*/

						case EIR_FLATTEN_FULL_EXTENT:
						{
						} break;

						/*
							+----+----+
							| xx |    |
							+----+----+
							|    | xx |
							+----+----+
						*/

						case EIR_FLATTEN_MULTI_OFFSET:
						{
							std::vector<asset::IImage::SBufferCopy> newRegions;

							for (size_t i = 0; i < inAmountOfRegions; ++i)
							{
								const auto* const inRegion = inRegions->begin() + i;

								if (inRegion->imageSubresource.mipLevel == 0) // override 0th level
								{
									// all of that is valid because input comes from loaders hence is EIR_FLATTEN_FULL_EXTENT & tightly packed
									const auto quarterWidth = core::max(inRegion->imageExtent.width / 4, 1u);
									const auto quarterHeight = core::max(inRegion->imageExtent.height / 4, 1u);
									const auto texelBlockInfo = asset::TexelBlockInfo(inImageParams.format);
									const auto imageExtentsInBlocks = texelBlockInfo.convertTexelsToBlocks(core::vector3du32_SIMD(inRegion->imageExtent.width, inRegion->imageExtent.height, inRegion->imageExtent.depth));

									auto emplaceNewRegion = [&](uint32_t offsetMultiplier)
										{
											auto& newRegion = newRegions.emplace_back() = *inRegion;
											newRegion.imageExtent.width = quarterWidth;
											newRegion.imageExtent.height = quarterHeight;
											newRegion.imageExtent.depth = inRegion->imageExtent.depth;
											newRegion.imageOffset.x = quarterWidth * offsetMultiplier;
											newRegion.imageOffset.y = quarterHeight * offsetMultiplier;
											newRegion.imageOffset.z = 0u;
											auto offsetInBlocks = texelBlockInfo.convertTexelsToBlocks(core::vector3du32_SIMD(newRegion.imageOffset.x, newRegion.imageOffset.y, newRegion.imageOffset.z));
											newRegion.bufferOffset = (offsetInBlocks.y * imageExtentsInBlocks.x + offsetInBlocks.x) * texelBlockInfo.getBlockByteSize();
											newRegion.bufferRowLength = inRegion->imageExtent.width;
											newRegion.bufferImageHeight = inRegion->imageExtent.height;
										};

									emplaceNewRegion(1u);
									emplaceNewRegion(2u);
								}
								else
									newRegions.emplace_back() = *inRegion; // copy all left regions
							}

							auto outRegions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(newRegions.size());
							memcpy(outRegions->data(), newRegions.data(), outRegions->bytesize()); // full content copy

							auto outImage = smart_refctd_ptr_static_cast<ICPUImage>(inImage->clone(0u)); // without contents
							if (!outImage->setBufferAndRegions(smart_refctd_ptr(inBuffer), std::move(outRegions))) // do NOT make copy of the input buffer (we won't modify its content!) & set respecified regions
							{
								assert(false);
								return nullptr;
							}

							outImage->setContentHash(outImage->computeContentHash());

							outViewParams.image = std::move(outImage);
						} break;

						/*
							+====+----+
							||xx||    |
							+====+ xx +
							|         |
							+----+----+
						*/

						case EIR_MULTI_OVERLAPPING_FULL_EXTENT:
						{
							std::vector<asset::IImage::SBufferCopy> newRegions;

							for (size_t i = 0; i < inAmountOfRegions; ++i)
							{
								const auto* const inRegion = inRegions->begin() + i;
								newRegions.emplace_back() = *inRegion;


								if (inRegion->imageSubresource.mipLevel == 0) // add overlay region
								{
									const auto halfWidth = core::max(inRegion->imageExtent.width / 2, 1u);
									const auto halfHeight = core::max(inRegion->imageExtent.height / 2, 1u);

									auto& newRegion = newRegions.emplace_back() = *inRegion;
									newRegion.bufferRowLength = inRegion->imageExtent.width;
									newRegion.bufferImageHeight = inRegion->imageExtent.height;

									newRegion.imageExtent = { .width = halfWidth, .height = halfHeight, .depth = inRegion->imageExtent.depth };
									newRegion.imageOffset = { .x = 0, .y = 0, .z = 0 };
									newRegion.bufferOffset = 0;
								}
							}

							auto outRegions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(newRegions.size());
							memcpy(outRegions->data(), newRegions.data(), outRegions->bytesize()); // full content copy

							auto outImage = smart_refctd_ptr_static_cast<ICPUImage>(inImage->clone(0u)); // without contents
							if (!outImage->setBufferAndRegions(smart_refctd_ptr(inBuffer), std::move(outRegions))) // do NOT make copy of the input buffer (we won't modify its content!) & set respecified regions
							{
								assert(false);
								return nullptr;
							}

							outImage->setContentHash(outImage->computeContentHash());

							outViewParams.image = std::move(outImage);
						} break;

						default:
						{
							assert(false);
							return nullptr;
						} break;
					}

					return ICPUImageView::create(std::move(outViewParams));
				};

				const auto cpuImgView = getRespecifedView();

				if(!cpuImgView)
					return false;

				if (options.tests.enabled)
				{
					if (options.tests.mode == "hash")
					{
						bool passed = true;

						const auto* const image = cpuImgView->getCreationParameters().image.get();

						{
							auto hash = image->getContentHash();
							memcpy(static_cast<void*>(&hashes[mode]), hash.data, sizeof(hash));
						}

						const auto hash = [&image]()
						{
							auto hash = image->getContentHash();

							std::array<size_t, 4> data;
							memcpy(data.data(), hash.data, sizeof(hash));

							return data;
						}();

						struct
						{
							std::string path;
							json data;
						} current, reference;

						current.path = (localOutputCWD / filename).make_preferred().string() + "_" + modeAsString + extension.string() + ".json";

						m_logger->log("Perfoming [%s]th test!", ILogger::ELL_PERFORMANCE, std::to_string(options.tests.count.total).c_str());
						m_logger->log("Asset: \"%s\"", ILogger::ELL_INFO, m_nextPath.c_str());
						m_logger->log("Asset load time: %llu ms", ILogger::ELL_INFO, perfRes.lastLoadDuration);
						m_logger->log("Mode: \"%s\"", ILogger::ELL_INFO, modeAsString.c_str());
						m_logger->log("Writing \"%ls\"'s image hash to \"%s\"", ILogger::ELL_INFO, filename.c_str(), current.path.c_str());

						current.data["image"] = json::array();
						for (const auto& it : hash)
							current.data["image"].push_back(it);

						current.data["mode"] = modeAsString;

						const std::string prettyJson = current.data.dump(4);

						if (options.verbose)
							m_logger->log(prettyJson, ILogger::ELL_INFO);

						system::ISystem::future_t<core::smart_refctd_ptr<system::IFile>> future;
						m_system->createFile(future, current.path, system::IFileBase::ECF_WRITE);
						if (auto file = future.acquire(); file && bool(*file))
						{
							system::IFile::success_t succ;
							(*file)->write(succ, prettyJson.data(), 0, prettyJson.size());
							succ.getBytesProcessed(true);
						}
						else
						{
							m_logger->log("Failed to write \"%ls\"'s data to \"%s\" location!", ILogger::ELL_ERROR, filename.c_str(), current.path.c_str());
							passed = false;
						}

						reference.path = std::filesystem::absolute((localOutputCWD / (std::string("../test/references/") + filename.string() + "_" + modeAsString + extension.string() + ".json")).make_preferred()).string();
						{
							std::ifstream referenceFile(reference.path);

							if (referenceFile.is_open())
							{
								referenceFile >> reference.data;

								m_logger->log("Comparing \"%ls\"'s reference data..", ILogger::ELL_INFO, filename.c_str());
								const bool ok = current.data["image"] == reference.data["image"];

								if (ok)
								{
									++options.tests.count.passed;
									m_logger->log("Passed tests!", ILogger::ELL_WARNING);
								}
								else
								{
									logFail("Failed tests!");
									passed = false;
								}
							}
							else
							{
								m_logger->log("Could not open \"%s\"'s reference file! If the reference doesn't exist make sure to create one by executing the program with \"--update-references\" flag.", ILogger::ELL_ERROR, reference.path.c_str());
								passed = false;
							}
						}

						if (options.tests.updateReferences)
						{
							std::error_code errorCode;
							std::filesystem::copy(current.path, reference.path, std::filesystem::copy_options::overwrite_existing, errorCode);
							if (errorCode)
							{
								m_logger->log("Failed to update \"%ls\"'s reference file!", ILogger::ELL_ERROR, filename.c_str());
								passed = false;
							}
							else
								m_logger->log("Updated \"%ls\"'s reference file & saved to \"%s\"!", ILogger::ELL_INFO, filename.c_str(), reference.path.c_str());
						}

						++options.tests.count.total;

						if(!passed)
							options.tests.passed = false;
					}
					else
						assert(false);
				}
				else
				{
					// Can't reset a cmdbuffer before the previous use of commandbuffer is finished!
					if (m_submitIx>=m_maxFramesInFlight)
					{
						const ISemaphore::SWaitInfo cmdbufDonePending[] = {
							{ 
								.semaphore = m_semaphore.get(),
								.value = m_submitIx+1-m_maxFramesInFlight
							}
						};
						if (m_device->blockForSemaphores(cmdbufDonePending)!=ISemaphore::WAIT_RESULT::SUCCESS)
							return false;
					}
					const auto resourceIx = m_submitIx%m_maxFramesInFlight;

					// we don't want to overcomplicate the example with multi-queue
					auto queue = getGraphicsQueue();
					auto cmdbuf = m_cmdBufs[resourceIx].get();
					// needs to be open for the utility
					cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

					auto ds = m_descriptorSets[resourceIx].get();

					// want to capture the image data upload as well
					m_api->startCapture();

					// make sure we don't leave the tooling dangling if we fail
					auto endCaptureOnScopeExit = core::makeRAIIExiter([this]()->void{this->m_api->endCapture();});

					// get our GPU Image view
					auto converter = CAssetConverter::create({.device=m_device.get()});
					{
						// Test the provision of a custom patch this time
						CAssetConverter::patch_t<ICPUImageView> patch(cpuImgView.get(),IImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT);

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
						std::get<CAssetConverter::SInputs::asset_span_t<ICPUImageView>>(inputs.assets) = { &cpuImgView.get(),1 };
						std::get<CAssetConverter::SInputs::patch_span_t<ICPUImageView>>(inputs.patches) = { &patch,1 };
						inputs.logger = m_logger.get();

						//
						auto reservation = converter->reserve(inputs);

						// get the created image view
						auto gpuView = reservation.getGPUObjects<ICPUImageView>().front().value;
						if (!gpuView)
							return false;
						gpuView->getCreationParameters().image->setObjectDebugName(m_nextPath.c_str());

						// write to descriptor set
						{
							/*
							* Since we're using a combined image sampler with an immutable sampler, we only need to update the sampled image at the binding. Do note however that had we chosen
							* to use a mutable sampler instead, we'd need to write to it at least once, via the SDescriptorInfo info.info.combinedImageSampler.sampler field
							* WARNING: With an immutable sampler on a combined image sampler, trying to write to it is valid according to Vulkan spec, although the sampler is ignored and only
							* the image is updated. Please note that this is NOT the case in Nabla: if you try to write to a combined image sampler, then
							* info.info.combinedImageSampler.sampler MUST be nullptr
							*/
							IGPUDescriptorSet::SDescriptorInfo info = {};
							info.desc = std::move(gpuView);
							info.info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
							const IGPUDescriptorSet::SWriteDescriptorSet writes[] = { {
								.dstSet = ds,
								.binding = 0,
								.arrayElement = 0,
								.count = 1,
								.info = &info
							} };
							m_device->updateDescriptorSets(writes, {});
						}
						
						// we should multi-buffer to not stall before renderpass recording but oh well
						IQueue::SSubmitInfo::SCommandBufferInfo cmdbufInfo = {cmdbuf};

						// now convert
						m_intendedSubmit.scratchCommandBuffers = {&cmdbufInfo,1};
						// TODO: FIXME ImGUI needs to know what Queue will have ownership of the image AFTER its uploaded (need to know the family of the graphics queue)
						// right now, the transfer queue will stay the owner after upload
						CAssetConverter::SConvertParams params = {};
						params.transfer = &m_intendedSubmit;
						params.utilities = m_utils.get();
						auto result = reservation.convert(params);
						// block immediately
						if (result.copy()!=IQueue::RESULT::SUCCESS)
							return false;
					}

					// now we can sleep till we're ready for next render
					std::this_thread::sleep_until(m_lastImageEnqueued+DisplayImageDuration);
					m_lastImageEnqueued = clock_t::now();

					const auto& params = cpuImgView->getCreationParameters().image->getCreationParameters();
					const auto imageExtent = params.extent;
					push_constants_t pc;
					{
						const float realLayers = core::max(params.arrayLayers,imageExtent.depth);
						pc.grid.x = ceil(sqrt(realLayers));
						pc.grid.y = ceil(realLayers/float(pc.grid.x));
					}
					const VkExtent2D newWindowResolution = {imageExtent.width*pc.grid.x,imageExtent.height*pc.grid.y};
					if (newWindowResolution.width!=m_window->getWidth() || newWindowResolution.height!=m_window->getHeight())
					{
						// Resize the window
						m_winMgr->setWindowSize(m_window.get(),newWindowResolution.width,newWindowResolution.height);
						// Don't want to rely on the Swapchain OUT_OF_DATE causing an implicit re-create in the `acquireNextImage` because the
						// swapchain may report OUT_OF_DATE after the next VBlank after the resize, not getting the message right away.
						m_surface->recreateSwapchain();
					}
					// Now show the window (ideally should happen just after present, but don't want to mess with acquire/recreation)
					m_winMgr->show(m_window.get());

					// Acquire
					auto acquire = m_surface->acquireNextImage();
					if (!acquire)
						return false;

					// Render to the Image
					{
						// don't need a barrier on the image because the Asset Converter did a full barrier because of Layout Transition from TRANSFER DST to READ ONLY
						// and it also did a submit we blocked on with the host
						const VkRect2D currentRenderArea =
						{
							.offset = {0,0},
							.extent = {newWindowResolution.width,newWindowResolution.height}
						};
						// set viewport
						{
							const asset::SViewport viewport =
							{
								.width = float(newWindowResolution.width),
								.height = float(newWindowResolution.height)
							};
							cmdbuf->setViewport({&viewport,1});
						}
						cmdbuf->setScissor({&currentRenderArea,1});

						// begin the renderpass
						{
							const IGPUCommandBuffer::SClearColorValue clearValue = { .float32 = {1.f,0.f,1.f,1.f} };
							auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
							const IGPUCommandBuffer::SRenderpassBeginInfo info = {
								.framebuffer = scRes->getFramebuffer(acquire.imageIndex),
								.colorClearValues = &clearValue,
								.depthStencilClearValues = nullptr,
								.renderArea = currentRenderArea
							};
							cmdbuf->beginRenderPass(info,IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
						}
						cmdbuf->bindGraphicsPipeline(m_pipeline.get());
						cmdbuf->pushConstants(m_pipeline->getLayout(),IGPUShader::E_SHADER_STAGE::ESS_FRAGMENT,0,sizeof(push_constants_t),&pc);
						cmdbuf->bindDescriptorSets(nbl::asset::EPBP_GRAPHICS,m_pipeline->getLayout(),3,1,&ds);
						ext::FullScreenTriangle::recordDrawCall(cmdbuf);
						cmdbuf->endRenderPass();

						cmdbuf->end();
					}

					// submit
					const IQueue::SSubmitInfo::SSemaphoreInfo rendered[1] = {{
						.semaphore = m_semaphore.get(),
						.value = ++m_submitIx,
						// just as we've outputted all pixels, signal
						.stageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT
					}};
					{
						{
							const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[1] = {{
								.cmdbuf = cmdbuf
							}};
							// we don't need to wait for the transfer semaphore, because we submit everything to the same queue
							const IQueue::SSubmitInfo::SSemaphoreInfo acquired[1] = {{
								.semaphore = acquire.semaphore,
								.value = acquire.acquireCount,
								.stageMask = PIPELINE_STAGE_FLAGS::NONE
							}};
							const IQueue::SSubmitInfo infos[1] = {{
								.waitSemaphores = acquired,
								.commandBuffers = commandBuffers,
								.signalSemaphores = rendered
							}};
							// we won't signal the sema if no success
							if (queue->submit(infos)!=IQueue::RESULT::SUCCESS)
								m_submitIx--;
						}
					}
			
					// Set the Caption
					std::string viewTypeStr;
					switch (cpuImgView->getCreationParameters().viewType)
					{
						case IImageView<video::IGPUImage>::ET_2D:
							viewTypeStr = "ET_2D";
						case IImageView<video::IGPUImage>::ET_2D_ARRAY:
							viewTypeStr = "ET_2D_ARRAY";
							break;
						case IImageView<video::IGPUImage>::ET_CUBE_MAP:
							viewTypeStr = "ET_CUBE_MAP";
							break;
						default:
							assert(false);
							break;
					};
					m_window->setCaption("[Nabla Engine] Color Space Test Demo - CURRENT IMAGE: " + filename.string() + " - VIEW TYPE: " + viewTypeStr + " - EXTENSION: " + extension.string() + " - MODE: " + modeAsString);

					// Present
					m_surface->present(acquire.imageIndex, rendered);

					// Now do a write to disk in the meantime
					{
						const std::string assetPath = "imageAsset_" + filename.string() + modeAsString + extension.string();

						auto tryToWrite = [&](IAsset* asset)->bool
						{
							IAssetWriter::SAssetWriteParams wparams(asset);
							wparams.workingDirectory = localOutputCWD;
							return m_assetMgr->writeAsset(assetPath,wparams);
						};

						// try write as an image, else try as image view
						if (!tryToWrite(cpuImgView->getCreationParameters().image.get()))
							if (!tryToWrite(cpuImgView.get()))
								m_logger->log("Failed to write %s to disk!",ILogger::ELL_ERROR,assetPath.c_str());
					}

					// TODO: reuse the resources without frames in flight (cmon we block and do like one swap every 900ms)
				}
			
				return true;
			};

			for (const auto mode : { EIR_FLATTEN_FULL_EXTENT, EIR_FLATTEN_MULTI_OFFSET, EIR_MULTI_OVERLAPPING_FULL_EXTENT })
			{
				if (!execute(mode))
				{
					m_logger->log("Internal error, skipping execution for \"%s\" with \"%s\" mode!", ILogger::ELL_ERROR, m_nextPath.c_str(), std::to_string(mode).c_str());
					options.tests.passed = false;
				}
			}
			if (options.tests.enabled) 
			{
				if (options.tests.mode == "hash") 
				{
					bool passed = true;
					m_logger->log("Perfoming [%s]th test!", ILogger::ELL_PERFORMANCE, std::to_string(options.tests.count.total).c_str());
					m_logger->log("Asset: \"%s\"", ILogger::ELL_INFO, m_nextPath.c_str());
					m_logger->log("Comparing \"%ls\"'s hash between modes..", ILogger::ELL_INFO, filename.c_str());
					
					if (hashes[EIR_FLATTEN_FULL_EXTENT] != hashes[EIR_MULTI_OVERLAPPING_FULL_EXTENT]) 
					{
						logFail("failed EIR_FLATTEN_FULL_EXTENT == EIR_MULTI_OVERLAPPING_FULL_EXTENT hash check");
						passed = false;

					}
					if (hashes[EIR_FLATTEN_FULL_EXTENT] == hashes[EIR_FLATTEN_MULTI_OFFSET]) 
					{
						logFail("failed EIR_FLATTEN_FULL_EXTENT != EIR_FLATTEN_MULTI_OFFSET hash check");
						passed = false;
					}

					options.tests.count.total++;
					if (passed)
					{
						options.tests.count.passed++;
						m_logger->log("Passed tests!", ILogger::ELL_WARNING);
					}
					else 
					{
						options.tests.passed = false;
					}

				}
			}
		}

		inline bool keepRunning() override
		{
			if (!options.tests.enabled)
			{
				// Keep arunning as long as we have a surface to present to (usually this means, as long as the window is open)
				if (m_surface->irrecoverable())
					return false;
			}

			while (std::getline(m_testPathsFile,m_nextPath))
				if (m_nextPath!="" && m_nextPath[0]!=';')
					return true;

			// no more inputs in the file
			return false;
		}

		inline bool onAppTerminated() override
		{
			if (options.tests.enabled)
			{
				m_logger->log("Testing completed!", ILogger::ELL_PERFORMANCE);
				m_logger->log("Passed [%s/%s] tests.", ILogger::ELL_WARNING, std::to_string(options.tests.count.passed).c_str(), std::to_string(options.tests.count.total).c_str());
				m_logger->log("Load perf: \t total %llu ms \t average %llu ms", ILogger::ELL_PERFORMANCE, perfRes.totalLoadDuration, perfRes.totalLoadDuration / perfRes.count);
				exit(options.tests.passed ? 0 : 0x45); // do not remove this unless you want to refactor the example to cover destructors properly when in test mode & not crash the program
			}

			return device_base_t::onAppTerminated();
		}

	protected:
		smart_refctd_ptr<IWindow> m_window;
		smart_refctd_ptr<CSimpleResizeSurface<CDefaultSwapchainFramebuffers>> m_surface;
		//
		std::ifstream m_testPathsFile;
		system::path m_loadCWD;
		//
		std::string m_nextPath;
		clock_t::time_point m_lastImageEnqueued = {};
		//
		smart_refctd_ptr<IGPUGraphicsPipeline> m_pipeline;
		// We can't use the same semaphore for acquire and present, because that would disable "Frames in Flight" by syncing previous present against next acquire.
		smart_refctd_ptr<ISemaphore> m_semaphore;
		smart_refctd_ptr<IGPUCommandPool> m_cmdPool;
		// for image uploads
		smart_refctd_ptr<ISemaphore> m_scratchSemaphore;
		SIntendedSubmitInfo m_intendedSubmit;
		// Use a separate counter to cycle through our resources for clarity
		uint64_t m_submitIx : 59 = 0;
		// Maximum frames which can be simultaneously rendered
		uint64_t m_maxFramesInFlight : 5;
		// Enough Command Buffers and other resources for all frames in flight!
		std::array<smart_refctd_ptr<IGPUDescriptorSet>,ISwapchain::MaxImages> m_descriptorSets;
		std::array<smart_refctd_ptr<IGPUCommandBuffer>,ISwapchain::MaxImages> m_cmdBufs;

	private:

		struct PerfResult {
			unsigned long long lastLoadDuration = 0;
			unsigned long long totalLoadDuration = 0;
			size_t count = 0;
		} perfRes;

		std::optional<smart_refctd_ptr<ICPUImageView>> getImageView(std::string inAssetPath, system::path& outFilename, system::path& outExtension)
		{
			smart_refctd_ptr<ICPUImageView> view;

			m_logger->log("Loading image from path %s", ILogger::ELL_INFO, inAssetPath.c_str());

			constexpr auto cachingFlags = static_cast<IAssetLoader::E_CACHING_FLAGS>(IAssetLoader::ECF_DONT_CACHE_REFERENCES & IAssetLoader::ECF_DONT_CACHE_TOP_LEVEL);
			const IAssetLoader::SAssetLoadParams loadParams(0ull, nullptr, cachingFlags, IAssetLoader::ELPF_NONE, m_logger.get(), m_loadCWD);
			
			auto perfStart = clock_t::now();
			auto bundle = m_assetMgr->getAsset(inAssetPath, loadParams);
			auto perfEnd = clock_t::now();

			perfRes.lastLoadDuration = std::chrono::duration_cast<perf_clock_resolution_t>(perfEnd - perfStart).count();
			perfRes.totalLoadDuration += perfRes.lastLoadDuration;
			perfRes.count += 1;
			
			auto contents = bundle.getContents();
			if (contents.empty())
			{
				m_logger->log("Failed to load image with path %s, skipping!", ILogger::ELL_ERROR, (m_loadCWD / inAssetPath).c_str());
				return {};
			}

			core::splitFilename(inAssetPath.c_str(), nullptr, &outFilename, &outExtension);

			const auto& asset = contents[0];
			switch (asset->getAssetType())
			{
				case IAsset::ET_IMAGE:
				{
					auto image = smart_refctd_ptr_static_cast<ICPUImage>(asset);
					const auto format = image->getCreationParameters().format;

					ICPUImageView::SCreationParams viewParams = 
					{
						.flags = ICPUImageView::E_CREATE_FLAGS::ECF_NONE,
						.image = std::move(image),
						.viewType = IImageView<ICPUImage>::E_TYPE::ET_2D_ARRAY,
						.format = format,
						.subresourceRange = {
							.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT,
							.baseMipLevel = 0u,
							.levelCount = ICPUImageView::remaining_mip_levels,
							.baseArrayLayer = 0u,
							.layerCount = ICPUImageView::remaining_array_layers
						}
					};

					view = ICPUImageView::create(std::move(viewParams));
				} break;

				case IAsset::ET_IMAGE_VIEW:
					view = smart_refctd_ptr_static_cast<ICPUImageView>(asset);
					break;
				default:
					m_logger->log("Failed to load ICPUImage or ICPUImageView got some other Asset Type, skipping!", ILogger::ELL_ERROR);
					return {};
			}

			return view;
		}

		struct NBL_APP_OPTIONS
		{
			struct 
			{
				bool enabled = false, passed = true, updateReferences = false;
				std::string mode = {};
				
				struct
				{
					size_t total = {}, passed = {};
				} count;

			} tests;

			bool verbose = false;
		} options;

		enum E_IMAGE_REGIONS
		{
			EIR_FLATTEN_FULL_EXTENT,		//! from image loaders, single region per mip level, no overlapping & covers whole mip
			EIR_FLATTEN_MULTI_OFFSET,		//! respecified to have 2 regions for 0th mip level with no overlapping
			EIR_MULTI_OVERLAPPING_FULL_EXTENT,			//! respecified to have 2 regions for 0th mip level, second is embeded in first hence they are overlapping
			EIR_COUNT
		};
};

NBL_MAIN_FUNC(ColorSpaceTestSampleApp)