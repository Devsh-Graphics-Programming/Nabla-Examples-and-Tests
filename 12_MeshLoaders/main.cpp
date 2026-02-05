// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "argparse/argparse.hpp"
#include "common.hpp"

#include "../3rdparty/portable-file-dialogs/portable-file-dialogs.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cctype>

#ifdef NBL_BUILD_MITSUBA_LOADER
#include "nbl/ext/MitsubaLoader/CSerializedLoader.h"
#endif

#ifdef NBL_BUILD_DEBUG_DRAW
#include "nbl/ext/DebugDraw/CDrawAABB.h"
#endif
#include "nbl/ext/ScreenShot/ScreenShot.h"

class MeshLoadersApp final : public MonoWindowApplication, public BuiltinResourcesApplication
{
	using device_base_t = MonoWindowApplication;
	using asset_base_t = BuiltinResourcesApplication;

  enum DrawBoundingBoxMode
  {
    DBBM_NONE,
    DBBM_AABB,
    DBBM_OBB,
    DBBM_COUNT
  };

	enum class RunMode
	{
		Interactive,
		Batch,
		CI
	};

	enum class Phase
	{
		RenderOriginal,
		RenderWritten
	};

	struct TestCase
	{
		std::string name;
		nbl::system::path path;
	};

	struct CameraState
	{
		core::vectorSIMDf position;
		core::vectorSIMDf target;
		core::matrix4SIMD projection;
		float moveSpeed = 1.0f;
	};

	public:
		inline MeshLoadersApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
			: IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD),
			device_base_t({1280,720}, EF_D32_SFLOAT, _localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

	inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		if (!asset_base_t::onAppInitialized(smart_refctd_ptr(system)))
			return false;
#ifdef NBL_BUILD_MITSUBA_LOADER
		m_assetMgr->addAssetLoader(make_smart_refctd_ptr<ext::MitsubaLoader::CSerializedLoader>());
#endif
		if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
			return false;

		m_runMode = RunMode::Batch;
		m_saveGeomPrefixPath = localOutputCWD / "saved";
		m_screenshotPrefixPath = localOutputCWD / "screenshots";
		m_testListPath = localInputCWD / "meshloaders_inputs.json";

		argparse::ArgumentParser parser("12_meshloaders");
		parser.add_argument("--savegeometry")
			.help("Save the mesh on exit or reload")
			.flag();

		parser.add_argument("--savepath")
			.nargs(1)
			.help("Specify the file to which the mesh will be saved");
		parser.add_argument("--ci")
			.help("Run in CI mode: load test list, write .ply, capture screenshots, compare data, and exit.")
			.flag();
		parser.add_argument("--interactive")
			.help("Use file dialog to select a single model.")
			.flag();
		parser.add_argument("--testlist")
			.nargs(1)
			.help("JSON file with test cases. Relative paths are resolved against local input CWD.");

		try
		{
			parser.parse_args({ argv.data(), argv.data() + argv.size() });
		}
		catch (const std::exception& e)
		{
			return logFail(e.what());
		}

		if (parser["--savegeometry"] == true)
			m_saveGeom = true;
		if (parser["--interactive"] == true)
			m_runMode = RunMode::Interactive;
		if (parser["--ci"] == true)
			m_runMode = RunMode::CI;

		if (parser.present("--savepath"))
		{
			auto tmp = path(parser.get<std::string>("--savepath"));

			if (tmp.empty() || !tmp.has_filename())
				return logFail("Invalid path has been specified in --savepath argument");

			if (!std::filesystem::exists(tmp.parent_path()))
				return logFail("Path specified in --savepath argument doesn't exist");

			m_specifiedGeomSavePath.emplace(std::move(tmp.generic_string()));
		}

		if (parser.present("--testlist"))
		{
			auto tmp = path(parser.get<std::string>("--testlist"));
			if (tmp.empty())
				return logFail("Invalid path has been specified in --testlist argument");
			if (tmp.is_relative())
				tmp = localInputCWD / tmp;
			m_testListPath = tmp;
		}

		if (m_saveGeom)
			std::filesystem::create_directories(m_saveGeomPrefixPath);
		std::filesystem::create_directories(m_screenshotPrefixPath);

		m_semaphore = m_device->createSemaphore(m_realFrameIx);
		if (!m_semaphore)
			return logFail("Failed to Create a Semaphore!");

		auto pool = m_device->createCommandPool(getGraphicsQueue()->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
		for (auto i = 0u; i < MaxFramesInFlight; i++)
		{
			if (!pool)
				return logFail("Couldn't create Command Pool!");
			if (!pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { m_cmdBufs.data() + i,1 }))
				return logFail("Couldn't create Command Buffer!");
		}

		auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
		m_renderer = CSimpleDebugRenderer::create(m_assetMgr.get(), scRes->getRenderpass(), 0, {});
		if (!m_renderer)
			return logFail("Failed to create renderer!");

#ifdef NBL_BUILD_DEBUG_DRAW
			{
				auto* renderpass = scRes->getRenderpass();
				ext::debug_draw::DrawAABB::SCreationParameters params = {};
				params.assetManager = m_assetMgr;
				params.transfer = getTransferUpQueue();
				params.drawMode = ext::debug_draw::DrawAABB::ADM_DRAW_BATCH;
				params.batchPipelineLayout = ext::debug_draw::DrawAABB::createDefaultPipelineLayout(m_device.get());
				params.renderpass = smart_refctd_ptr<IGPURenderpass>(renderpass);
				params.utilities = m_utils;
				m_drawAABB = ext::debug_draw::DrawAABB::create(std::move(params));
			}
#endif

		if (!initTestCases())
			return false;

		if (isRowViewActive())
		{
			m_nonInteractiveTest = false;
			if (!loadRowView())
				return false;
		}
		else
		{
			if (m_runMode != RunMode::Interactive)
				m_nonInteractiveTest = true;
			if (!startCase(0u))
				return false;
		}

		camera.mapKeysToArrows();

		onAppInitializedFinish();
		return true;
	}

	inline IQueue::SSubmitInfo::SSemaphoreInfo renderFrame(const std::chrono::microseconds nextPresentationTimestamp) override
	{
		m_inputSystem->getDefaultMouse(&mouse);
		m_inputSystem->getDefaultKeyboard(&keyboard);

		//
		const auto resourceIx = m_realFrameIx % MaxFramesInFlight;

		auto* const cb = m_cmdBufs.data()[resourceIx].get();
		cb->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
		cb->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
		// clear to black for both things
		{
			// begin renderpass
			{
				auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
				auto* framebuffer = scRes->getFramebuffer(device_base_t::getCurrentAcquire().imageIndex);
				const IGPUCommandBuffer::SClearColorValue clearValue = { .float32 = {1.f,0.f,1.f,1.f} };
				const IGPUCommandBuffer::SClearDepthStencilValue depthValue = { .depth = 0.f };
				const VkRect2D currentRenderArea =
				{
					.offset = {0,0},
					.extent = {framebuffer->getCreationParameters().width,framebuffer->getCreationParameters().height}
				};
				const IGPUCommandBuffer::SRenderpassBeginInfo info =
				{
					.framebuffer = framebuffer,
					.colorClearValues = &clearValue,
					.depthStencilClearValues = &depthValue,
					.renderArea = currentRenderArea
				};
				cb->beginRenderPass(info, IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);

					const SViewport viewport = {
						.x = static_cast<float>(currentRenderArea.offset.x),
						.y = static_cast<float>(currentRenderArea.offset.y),
						.width = static_cast<float>(currentRenderArea.extent.width),
						.height = static_cast<float>(currentRenderArea.extent.height)
					};
					cb->setViewport(0u,1u,&viewport);
		
					cb->setScissor(0u,1u,&currentRenderArea);
				}
				// late latch input
				if (!m_nonInteractiveTest)
				{
					bool reload = false;
					camera.beginInputProcessing(nextPresentationTimestamp);
					mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void { camera.mouseProcess(events); }, m_logger.get());
					keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void
						{
							for (const auto& event : events)
							{
								if (event.keyCode == E_KEY_CODE::EKC_R && event.action == SKeyboardEvent::ECA_RELEASED)
									reload = true;
							}
							camera.keyboardProcess(events);
						},
						m_logger.get()
					);
					camera.endInputProcessing(nextPresentationTimestamp);
					if (reload)
						reloadInteractive();
				}
				// draw scene
				float32_t3x4 viewMatrix;
				float32_t4x4 viewProjMatrix;
				{
					// TODO: get rid of legacy matrices
					{
						memcpy(&viewMatrix,camera.getViewMatrix().pointer(),sizeof(viewMatrix));
						memcpy(&viewProjMatrix,camera.getConcatenatedMatrix().pointer(),sizeof(viewProjMatrix));
					}
 					m_renderer->render(cb,CSimpleDebugRenderer::SViewParams(viewMatrix,viewProjMatrix));
				}
#ifdef NBL_BUILD_DEBUG_DRAW
				{
					const ISemaphore::SWaitInfo drawFinished = { .semaphore = m_semaphore.get(),.value = m_realFrameIx + 1u };
					ext::debug_draw::DrawAABB::DrawParameters drawParams;
					drawParams.commandBuffer = cb;
					drawParams.cameraMat = viewProjMatrix;
					m_drawAABB->render(drawParams, drawFinished, m_aabbInstances);
				}
#endif
				cb->endRenderPass();
			}
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

		std::string caption = "[Nabla Engine] Mesh Loaders";
		{
			caption += ", displaying [";
			caption += m_modelPath;
			caption += "]";
			m_window->setCaption(caption);
		}
		if (isRowViewActive() && !m_rowViewScreenshotCaptured && m_realFrameIx >= RowViewFramesBeforeCapture)
		{
			if (!captureScreenshot(m_rowViewScreenshotPath, m_loadedScreenshot))
				failExit("Failed to capture row view screenshot.");
			m_rowViewScreenshotCaptured = true;
		}
		advanceCase();
		return retval;
	}

	inline bool onAppTerminated() override
	{
		return device_base_t::onAppTerminated();
	}

	inline bool keepRunning() override
	{
		if (m_shouldQuit)
			return false;
		return device_base_t::keepRunning();
	}

protected:
	const video::IGPURenderpass::SCreationParams::SSubpassDependency* getDefaultSubpassDependencies() const override
	{
		// Subsequent submits don't wait for each other, hence its important to have External Dependencies which prevent users of the depth attachment overlapping.
		const static IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] = {
			// wipe-transition of Color to ATTACHMENT_OPTIMAL and depth
			{
				.srcSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
				.dstSubpass = 0,
				.memoryBarrier = {
				// last place where the depth can get modified in previous frame, `COLOR_ATTACHMENT_OUTPUT_BIT` is implicitly later
				.srcStageMask = PIPELINE_STAGE_FLAGS::LATE_FRAGMENT_TESTS_BIT,
				// don't want any writes to be available, we'll clear 
				.srcAccessMask = ACCESS_FLAGS::NONE,
				// destination needs to wait as early as possible
				// TODO: `COLOR_ATTACHMENT_OUTPUT_BIT` shouldn't be needed, because its a logically later stage, see TODO in `ECommonEnums.h`
				.dstStageMask = PIPELINE_STAGE_FLAGS::EARLY_FRAGMENT_TESTS_BIT | PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
				// because depth and color get cleared first no read mask
				.dstAccessMask = ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
			}
			// leave view offsets and flags default
		},
			// color from ATTACHMENT_OPTIMAL to PRESENT_SRC
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

private:
	// TODO: standardise this across examples, and take from `argv`
	bool m_nonInteractiveTest = false;
	bool m_rowViewEnabled = true;
	bool m_rowViewScreenshotCaptured = false;

	template<typename... Args>
	[[noreturn]] void failExit(const char* msg, Args... args)
	{
		if (m_logger)
			m_logger->log(msg, ILogger::ELL_ERROR, args...);
		std::exit(-1);
	}

	bool initTestCases()
	{
		m_cases.clear();
		if (m_runMode == RunMode::Interactive)
		{
			system::path picked;
			if (!pickModelPath(picked))
				return logFail("No file selected.");
			m_cases.push_back({ picked.stem().string(), picked });
			return true;
		}
		return loadTestList(m_testListPath);
	}

	bool pickModelPath(system::path& outPath)
	{
		pfd::open_file file("Choose a supported Model File", sharedInputCWD.string(),
			{
				"All Supported Formats", "*.ply *.stl *.serialized *.obj",
				"TODO (.ply)", "*.ply",
				"TODO (.stl)", "*.stl",
				"Mitsuba 0.6 Serialized (.serialized)", "*.serialized",
				"Wavefront Object (.obj)", "*.obj"
			},
			false
		);
		if (file.result().empty())
			return false;
		outPath = file.result()[0];
		return true;
	}

	bool loadTestList(const system::path& jsonPath)
	{
		if (!std::filesystem::exists(jsonPath))
			return logFail("Missing test list: %s", jsonPath.string().c_str());

		std::ifstream stream(jsonPath);
		if (!stream.is_open())
			return logFail("Failed to open test list: %s", jsonPath.string().c_str());

		nlohmann::json doc;
		try
		{
			stream >> doc;
		}
		catch (const std::exception& e)
		{
			return logFail("Invalid JSON in test list: %s", e.what());
		}

		if (!doc.contains("cases") || !doc["cases"].is_array())
			return logFail("Test list JSON missing \"cases\" array.");

		if (doc.contains("row_view"))
		{
			if (!doc["row_view"].is_boolean())
				return logFail("\"row_view\" must be a boolean.");
			m_rowViewEnabled = doc["row_view"].get<bool>();
		}

		for (const auto& entry : doc["cases"])
		{
			std::string pathString;
			std::string name;
			std::string extOverride;

			if (entry.is_string())
			{
				pathString = entry.get<std::string>();
			}
			else if (entry.is_object())
			{
				if (!entry.contains("path") || !entry["path"].is_string())
					return logFail("Test list entry missing \"path\".");
				pathString = entry["path"].get<std::string>();
				if (entry.contains("name") && entry["name"].is_string())
					name = entry["name"].get<std::string>();
				if (entry.contains("extension") && entry["extension"].is_string())
					extOverride = entry["extension"].get<std::string>();
			}
			else
				return logFail("Invalid test list entry.");

			system::path path = pathString;
			if (path.is_relative())
				path = sharedInputCWD / path;
			if (!std::filesystem::exists(path))
				return logFail("Missing test input: %s", path.string().c_str());
			if (!extOverride.empty())
			{
				if (path.extension().string() != extOverride)
					return logFail("Extension mismatch for %s", path.string().c_str());
			}

			if (name.empty())
				name = path.stem().string();

			m_cases.push_back({ name, path });
		}

		if (m_cases.empty())
			return logFail("No test cases in test list.");

		return true;
	}

	bool isRowViewActive() const
	{
		return m_rowViewEnabled && m_runMode != RunMode::CI && m_runMode != RunMode::Interactive;
	}

	static inline std::string normalizeExtension(const system::path& path)
	{
		auto ext = path.extension().string();
		std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
		return ext;
	}

	bool isWriteExtensionSupported(const std::string& ext) const
	{
		if (ext == ".ply" || ext == ".stl")
			return true;
#ifdef _NBL_COMPILE_WITH_OBJ_WRITER_
		if (ext == ".obj")
			return true;
#endif
		return false;
	}

	system::path resolveSavePath(const system::path& modelPath) const
	{
		if (m_specifiedGeomSavePath)
			return path(*m_specifiedGeomSavePath);
		const auto stem = modelPath.stem().string();
		auto ext = normalizeExtension(modelPath);
		if (ext.empty())
			ext = ".ply";
		if (!isWriteExtensionSupported(ext))
		{
			if (m_logger)
				m_logger->log("No writer for %s, writing .ply instead.", ILogger::ELL_WARNING, ext.c_str());
			ext = ".ply";
		}
		return m_saveGeomPrefixPath / (stem + "_written" + ext);
	}

	bool startCase(const size_t index)
	{
		if (index >= m_cases.size())
			return false;

		m_caseIndex = index;
		m_phase = Phase::RenderOriginal;
		m_phaseFrameCounter = 0u;
		m_loadedScreenshot = nullptr;
		m_writtenScreenshot = nullptr;
		m_referenceCamera.reset();
		m_referenceCpuGeom = nullptr;
		m_hasReferenceGeometry = false;
		m_hasReferenceGeometryHash = false;

		const auto& testCase = m_cases[m_caseIndex];
		m_caseName = testCase.name.empty() ? testCase.path.stem().string() : testCase.name;
		m_writtenPath = resolveSavePath(testCase.path);
		m_loadedScreenshotPath = m_screenshotPrefixPath / ("meshloaders_" + m_caseName + "_loaded.png");
		m_writtenScreenshotPath = m_screenshotPrefixPath / ("meshloaders_" + m_caseName + "_written.png");

		if (!loadModel(testCase.path, true, true))
			return false;

		if (m_currentCpuGeom)
		{
			m_referenceCpuGeom = m_currentCpuGeom;
			m_hasReferenceGeometry = true;
			m_referenceGeometryHash = hashGeometry(m_referenceCpuGeom.get());
			m_hasReferenceGeometryHash = true;
		}

		return true;
	}

	bool advanceToNextCase()
	{
		const auto nextIndex = m_caseIndex + 1u;
		if (nextIndex >= m_cases.size())
		{
			m_shouldQuit = true;
			return false;
		}
		if (!startCase(nextIndex))
		{
			m_shouldQuit = true;
			return false;
		}
		return true;
	}

	void reloadInteractive()
	{
		system::path picked;
		if (!pickModelPath(picked))
			failExit("No file selected.");
		if (!loadModel(picked, true, true))
			failExit("Failed to load asset %s.", picked.string().c_str());
		if (m_currentCpuGeom && m_saveGeom)
		{
			const auto savePath = resolveSavePath(picked);
			if (!writeGeometry(m_currentCpuGeom, savePath.string()))
				failExit("Geometry write failed.");
		}
	}

	bool loadModel(const system::path& modelPath, const bool updateCamera, const bool storeCamera)
	{
		if (modelPath.empty())
			failExit("Empty model path.");
		if (!std::filesystem::exists(modelPath))
			failExit("Missing input: %s", modelPath.string().c_str());

		m_modelPath = modelPath.string();

		// free up
		m_renderer->m_instances.clear();
		m_renderer->clearGeometries({ .semaphore = m_semaphore.get(),.value = m_realFrameIx });
		m_assetMgr->clearAllAssetCache();

		//! load the geometry
		IAssetLoader::SAssetLoadParams params = {};
		params.logger = m_logger.get();
		auto asset = m_assetMgr->getAsset(m_modelPath, params);
		if (asset.getContents().empty())
			failExit("Failed to load asset %s.", m_modelPath.c_str());

		// 
		core::vector<smart_refctd_ptr<const ICPUPolygonGeometry>> geometries;
		if (!appendGeometriesFromBundle(asset, geometries))
			failExit("Asset loaded but not a supported type for %s.", m_modelPath.c_str());
		if (geometries.empty())
			failExit("No geometry found in asset %s.", m_modelPath.c_str());

		m_currentCpuGeom = geometries[0];

		using aabb_t = hlsl::shapes::AABB<3, double>;
		auto printAABB = [&](const aabb_t& aabb, const char* extraMsg = "")->void
			{
				m_logger->log("%s AABB is (%f,%f,%f) -> (%f,%f,%f)", ILogger::ELL_INFO, extraMsg, aabb.minVx.x, aabb.minVx.y, aabb.minVx.z, aabb.maxVx.x, aabb.maxVx.y, aabb.maxVx.z);
			};
		auto bound = aabb_t::create();
		// convert the geometries
		{
			smart_refctd_ptr<CAssetConverter> converter = CAssetConverter::create({ .device = m_device.get() });

			const auto transferFamily = getTransferUpQueue()->getFamilyIndex();

			struct SInputs : CAssetConverter::SInputs
			{
				virtual inline std::span<const uint32_t> getSharedOwnershipQueueFamilies(const size_t groupCopyID, const asset::ICPUBuffer* buffer, const CAssetConverter::patch_t<asset::ICPUBuffer>& patch) const
				{
					return sharedBufferOwnership;
				}

				core::vector<uint32_t> sharedBufferOwnership;
			} inputs = {};
			core::vector<CAssetConverter::patch_t<ICPUPolygonGeometry>> patches(geometries.size(), CSimpleDebugRenderer::DefaultPolygonGeometryPatch);
			{
				inputs.logger = m_logger.get();
				std::get<CAssetConverter::SInputs::asset_span_t<ICPUPolygonGeometry>>(inputs.assets) = { &geometries.front().get(),geometries.size() };
				std::get<CAssetConverter::SInputs::patch_span_t<ICPUPolygonGeometry>>(inputs.patches) = patches;
				// set up shared ownership so we don't have to 
				core::unordered_set<uint32_t> families;
				families.insert(transferFamily);
				families.insert(getGraphicsQueue()->getFamilyIndex());
				if (families.size() > 1)
					for (const auto fam : families)
						inputs.sharedBufferOwnership.push_back(fam);
			}

			// reserve
			auto reservation = converter->reserve(inputs);
			if (!reservation)
			{
				failExit("Failed to reserve GPU objects for CPU->GPU conversion.");
			}

			// convert
			{
				auto semaphore = m_device->createSemaphore(0u);

				constexpr auto MultiBuffering = 2;
				std::array<smart_refctd_ptr<IGPUCommandBuffer>, MultiBuffering> commandBuffers = {};
				{
					auto pool = m_device->createCommandPool(transferFamily, IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT | IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
					pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, commandBuffers, smart_refctd_ptr(m_logger));
				}
				commandBuffers.front()->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

				std::array<IQueue::SSubmitInfo::SCommandBufferInfo, MultiBuffering> commandBufferSubmits;
				for (auto i = 0; i < MultiBuffering; i++)
					commandBufferSubmits[i].cmdbuf = commandBuffers[i].get();

				SIntendedSubmitInfo transfer = {};
				transfer.queue = getTransferUpQueue();
				transfer.scratchCommandBuffers = commandBufferSubmits;
				transfer.scratchSemaphore = {
					.semaphore = semaphore.get(),
					.value = 0u,
					.stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
				};

				CAssetConverter::SConvertParams cpar = {};
				cpar.utilities = m_utils.get();
				cpar.transfer = &transfer;

					// basically it records all data uploads and submits them right away
					auto future = reservation.convert(cpar);
					if (future.copy()!=IQueue::RESULT::SUCCESS)
					{
						failExit("Failed to await submission feature.");
					}
				}

				auto tmp = hlsl::float32_t4x3(
					hlsl::float32_t3(1,0,0),
					hlsl::float32_t3(0,1,0),
					hlsl::float32_t3(0,0,1),
					hlsl::float32_t3(0,0,0)
				);
				core::vector<hlsl::float32_t3x4> worldTforms;
				const auto& converted = reservation.getGPUObjects<ICPUPolygonGeometry>();
				m_aabbInstances.resize(converted.size());
				m_obbInstances.resize(converted.size());
				for (uint32_t i = 0; i < converted.size(); i++)
				{
					const auto& geom = converted[i];
					const auto& cpuGeom = geometries[i].get();
					CPolygonGeometryManipulator::recomputeAABB(cpuGeom);
					const auto promoted = cpuGeom->getAABB<aabb_t>();
					printAABB(promoted,"Geometry");
					const auto promotedWorld = hlsl::float64_t3x4(worldTforms.emplace_back(hlsl::transpose(tmp)));
					const auto translation = hlsl::float64_t3(
						static_cast<double>(tmp[3].x),
						static_cast<double>(tmp[3].y),
						static_cast<double>(tmp[3].z));
					const auto transformed = translateAABB(promoted, translation);
					printAABB(transformed,"Transformed");
					bound = hlsl::shapes::util::union_(transformed,bound);

#ifdef NBL_BUILD_DEBUG_DRAW

					auto& aabbInst = m_aabbInstances[i];
					const auto tmpAabb = shapes::AABB<3,float>(promoted.minVx, promoted.maxVx);

					hlsl::float32_t3x4 aabbTransform = ext::debug_draw::DrawAABB::getTransformFromAABB(tmpAabb);
					const auto tmpWorld = hlsl::float32_t3x4(promotedWorld);
					const auto world4x4 = float32_t4x4{
						tmpWorld[0],
						tmpWorld[1],
						tmpWorld[2],
						float32_t4(0, 0, 0, 1)
					};

					aabbInst.color = { 1,1,1,1 };
					aabbInst.transform = math::linalg::promoted_mul(world4x4, aabbTransform);

					auto& obbInst = m_obbInstances[i];
					const auto obb = CPolygonGeometryManipulator::calculateOBB(
						cpuGeom->getPositionView().getElementCount(), 
						[geo = cpuGeom, &world4x4](size_t vertex_i) {
							hlsl::float32_t3 pt;
							geo->getPositionView().decodeElement(vertex_i, pt);
							return pt;
						});
					obbInst.color = { 0, 0, 1, 1 };
					obbInst.transform = math::linalg::promoted_mul(world4x4, obb.transform);
#endif
				}

				printAABB(bound,"Total");
				if (!m_renderer->addGeometries({ &converted.front().get(),converted.size() }))
					failExit("Failed to add geometries to renderer.");

			auto worlTformsIt = worldTforms.begin();
			for (const auto& geo : m_renderer->getGeometries())
				m_renderer->m_instances.push_back({
					.world = *(worlTformsIt++),
					.packedGeo = &geo
					});
		}

		if (updateCamera)
		{
			setupCameraFromAABB(bound);
			if (storeCamera)
				storeCameraState();
		}
		else if (m_referenceCamera)
			applyCameraState(*m_referenceCamera);
		else
			setupCameraFromAABB(bound);

		return true;
	}

	bool loadRowView()
	{
		if (m_cases.empty())
			failExit("No test cases loaded for row view.");

		m_renderer->m_instances.clear();
		m_renderer->clearGeometries({ .semaphore = m_semaphore.get(),.value = m_realFrameIx });
		m_assetMgr->clearAllAssetCache();

		core::vector<smart_refctd_ptr<const ICPUPolygonGeometry>> geometries;
		core::vector<hlsl::shapes::AABB<3, double>> aabbs;
		geometries.reserve(m_cases.size());
		aabbs.reserve(m_cases.size());

		IAssetLoader::SAssetLoadParams params = {};
		params.logger = m_logger.get();

		for (const auto& testCase : m_cases)
		{
			const auto& path = testCase.path;
			if (!std::filesystem::exists(path))
				failExit("Missing input: %s", path.string().c_str());

			auto asset = m_assetMgr->getAsset(path.string(), params);
			if (asset.getContents().empty())
				failExit("Failed to load asset %s.", path.string().c_str());

			smart_refctd_ptr<const ICPUPolygonGeometry> geom;
			core::vector<smart_refctd_ptr<const ICPUPolygonGeometry>> found;
			if (appendGeometriesFromBundle(asset, found))
			{
				if (!found.empty())
					geom = found.front();
			}
			if (!geom)
				failExit("No geometry found in asset %s.", path.string().c_str());

			CPolygonGeometryManipulator::recomputeAABB(geom.get());
			const auto aabb = geom->getAABB<hlsl::shapes::AABB<3, double>>();

			geometries.push_back(std::move(geom));
			aabbs.push_back(aabb);
		}

		if (geometries.empty())
			failExit("No geometry found for row view.");

		using aabb_t = hlsl::shapes::AABB<3, double>;
		auto printAABB = [&](const aabb_t& aabb, const char* extraMsg = "")->void
			{
				m_logger->log("%s AABB is (%f,%f,%f) -> (%f,%f,%f)", ILogger::ELL_INFO, extraMsg, aabb.minVx.x, aabb.minVx.y, aabb.minVx.z, aabb.maxVx.x, aabb.maxVx.y, aabb.maxVx.z);
			};
		auto bound = aabb_t::create();

		smart_refctd_ptr<CAssetConverter> converter = CAssetConverter::create({ .device = m_device.get() });
		const auto transferFamily = getTransferUpQueue()->getFamilyIndex();

		struct SInputs : CAssetConverter::SInputs
		{
			virtual inline std::span<const uint32_t> getSharedOwnershipQueueFamilies(const size_t, const asset::ICPUBuffer*, const CAssetConverter::patch_t<asset::ICPUBuffer>&) const
			{
				return sharedBufferOwnership;
			}

			core::vector<uint32_t> sharedBufferOwnership;
		} inputs = {};
		core::vector<CAssetConverter::patch_t<ICPUPolygonGeometry>> patches(geometries.size(), CSimpleDebugRenderer::DefaultPolygonGeometryPatch);
		{
			inputs.logger = m_logger.get();
			std::get<CAssetConverter::SInputs::asset_span_t<ICPUPolygonGeometry>>(inputs.assets) = { &geometries.front().get(),geometries.size() };
			std::get<CAssetConverter::SInputs::patch_span_t<ICPUPolygonGeometry>>(inputs.patches) = patches;
			core::unordered_set<uint32_t> families;
			families.insert(transferFamily);
			families.insert(getGraphicsQueue()->getFamilyIndex());
			if (families.size() > 1)
				for (const auto fam : families)
					inputs.sharedBufferOwnership.push_back(fam);
		}

		auto reservation = converter->reserve(inputs);
		if (!reservation)
			failExit("Failed to reserve GPU objects for CPU->GPU conversion.");

		{
			auto semaphore = m_device->createSemaphore(0u);

			constexpr auto MultiBuffering = 2;
			std::array<smart_refctd_ptr<IGPUCommandBuffer>, MultiBuffering> commandBuffers = {};
			{
				auto pool = m_device->createCommandPool(transferFamily, IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT | IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
				pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, commandBuffers, smart_refctd_ptr(m_logger));
			}
			commandBuffers.front()->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

			std::array<IQueue::SSubmitInfo::SCommandBufferInfo, MultiBuffering> commandBufferSubmits;
			for (auto i = 0; i < MultiBuffering; i++)
				commandBufferSubmits[i].cmdbuf = commandBuffers[i].get();

			SIntendedSubmitInfo transfer = {};
			transfer.queue = getTransferUpQueue();
			transfer.scratchCommandBuffers = commandBufferSubmits;
			transfer.scratchSemaphore = {
				.semaphore = semaphore.get(),
				.value = 0u,
				.stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
			};

			CAssetConverter::SConvertParams cpar = {};
			cpar.utilities = m_utils.get();
			cpar.transfer = &transfer;

			auto future = reservation.convert(cpar);
			if (future.copy() != IQueue::RESULT::SUCCESS)
				failExit("Failed to await submission feature.");
		}

		double targetExtent = 0.0;
		core::vector<double> maxDims;
		maxDims.reserve(aabbs.size());
		for (const auto& aabb : aabbs)
		{
			const auto extent = aabb.getExtent();
			const double maxDim = std::max({ extent.x, extent.y, extent.z, 0.001 });
			maxDims.push_back(maxDim);
			if (maxDim > targetExtent)
				targetExtent = maxDim;
		}

		core::vector<double> scales;
		scales.reserve(aabbs.size());
		for (const auto maxDim : maxDims)
			scales.push_back(targetExtent / maxDim);

		double maxWidth = 0.0;
		double totalWidth = 0.0;
		core::vector<double> widths;
		widths.reserve(aabbs.size());
		for (size_t i = 0; i < aabbs.size(); ++i)
		{
			const auto extent = aabbs[i].getExtent();
			const double width = std::max(0.001, extent.x * scales[i]);
			widths.push_back(width);
			totalWidth += width;
			if (width > maxWidth)
				maxWidth = width;
		}
		const double spacing = std::max(0.05 * maxWidth, 0.01);
		const double totalSpan = totalWidth + spacing * double(widths.size() > 0 ? widths.size() - 1 : 0);
		double cursor = -0.5 * totalSpan;

		auto tmp = hlsl::float32_t4x3(
			hlsl::float32_t3(1, 0, 0),
			hlsl::float32_t3(0, 1, 0),
			hlsl::float32_t3(0, 0, 1),
			hlsl::float32_t3(0, 0, 0)
		);
		const auto& converted = reservation.getGPUObjects<ICPUPolygonGeometry>();
		core::vector<hlsl::float32_t3x4> worldTforms;
		worldTforms.reserve(converted.size());
		m_aabbInstances.resize(converted.size());
		m_obbInstances.resize(converted.size());

		for (uint32_t i = 0; i < converted.size(); i++)
		{
			const auto& geom = converted[i];
			const auto& cpuGeom = geometries[i].get();
			const auto aabb = aabbs[i];
			printAABB(aabb, "Geometry");

			const double scale = scales[i];
			const auto center = (aabb.minVx + aabb.maxVx) * 0.5;
			const double width = widths[i];
			const double targetCenterX = cursor + 0.5 * width;
			cursor += width + spacing;

			const double tx = targetCenterX - scale * center.x;
			const double ty = -scale * center.y;
			const double tz = -scale * center.z;
			tmp[0] = hlsl::float32_t3(static_cast<float>(scale), 0.f, 0.f);
			tmp[1] = hlsl::float32_t3(0.f, static_cast<float>(scale), 0.f);
			tmp[2] = hlsl::float32_t3(0.f, 0.f, static_cast<float>(scale));
			tmp[3] = hlsl::float32_t3(static_cast<float>(tx), static_cast<float>(ty), static_cast<float>(tz));

			const auto promotedWorld = hlsl::float64_t3x4(worldTforms.emplace_back(hlsl::transpose(tmp)));
			const auto translation = hlsl::float64_t3(tx, ty, tz);
			const auto scaled = scaleAABB(aabb, scale);
			const auto transformed = translateAABB(scaled, translation);
			printAABB(transformed, "Transformed");
			bound = hlsl::shapes::util::union_(transformed, bound);

#ifdef NBL_BUILD_DEBUG_DRAW
			auto& aabbInst = m_aabbInstances[i];
			const auto tmpAabb = shapes::AABB<3, float>(aabb.minVx, aabb.maxVx);
			hlsl::float32_t3x4 aabbTransform = ext::debug_draw::DrawAABB::getTransformFromAABB(tmpAabb);
			const auto tmpWorld = hlsl::float32_t3x4(promotedWorld);
			const auto world4x4 = float32_t4x4{
				tmpWorld[0],
				tmpWorld[1],
				tmpWorld[2],
				float32_t4(0, 0, 0, 1)
			};
			aabbInst.color = { 1,1,1,1 };
			aabbInst.transform = math::linalg::promoted_mul(world4x4, aabbTransform);

			auto& obbInst = m_obbInstances[i];
			const auto obb = CPolygonGeometryManipulator::calculateOBB(
				cpuGeom->getPositionView().getElementCount(),
				[geo = cpuGeom](size_t vertex_i) {
					hlsl::float32_t3 pt;
					geo->getPositionView().decodeElement(vertex_i, pt);
					return pt;
				});
			obbInst.color = { 0, 0, 1, 1 };
			obbInst.transform = math::linalg::promoted_mul(world4x4, obb.transform);
#endif
		}

		printAABB(bound, "Total");
		if (!m_renderer->addGeometries({ &converted.front().get(),converted.size() }))
			failExit("Failed to add geometries to renderer.");

		for (uint32_t i = 0; i < converted.size(); i++)
		{
			m_renderer->m_instances.push_back({
				.world = worldTforms[i],
				.packedGeo = &m_renderer->getGeometry(i)
				});
		}

		setupCameraFromAABB(bound);
		m_modelPath = "Row view (all meshes)";
		m_rowViewScreenshotPath = m_screenshotPrefixPath / "meshloaders_row_view.png";
		m_rowViewScreenshotCaptured = false;
		return true;
	}

	bool writeGeometry(smart_refctd_ptr<const ICPUPolygonGeometry> geometry, const std::string& savePath)
	{
		IAsset* assetPtr = const_cast<IAsset*>(static_cast<const IAsset*>(geometry.get()));
		const auto ext = normalizeExtension(system::path(savePath));
		auto flags = asset::EWF_MESH_IS_RIGHT_HANDED;
		if (ext != ".obj")
			flags = static_cast<asset::E_WRITER_FLAGS>(flags | asset::EWF_BINARY);
		IAssetWriter::SAssetWriteParams params{ assetPtr, flags };
		m_logger->log("Saving mesh to %s", ILogger::ELL_INFO, savePath.c_str());
		if (!m_assetMgr->writeAsset(savePath, params))
		{
			m_logger->log("Failed to save %s", ILogger::ELL_ERROR, savePath.c_str());
			return false;
		}
		m_logger->log("Mesh successfully saved!", ILogger::ELL_INFO);
		return true;
	}

	void setupCameraFromAABB(const hlsl::shapes::AABB<3, double>& bound)
	{
		const auto extent = bound.getExtent();
		const auto aspectRatio = double(m_window->getWidth()) / double(m_window->getHeight());
		const double fovY = 1.2;
		const double fovX = 2.0 * std::atan(std::tan(fovY * 0.5) * aspectRatio);
		const auto center = (bound.minVx + bound.maxVx) * 0.5;
		const auto halfExtent = extent * 0.5;
		const double halfX = std::max(halfExtent.x, 0.001);
		const double halfY = std::max(halfExtent.y, 0.001);
		const double halfZ = std::max(halfExtent.z, 0.001);
		const double safeRadius = std::max({ halfX, halfY, halfZ });

		const double distY = halfY / std::tan(fovY * 0.5);
		const double distX = halfX / std::tan(fovX * 0.5);
		double dist = std::max(distX, distY) + halfZ;
		dist *= 1.1;

		const auto dir = hlsl::float64_t3(0.0, 0.0, 1.0);
		const auto pos = center + dir * dist;

		const double margin = halfZ * 0.1 + 0.01;
		const double nearPlane = std::max(0.001, dist - halfZ - margin);
		const double farPlane = dist + halfZ + margin;

		camera.setProjectionMatrix(core::matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(static_cast<float>(fovY), static_cast<float>(aspectRatio), static_cast<float>(nearPlane), static_cast<float>(farPlane)));
		camera.setMoveSpeed(static_cast<float>(safeRadius * 0.1));
		camera.setPosition(vectorSIMDf(pos.x, pos.y, pos.z));
		camera.setTarget(vectorSIMDf(center.x, center.y, center.z));
	}

	static inline hlsl::shapes::AABB<3, double> translateAABB(const hlsl::shapes::AABB<3, double>& aabb, const hlsl::float64_t3& translation)
	{
		auto out = aabb;
		out.minVx += translation;
		out.maxVx += translation;
		return out;
	}

	static inline hlsl::shapes::AABB<3, double> scaleAABB(const hlsl::shapes::AABB<3, double>& aabb, const double scale)
	{
		auto out = aabb;
		out.minVx *= scale;
		out.maxVx *= scale;
		return out;
	}

	void storeCameraState()
	{
		m_referenceCamera = CameraState{
			camera.getPosition(),
			camera.getTarget(),
			camera.getProjectionMatrix(),
			camera.getMoveSpeed()
		};
	}

	void applyCameraState(const CameraState& state)
	{
		camera.setProjectionMatrix(state.projection);
		camera.setPosition(state.position);
		camera.setTarget(state.target);
		camera.setMoveSpeed(state.moveSpeed);
	}

	core::blake3_hash_t hashGeometry(const ICPUPolygonGeometry* geo)
	{
		core::blake3_hasher hasher;
		if (!geo)
			return static_cast<core::blake3_hash_t>(hasher);

		const auto* indexing = geo->getIndexingCallback();
		const bool hasIndexing = indexing != nullptr;
		hasher.update(&hasIndexing, sizeof(hasIndexing));
		if (hasIndexing)
		{
			const auto topology = indexing->knownTopology();
			hasher << topology;
		}

		auto hashView = [&](const ICPUPolygonGeometry::SDataView& view)
		{
			const bool present = static_cast<bool>(view);
			hasher.update(&present, sizeof(present));
			if (!present)
				return;

			const auto format = view.composed.format;
			const auto stride = view.composed.getStride();
			hasher.update(&format, sizeof(format));
			hasher.update(&stride, sizeof(stride));
			hasher.update(&view.src.offset, sizeof(view.src.offset));
			hasher.update(&view.src.size, sizeof(view.src.size));
			const auto rangeFormat = view.composed.rangeFormat;
			hasher.update(&rangeFormat, sizeof(rangeFormat));
			view.composed.visitRange([&](const auto& range)
				{
					hasher.update(&range.minVx, sizeof(range.minVx));
					hasher.update(&range.maxVx, sizeof(range.maxVx));
				});

			if (view.src.buffer)
			{
				const auto bufHash = view.src.buffer->computeContentHash();
				hasher << bufHash;
			}
		};

		hashView(geo->getPositionView());
		hashView(geo->getNormalView());
		hashView(geo->getIndexView());

		const auto& auxViews = geo->getAuxAttributeViews();
		const uint64_t auxCount = static_cast<uint64_t>(auxViews.size());
		hasher.update(&auxCount, sizeof(auxCount));
		for (const auto& view : auxViews)
			hashView(view);

		return static_cast<core::blake3_hash_t>(hasher);
	}

	struct GeometryCompareResult
	{
		uint64_t vertexCountA = 0u;
		uint64_t vertexCountB = 0u;
		bool hasNormalA = false;
		bool hasNormalB = false;
		bool hasUvA = false;
		bool hasUvB = false;
		uint64_t indexCountA = 0u;
		uint64_t indexCountB = 0u;
		uint64_t posDiffCount = 0u;
		double posMaxAbs = 0.0;
		uint64_t normalDiffCount = 0u;
		double normalMaxAbs = 0.0;
		uint64_t uvDiffCount = 0u;
		double uvMaxAbs = 0.0;
		uint64_t indexDiffCount = 0u;
	};

	const ICPUPolygonGeometry::SDataView* findUvView(const ICPUPolygonGeometry* geo) const
	{
		if (!geo)
			return nullptr;
		for (const auto& view : geo->getAuxAttributeViews())
		{
			if (!view)
				continue;
			const auto channels = getFormatChannelCount(view.composed.format);
			if (channels >= 2u)
				return &view;
		}
		return nullptr;
	}

	bool compareGeometry(const ICPUPolygonGeometry* a, const ICPUPolygonGeometry* b, const double tol, GeometryCompareResult& out) const
	{
		if (!a || !b)
			return false;

		const auto& posA = a->getPositionView();
		const auto& posB = b->getPositionView();
		if (!posA || !posB)
			return false;

		out.vertexCountA = posA.getElementCount();
		out.vertexCountB = posB.getElementCount();
		if (out.vertexCountA != out.vertexCountB)
			return false;

		auto compareVec = [&](const ICPUPolygonGeometry::SDataView& viewA, const ICPUPolygonGeometry::SDataView& viewB, const uint32_t components, uint64_t& diffCount, double& maxAbs)->bool
		{
			hlsl::float32_t4 va = {};
			hlsl::float32_t4 vb = {};
			for (uint64_t i = 0; i < out.vertexCountA; ++i)
			{
				if (!viewA.decodeElement(i, va) || !viewB.decodeElement(i, vb))
					return false;
				const float* aVals = &va.x;
				const float* bVals = &vb.x;
				for (uint32_t c = 0; c < components; ++c)
				{
					const double diff = std::abs(static_cast<double>(aVals[c]) - static_cast<double>(bVals[c]));
					if (diff > maxAbs)
						maxAbs = diff;
					if (diff > tol)
						++diffCount;
				}
			}
			return true;
		};

		if (!compareVec(posA, posB, 3u, out.posDiffCount, out.posMaxAbs))
			return false;

		const auto& normalA = a->getNormalView();
		const auto& normalB = b->getNormalView();
		out.hasNormalA = static_cast<bool>(normalA);
		out.hasNormalB = static_cast<bool>(normalB);
		if (out.hasNormalA != out.hasNormalB)
			return false;
		if (out.hasNormalA)
			if (!compareVec(normalA, normalB, 3u, out.normalDiffCount, out.normalMaxAbs))
				return false;

		const auto* uvA = findUvView(a);
		const auto* uvB = findUvView(b);
		out.hasUvA = uvA != nullptr;
		out.hasUvB = uvB != nullptr;
		if (out.hasUvA != out.hasUvB)
			return false;
		if (out.hasUvA)
			if (!compareVec(*uvA, *uvB, 2u, out.uvDiffCount, out.uvMaxAbs))
				return false;

		const auto& idxA = a->getIndexView();
		const auto& idxB = b->getIndexView();
		out.indexCountA = idxA ? idxA.getElementCount() : out.vertexCountA;
		out.indexCountB = idxB ? idxB.getElementCount() : out.vertexCountB;
		if (out.indexCountA != out.indexCountB)
			return false;

		auto getIndex = [&](const ICPUPolygonGeometry::SDataView& view, const uint64_t ix)->uint32_t
		{
			const void* src = view.getPointer();
			if (!src)
				return 0u;
			if (view.composed.format == EF_R32_UINT)
				return reinterpret_cast<const uint32_t*>(src)[ix];
			if (view.composed.format == EF_R16_UINT)
				return static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(src)[ix]);
			return 0u;
		};

		for (uint64_t i = 0; i < out.indexCountA; ++i)
		{
			const uint32_t aIdx = idxA ? getIndex(idxA, i) : static_cast<uint32_t>(i);
			const uint32_t bIdx = idxB ? getIndex(idxB, i) : static_cast<uint32_t>(i);
			if (aIdx != bIdx)
				++out.indexDiffCount;
		}

		return out.posDiffCount == 0u && out.normalDiffCount == 0u && out.uvDiffCount == 0u && out.indexDiffCount == 0u;
	}

	bool validateWrittenAsset(const system::path& path)
	{
		if (!std::filesystem::exists(path))
			return false;

		m_assetMgr->clearAllAssetCache();

		IAssetLoader::SAssetLoadParams params = {};
		params.logger = m_logger.get();
		auto asset = m_assetMgr->getAsset(path.string(), params);
		if (asset.getContents().empty())
			return false;

		core::vector<smart_refctd_ptr<const ICPUPolygonGeometry>> geometries;
		switch (asset.getAssetType())
		{
		case IAsset::E_TYPE::ET_GEOMETRY:
			for (const auto& item : asset.getContents())
				if (auto polyGeo = IAsset::castDown<ICPUPolygonGeometry>(item); polyGeo)
					geometries.push_back(polyGeo);
			break;
		default:
			return false;
		}
		return !geometries.empty();
	}

	bool captureScreenshot(const system::path& path, core::smart_refctd_ptr<asset::ICPUImageView>& outImage)
	{
		if (!m_device || !m_surface || !m_assetMgr)
			return false;

		m_device->waitIdle();

		auto* scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
		auto* fb = scRes ? scRes->getFramebuffer(device_base_t::getCurrentAcquire().imageIndex) : nullptr;
		if (!fb)
			return false;

		auto colorView = fb->getCreationParameters().colorAttachments[0u];
		if (!colorView)
			return false;

		auto cpuView = ext::ScreenShot::createScreenShot(
			m_device.get(),
			getGraphicsQueue(),
			nullptr,
			colorView.get(),
			asset::ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT,
			asset::IImage::LAYOUT::PRESENT_SRC);
		if (!cpuView)
			return false;

		if (!path.empty())
			std::filesystem::create_directories(path.parent_path());

		IAssetWriter::SAssetWriteParams params(cpuView.get());
		if (!m_assetMgr->writeAsset(path.string(), params))
			return false;

		outImage = cpuView;
		return true;
	}

	bool appendGeometriesFromBundle(const asset::SAssetBundle& bundle, core::vector<smart_refctd_ptr<const ICPUPolygonGeometry>>& out) const
	{
		if (bundle.getContents().empty())
			return false;

		switch (bundle.getAssetType())
		{
		case IAsset::E_TYPE::ET_GEOMETRY:
			for (const auto& item : bundle.getContents())
			{
				if (auto polyGeo = IAsset::castDown<ICPUPolygonGeometry>(item); polyGeo)
					out.push_back(polyGeo);
			}
			break;
		case IAsset::E_TYPE::ET_GEOMETRY_COLLECTION:
			for (const auto& item : bundle.getContents())
			{
				auto collection = IAsset::castDown<ICPUGeometryCollection>(item);
				if (!collection)
					continue;
				auto* refs = collection->getGeometries();
				if (!refs)
					continue;
				for (const auto& ref : *refs)
				{
					if (!ref.geometry)
						continue;
					if (ref.geometry->getPrimitiveType() != IGeometryBase::EPrimitiveType::Polygon)
						continue;
					auto poly = core::smart_refctd_ptr_static_cast<ICPUPolygonGeometry>(ref.geometry);
					if (poly)
						out.push_back(poly);
				}
			}
			break;
		default:
			return false;
		}

		return !out.empty();
	}

	bool compareImages(const asset::ICPUImageView* a, const asset::ICPUImageView* b, uint64_t& diffCount, uint8_t& maxDiff)
	{
		diffCount = 0u;
		maxDiff = 0u;
		if (!a || !b)
			return false;

		const auto* imgA = a->getCreationParameters().image.get();
		const auto* imgB = b->getCreationParameters().image.get();
		if (!imgA || !imgB)
			return false;

		const auto paramsA = imgA->getCreationParameters();
		const auto paramsB = imgB->getCreationParameters();
		if (paramsA.format != paramsB.format)
			return false;
		if (paramsA.extent != paramsB.extent)
			return false;

		const auto* bufA = imgA->getBuffer();
		const auto* bufB = imgB->getBuffer();
		if (!bufA || !bufB)
			return false;

		const size_t sizeA = bufA->getSize();
		if (sizeA != bufB->getSize())
			return false;

		const auto* dataA = static_cast<const uint8_t*>(bufA->getPointer());
		const auto* dataB = static_cast<const uint8_t*>(bufB->getPointer());
		if (!dataA || !dataB)
			return false;

		for (size_t i = 0; i < sizeA; ++i)
		{
			const uint8_t va = dataA[i];
			const uint8_t vb = dataB[i];
			const uint8_t diff = va > vb ? static_cast<uint8_t>(va - vb) : static_cast<uint8_t>(vb - va);
			if (diff)
			{
				++diffCount;
				if (diff > maxDiff)
					maxDiff = diff;
			}
		}

		return true;
	}

	void advanceCase()
	{
		if (m_runMode == RunMode::Interactive || m_cases.empty())
			return;
		if (isRowViewActive())
			return;

		const uint32_t frameLimit = m_runMode == RunMode::CI ? CiFramesBeforeCapture : NonCiFramesPerCase;
		++m_phaseFrameCounter;
		if (m_phaseFrameCounter < frameLimit)
			return;

		if (m_phase == Phase::RenderOriginal)
		{
			if (!captureScreenshot(m_loadedScreenshotPath, m_loadedScreenshot))
				failExit("Failed to capture loaded screenshot.");

			if (m_saveGeom)
			{
				if (!m_currentCpuGeom)
					failExit("No geometry to write.");
				if (!writeGeometry(m_currentCpuGeom, m_writtenPath.string()))
					failExit("Geometry write failed.");
			}

			if (m_runMode == RunMode::CI)
			{
				if (!loadModel(m_writtenPath, false, false))
					failExit("Failed to load written asset %s.", m_writtenPath.string().c_str());
				if (!m_currentCpuGeom)
					failExit("Written geometry missing.");
				m_phase = Phase::RenderWritten;
				m_phaseFrameCounter = 0u;
				return;
			}

			if (m_saveGeom)
			{
				if (!validateWrittenAsset(m_writtenPath))
					failExit("Failed to load written asset %s.", m_writtenPath.string().c_str());
			}

			advanceToNextCase();
			return;
		}

		if (m_phase == Phase::RenderWritten)
		{
			if (!captureScreenshot(m_writtenScreenshotPath, m_writtenScreenshot))
				failExit("Failed to capture written screenshot.");

			if (m_hasReferenceGeometryHash)
			{
				const auto writtenHash = hashGeometry(m_currentCpuGeom.get());
				if (writtenHash != m_referenceGeometryHash)
				{
					if (m_hasReferenceGeometry)
					{
						GeometryCompareResult diff = {};
						const double tol = 1e-5;
						const bool compareOk = compareGeometry(m_referenceCpuGeom.get(), m_currentCpuGeom.get(), tol, diff);
						m_logger->log("Geometry hash mismatch for %s. CompareOk(%d) Vtx(%llu vs %llu) Idx(%llu vs %llu) PosDiff(%llu max %.8f) NDiff(%llu max %.8f) UvDiff(%llu max %.8f) IdxDiff(%llu) Normals(%d/%d) UV(%d/%d)",
							ILogger::ELL_ERROR,
							m_caseName.c_str(),
							compareOk ? 1 : 0,
							static_cast<unsigned long long>(diff.vertexCountA),
							static_cast<unsigned long long>(diff.vertexCountB),
							static_cast<unsigned long long>(diff.indexCountA),
							static_cast<unsigned long long>(diff.indexCountB),
							static_cast<unsigned long long>(diff.posDiffCount),
							diff.posMaxAbs,
							static_cast<unsigned long long>(diff.normalDiffCount),
							diff.normalMaxAbs,
							static_cast<unsigned long long>(diff.uvDiffCount),
							diff.uvMaxAbs,
							static_cast<unsigned long long>(diff.indexDiffCount),
							diff.hasNormalA ? 1 : 0,
							diff.hasNormalB ? 1 : 0,
							diff.hasUvA ? 1 : 0,
							diff.hasUvB ? 1 : 0);
					}
					failExit("Geometry hash mismatch for %s.", m_caseName.c_str());
				}
			}

			if (m_hasReferenceGeometry)
			{
				GeometryCompareResult diff = {};
				const double tol = 1e-5;
				if (!compareGeometry(m_referenceCpuGeom.get(), m_currentCpuGeom.get(), tol, diff))
				{
					m_logger->log("Geometry compare failed for %s. Vtx(%llu vs %llu) Idx(%llu vs %llu) PosDiff(%llu max %.8f) NDiff(%llu max %.8f) UvDiff(%llu max %.8f) IdxDiff(%llu) Normals(%d/%d) UV(%d/%d)",
						ILogger::ELL_ERROR,
						m_caseName.c_str(),
						static_cast<unsigned long long>(diff.vertexCountA),
						static_cast<unsigned long long>(diff.vertexCountB),
						static_cast<unsigned long long>(diff.indexCountA),
						static_cast<unsigned long long>(diff.indexCountB),
						static_cast<unsigned long long>(diff.posDiffCount),
						diff.posMaxAbs,
						static_cast<unsigned long long>(diff.normalDiffCount),
						diff.normalMaxAbs,
						static_cast<unsigned long long>(diff.uvDiffCount),
						diff.uvMaxAbs,
						static_cast<unsigned long long>(diff.indexDiffCount),
						diff.hasNormalA ? 1 : 0,
						diff.hasNormalB ? 1 : 0,
						diff.hasUvA ? 1 : 0,
						diff.hasUvB ? 1 : 0);
					failExit("Geometry compare failed for %s.", m_caseName.c_str());
				}
			}

			uint64_t diffCount = 0u;
			uint8_t maxDiff = 0u;
			if (!compareImages(m_loadedScreenshot.get(), m_writtenScreenshot.get(), diffCount, maxDiff))
				failExit("Image compare failed for %s.", m_caseName.c_str());
			if (diffCount > MaxImageDiffBytes || maxDiff > MaxImageDiffValue)
				failExit("Image diff detected for %s. Bytes: %llu MaxDiff: %u", m_caseName.c_str(), static_cast<unsigned long long>(diffCount), maxDiff);
			if (diffCount != 0u)
				m_logger->log("Image diff within tolerance for %s. Bytes: %llu MaxDiff: %u", ILogger::ELL_WARNING, m_caseName.c_str(), static_cast<unsigned long long>(diffCount), maxDiff);

			advanceToNextCase();
		}
	}

	// Maximum frames which can be simultaneously submitted, used to cycle through our per-frame resources like command buffers
	constexpr static inline uint32_t MaxFramesInFlight = 3u;
	constexpr static inline uint32_t CiFramesBeforeCapture = 10u;
	constexpr static inline uint32_t NonCiFramesPerCase = 120u;
	constexpr static inline uint32_t RowViewFramesBeforeCapture = 10u;
	constexpr static inline uint64_t MaxImageDiffBytes = 16u;
	constexpr static inline uint8_t MaxImageDiffValue = 1u;
	//
	smart_refctd_ptr<CSimpleDebugRenderer> m_renderer;
	//
	smart_refctd_ptr<ISemaphore> m_semaphore;
	uint64_t m_realFrameIx = 0;
	std::array<smart_refctd_ptr<IGPUCommandBuffer>, MaxFramesInFlight> m_cmdBufs;
	//
	InputSystem::ChannelReader<IMouseEventChannel> mouse;
	InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;
	//
	Camera camera = Camera(core::vectorSIMDf(0, 0, 0), core::vectorSIMDf(0, 0, 0), core::matrix4SIMD());
	// mutables
	std::string m_modelPath;
	std::string m_caseName;

	DrawBoundingBoxMode m_drawBBMode = DBBM_AABB;
#ifdef NBL_BUILD_DEBUG_DRAW
		smart_refctd_ptr<ext::debug_draw::DrawAABB> m_drawAABB;
		std::vector<ext::debug_draw::InstanceData> m_aabbInstances;
		std::vector<ext::debug_draw::InstanceData> m_obbInstances;

#endif

	bool m_saveGeom = true;
	std::optional<const std::string> m_specifiedGeomSavePath;
	nbl::system::path m_saveGeomPrefixPath;
	nbl::system::path m_screenshotPrefixPath;
	nbl::system::path m_rowViewScreenshotPath;
	nbl::system::path m_testListPath;

	RunMode m_runMode = RunMode::Batch;
	Phase m_phase = Phase::RenderOriginal;
	uint32_t m_phaseFrameCounter = 0u;
	size_t m_caseIndex = 0u;
	core::vector<TestCase> m_cases;
	bool m_shouldQuit = false;

	nbl::system::path m_writtenPath;
	nbl::system::path m_loadedScreenshotPath;
	nbl::system::path m_writtenScreenshotPath;

	core::smart_refctd_ptr<const ICPUPolygonGeometry> m_currentCpuGeom;
	core::smart_refctd_ptr<const ICPUPolygonGeometry> m_referenceCpuGeom;
	bool m_hasReferenceGeometry = false;
	core::blake3_hash_t m_referenceGeometryHash = {};
	bool m_hasReferenceGeometryHash = false;

	core::smart_refctd_ptr<asset::ICPUImageView> m_loadedScreenshot;
	core::smart_refctd_ptr<asset::ICPUImageView> m_writtenScreenshot;

	std::optional<CameraState> m_referenceCamera;
};

NBL_MAIN_FUNC(MeshLoadersApp)
