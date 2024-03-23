#ifndef __NBL_COMMON_API_H_INCLUDED__
#define __NBL_COMMON_API_H_INCLUDED__

#include <nabla.h>

#include "MonoSystemMonoLoggerApplication.hpp"

#include "nbl/ui/CGraphicalApplicationAndroid.h"
#include "nbl/ui/CWindowManagerAndroid.h"

// TODO: see TODO below
// TODO: make these include themselves via `nabla.h`

#include "nbl/video/utilities/SPhysicalDeviceFilter.h"

#if 0
class CommonAPI
{
	CommonAPI() = delete;
public:		
	class CommonAPIEventCallback;

	class InputSystem : public nbl::core::IReferenceCounted
	{
		public:
			template <class ChannelType>
			struct Channels
			{
				nbl::core::mutex lock;
				std::condition_variable added;
				nbl::core::vector<nbl::core::smart_refctd_ptr<ChannelType>> channels;
				nbl::core::vector<std::chrono::microseconds> timeStamps;
				uint32_t defaultChannelIndex = 0;
			};
			// TODO: move to "nbl/ui/InputEventChannel.h" once the interface of this utility struct matures, also maybe rename to `Consumer` ?
			template <class ChannelType>
			struct ChannelReader
			{
				template<typename F>
				inline void consumeEvents(F&& processFunc, nbl::system::logger_opt_ptr logger=nullptr)
				{
					auto events = channel->getEvents();
					const auto frontBufferCapacity = channel->getFrontBufferCapacity();
					if (events.size()>consumedCounter+frontBufferCapacity)
					{
						logger.log(
							"Detected overflow, %d unconsumed events in channel of size %d!",
							nbl::system::ILogger::ELL_ERROR,events.size()-consumedCounter,frontBufferCapacity
						);
						consumedCounter = events.size()-frontBufferCapacity;
					}
					typename ChannelType::range_t rng(events.begin() + consumedCounter, events.end());
					processFunc(rng);
					consumedCounter = events.size();
				}

				nbl::core::smart_refctd_ptr<ChannelType> channel = nullptr;
				uint64_t consumedCounter = 0ull;
			};
		
			InputSystem(nbl::system::logger_opt_smart_ptr&& logger) : m_logger(std::move(logger)) {}

			void getDefaultMouse(ChannelReader<nbl::ui::IMouseEventChannel>* reader)
			{
				getDefault(m_mouse,reader);
			}
			void getDefaultKeyboard(ChannelReader<nbl::ui::IKeyboardEventChannel>* reader)
			{
				getDefault(m_keyboard,reader);
			}
			template<class ChannelType>
			void add(Channels<ChannelType>& channels, nbl::core::smart_refctd_ptr<ChannelType>&& channel)
			{
				std::unique_lock lock(channels.lock);
				channels.channels.push_back(std::move(channel));
				
				using namespace std::chrono;
				auto timeStamp = duration_cast<microseconds>(steady_clock::now().time_since_epoch());
				channels.timeStamps.push_back(timeStamp);

				channels.added.notify_all();
			}
			template<class ChannelType>
			void remove(Channels<ChannelType>& channels, const ChannelType* channel)
			{
				std::unique_lock lock(channels.lock);

				auto to_remove_itr = std::find_if(
						channels.channels.begin(),channels.channels.end(),[channel](const auto& chan)->bool{return chan.get()==channel;}
				);

				auto index = std::distance(channels.channels.begin(), to_remove_itr);

				channels.timeStamps.erase(channels.timeStamps.begin() + index);
				channels.channels.erase(to_remove_itr);
			}
			template<class ChannelType>
			void getDefault(Channels<ChannelType>& channels, ChannelReader<ChannelType>* reader)
			{
				/*
				* TODO: Improve default device switching.
				* For nice results, we should actually make a multi-channel reader,
				* and then keep a consumed counter together with a last consumed event from each channel.
				* If there is considerable pause in events received by our current chosen channel or
				* we can detect some other channel of the same "compatible class" is producing more events,
				* Switch the channel choice, but prune away all events younger than the old default's consumption timestamp.
				* (Basically switch keyboards but dont try to process events older than the events you've processed from the old keyboard)
				*/
				
				std::unique_lock lock(channels.lock);
				while (channels.channels.empty())
				{
					m_logger.log("Waiting For Input Device to be connected...",nbl::system::ILogger::ELL_INFO);
					channels.added.wait(lock);
				}
				
				uint64_t consumedCounter = 0ull;

				using namespace std::chrono;
				constexpr long long DefaultChannelTimeoutInMicroSeconds = 100*1e3; // 100 mili-seconds
				auto nowTimeStamp = duration_cast<microseconds>(steady_clock::now().time_since_epoch());

				// Update Timestamp of all channels
				for(uint32_t ch = 0u; ch < channels.channels.size(); ++ch) {
					auto & channel = channels.channels[ch];
					auto & timeStamp = channels.timeStamps[ch];
					auto events = channel->getEvents();
					if(events.size() > 0) {
						auto lastEventTimeStamp = (*(events.end() - 1)).timeStamp; // last event timestamp
						timeStamp = lastEventTimeStamp;
					}
				}

				auto defaultIdx = channels.defaultChannelIndex;
				if(defaultIdx >= channels.channels.size()) {
					defaultIdx = 0;
				}
				auto defaultChannel = channels.channels[defaultIdx];
				auto defaultChannelEvents = defaultChannel->getEvents();
				auto timeDiff = (nowTimeStamp - channels.timeStamps[defaultIdx]).count();
				
				constexpr size_t RewindBackEvents = 50u;

				// If the current one hasn't been active for a while
				if(defaultChannel->empty()) {
					if(timeDiff > DefaultChannelTimeoutInMicroSeconds) {
						// Look for the most active channel (the channel which has got the most events recently)
						auto newDefaultIdx = defaultIdx;
						microseconds maxEventTimeStamp = microseconds(0);

						for(uint32_t chIdx = 0; chIdx < channels.channels.size(); ++chIdx) {
							if(defaultIdx != chIdx) 
							{
								auto channelTimeDiff = (nowTimeStamp - channels.timeStamps[chIdx]).count();
								// Check if was more recently active than the current most active
								if(channelTimeDiff < DefaultChannelTimeoutInMicroSeconds)
								{
									auto & channel = channels.channels[chIdx];
									auto channelEvents = channel->getEvents();
									auto channelEventSize = channelEvents.size();
									const auto frontBufferCapacity = channel->getFrontBufferCapacity();

									size_t rewindBack = std::min(RewindBackEvents, frontBufferCapacity);
									rewindBack = std::min(rewindBack, channelEventSize);

									auto oldEvent = *(channelEvents.end() - rewindBack);

									// Which oldEvent of channels are most recent.
									if(oldEvent.timeStamp > maxEventTimeStamp) {
										maxEventTimeStamp = oldEvent.timeStamp;
										newDefaultIdx = chIdx;
									}
								}
							}
						}

						if(defaultIdx != newDefaultIdx) {
							m_logger.log("Default InputChannel for ChannelType changed from %u to %u",nbl::system::ILogger::ELL_INFO, defaultIdx, newDefaultIdx);

							defaultIdx = newDefaultIdx;
							channels.defaultChannelIndex = newDefaultIdx;
							defaultChannel = channels.channels[newDefaultIdx];
							
							consumedCounter = defaultChannel->getEvents().size() - defaultChannel->getFrontBufferCapacity(); // to not get overflow in reader when consuming.
						}
					}
				}

				if (reader->channel==defaultChannel)
					return;

				reader->channel = defaultChannel;
				reader->consumedCounter = consumedCounter;
			}

			nbl::system::logger_opt_smart_ptr m_logger;
			Channels<nbl::ui::IMouseEventChannel> m_mouse;
			Channels<nbl::ui::IKeyboardEventChannel> m_keyboard;
	};

	class CommonAPIEventCallback : public nbl::ui::IWindow::IEventCallback
	{
	public:
		CommonAPIEventCallback(nbl::core::smart_refctd_ptr<InputSystem>&& inputSystem, nbl::system::logger_opt_smart_ptr&& logger) : m_inputSystem(std::move(inputSystem)), m_logger(std::move(logger)), m_gotWindowClosedMsg(false){}
		CommonAPIEventCallback() {}
		bool isWindowOpen() const {return !m_gotWindowClosedMsg;}
		void setLogger(nbl::system::logger_opt_smart_ptr& logger)
		{
			m_logger = logger;
		}
		void setInputSystem(nbl::core::smart_refctd_ptr<InputSystem>&& inputSystem)
		{
			m_inputSystem = std::move(inputSystem);
		}
	private:
		
		bool onWindowClosed_impl() override
		{
			m_logger.log("Window closed");
			m_gotWindowClosedMsg = true;
			return true;
		}

		void onMouseConnected_impl(nbl::core::smart_refctd_ptr<nbl::ui::IMouseEventChannel>&& mch) override
		{
			m_logger.log("A mouse %p has been connected", nbl::system::ILogger::ELL_INFO, mch.get());
			m_inputSystem.get()->add(m_inputSystem.get()->m_mouse,std::move(mch));
		}
		void onMouseDisconnected_impl(nbl::ui::IMouseEventChannel* mch) override
		{
			m_logger.log("A mouse %p has been disconnected", nbl::system::ILogger::ELL_INFO, mch);
			m_inputSystem.get()->remove(m_inputSystem.get()->m_mouse,mch);
		}
		void onKeyboardConnected_impl(nbl::core::smart_refctd_ptr<nbl::ui::IKeyboardEventChannel>&& kbch) override
		{
			m_logger.log("A keyboard %p has been connected", nbl::system::ILogger::ELL_INFO, kbch.get());
			m_inputSystem.get()->add(m_inputSystem.get()->m_keyboard,std::move(kbch));
		}
		void onKeyboardDisconnected_impl(nbl::ui::IKeyboardEventChannel* kbch) override
		{
			m_logger.log("A keyboard %p has been disconnected", nbl::system::ILogger::ELL_INFO, kbch);
			m_inputSystem.get()->remove(m_inputSystem.get()->m_keyboard,kbch);
		}

	private:
		nbl::core::smart_refctd_ptr<InputSystem> m_inputSystem = nullptr;
		nbl::system::logger_opt_smart_ptr m_logger = nullptr;
		bool m_gotWindowClosedMsg;
	};
	
	
	struct InitParams
	{		
		uint32_t framesInFlight = 5u;
		uint32_t windowWidth = 800u;
		uint32_t windowHeight = 600u;
		uint32_t swapchainImageCount = 3u;

		nbl::core::smart_refctd_ptr<nbl::ui::IWindow> window = nullptr;
		nbl::core::smart_refctd_ptr<nbl::ui::IWindowManager> windowManager = nullptr;
		nbl::core::smart_refctd_ptr<CommonAPIEventCallback> windowCb = nullptr;

	};

	struct InitOutput
	{
		
		static constexpr uint32_t MaxQueuesInFamily = 32;
		static constexpr uint32_t MaxFramesInFlight = 10u;
		static constexpr uint32_t MaxSwapChainImageCount = 4;

		nbl::core::smart_refctd_ptr<nbl::video::IAPIConnection> apiConnection;
		nbl::core::smart_refctd_ptr<nbl::video::ISurface> surface;
		nbl::core::smart_refctd_ptr<nbl::video::IUtilities> utilities;
		nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice;
		nbl::video::IPhysicalDevice* physicalDevice;
		std::array<std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool>, MaxFramesInFlight>, MaxQueuesCount> commandPools; // TODO: Multibuffer and reset the commandpools
		nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderToSwapchainRenderpass;
		nbl::core::smart_refctd_ptr<nbl::system::ISystem> system;
		nbl::core::smart_refctd_ptr<nbl::asset::CCompilerSet> compilerSet;
		nbl::core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
		nbl::video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
		nbl::core::smart_refctd_ptr<InputSystem> inputSystem;
		nbl::video::ISwapchain::SCreationParams swapchainCreationParams;
		nbl::core::smart_refctd_ptr<nbl::system::ILogger> logger;
		nbl::core::smart_refctd_ptr<nbl::ui::IWindowManager> windowManager;
	};

	template<bool gpuInit = true, class EventCallback = CommonAPIEventCallback, nbl::video::DeviceFeatureDependantClass... device_feature_dependant_t>
	static InitOutput Init(InitParams&& params)
	{
		using namespace nbl;
		using namespace nbl::video;

		InitOutput result;

		bool headlessCompute = params.isHeadlessCompute();

		result.inputSystem = nbl::core::make_smart_refctd_ptr<InputSystem>(system::logger_opt_smart_ptr(nbl::core::smart_refctd_ptr(result.logger)));
		result.assetManager = nbl::core::make_smart_refctd_ptr<nbl::asset::IAssetManager>(nbl::core::smart_refctd_ptr(result.system), nbl::core::smart_refctd_ptr(result.compilerSet)); // we should let user choose it?

		if (!headlessCompute)
		{
			if (!params.windowCb)
			{
				params.windowCb = nbl::core::make_smart_refctd_ptr<EventCallback>(nbl::core::smart_refctd_ptr(result.inputSystem), system::logger_opt_smart_ptr(nbl::core::smart_refctd_ptr(result.logger)));
			}
			params.windowCb->setInputSystem(nbl::core::smart_refctd_ptr(result.inputSystem));
			if (!params.window)
			{
				#ifdef _NBL_PLATFORM_WINDOWS_
					result.windowManager = ui::IWindowManagerWin32::create(); // on the Windows path
				#elif defined(_NBL_PLATFORM_LINUX_)
					result.windowManager = nbl::core::make_smart_refctd_ptr<nbl::ui::CWindowManagerX11>(); // on the Android path
				#else
					#error "Unsupported platform"
				#endif
				
				nbl::ui::IWindow::SCreationParams windowsCreationParams;
				windowsCreationParams.width = params.windowWidth;
				windowsCreationParams.height = params.windowHeight;
				windowsCreationParams.x = 64u;
				windowsCreationParams.y = 64u;
				windowsCreationParams.flags = nbl::ui::IWindow::ECF_RESIZABLE;
				windowsCreationParams.windowCaption = params.appName.data();
				windowsCreationParams.callback = params.windowCb;

				params.window = result.windowManager->createWindow(std::move(windowsCreationParams));
			}
			params.windowCb = nbl::core::smart_refctd_ptr<CommonAPIEventCallback>((CommonAPIEventCallback*) params.window->getEventCallback());
		}

		if constexpr (gpuInit)
		{
			performGpuInit<device_feature_dependant_t...>(params, result);
		}
		else
		{
			result.cpu2gpuParams.device = nullptr;
			result.cpu2gpuParams.finalQueueFamIx = 0u;
			result.cpu2gpuParams.limits = {};
			result.cpu2gpuParams.pipelineCache = nullptr;
			result.cpu2gpuParams.utilities = nullptr;
		}

		return result;
	}



	static std::pair<nbl::core::smart_refctd_ptr<nbl::video::IGPUImage>, nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView>> createEmpty2DTexture(
		const nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice>& device,
		uint32_t width,
		uint32_t height,
		nbl::asset::E_FORMAT format)
	{
		nbl::video::IGPUImage::SCreationParams gpu_image_params = {};
		gpu_image_params.mipLevels = 1;
		gpu_image_params.extent = { width, height, 1 };
		gpu_image_params.format = format;
		gpu_image_params.arrayLayers = 1u;
		gpu_image_params.type = nbl::asset::IImage::ET_2D;
		gpu_image_params.samples = nbl::asset::IImage::ESCF_1_BIT;
		gpu_image_params.flags = static_cast<nbl::asset::IImage::E_CREATE_FLAGS>(0u);
		nbl::core::smart_refctd_ptr image = device->createImage(std::move(gpu_image_params));
		auto imagereqs = image->getMemoryReqs();
		imagereqs.memoryTypeBits &= device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
		auto imageMem = device->allocate(imagereqs, image.get());

		nbl::video::IGPUImageView::SCreationParams creation_params = {};
		creation_params.format = image->getCreationParameters().format;
		creation_params.image = image;
		creation_params.viewType = nbl::video::IGPUImageView::ET_2D;
		creation_params.subresourceRange = { static_cast<nbl::asset::IImage::E_ASPECT_FLAGS>(0u), 0, 1, 0, 1 };
		creation_params.flags = static_cast<nbl::video::IGPUImageView::E_CREATE_FLAGS>(0u);
		nbl::core::smart_refctd_ptr image_view = device->createImageView(std::move(creation_params));
		return std::pair(image, image_view);
	}

protected:

	
	template<nbl::video::DeviceFeatureDependantClass... device_feature_dependant_t>
	static void performGpuInit(InitParams& params, InitOutput& result)
	{		
		(device_feature_dependant_t::enablePreferredFeatures(selectedPhysicalDevice->getFeatures(), params.physicalDeviceFilter.requiredFeatures),...);
		
		auto queuesInfo = extractPhysicalDeviceQueueInfos(selectedPhysicalDevice, result.surface, headlessCompute);
		
		// Fill QueueCreationParams
		constexpr uint32_t MaxQueuesInFamily = 32;
		float queuePriorities[MaxQueuesInFamily];
		std::fill(queuePriorities, queuePriorities + MaxQueuesInFamily, IGPUQueue::DEFAULT_QUEUE_PRIORITY);

		constexpr uint32_t MaxQueueFamilyCount = 4;
		nbl::video::ILogicalDevice::SQueueCreationParams qcp[MaxQueueFamilyCount] = {};

		uint32_t actualQueueParamsCount = 0u;

		uint32_t queuesIndexInFamily[InitOutput::EQT_COUNT];
		uint32_t presentQueueIndexInFamily = 0u;

		// TODO(Erfan): There is a much better way to get queue creation params.
		// Graphics Queue
		if (!headlessCompute)
		{
			uint32_t dedicatedQueuesInFamily = queuesInfo.graphics.dedicatedQueueCount;
			assert(dedicatedQueuesInFamily >= 1u);

			qcp[0].familyIndex = queuesInfo.graphics.index;
			qcp[0].count = dedicatedQueuesInFamily;
			qcp[0].flags = static_cast<nbl::video::IGPUQueue::E_CREATE_FLAGS>(0);
			qcp[0].priorities = queuePriorities;
			queuesIndexInFamily[InitOutput::EQT_GRAPHICS] = 0u;
			actualQueueParamsCount++;
		}

		// Compute Queue
		bool foundComputeInOtherFamily = false;
		for (uint32_t i = 0; i < actualQueueParamsCount; ++i)
		{
			auto& otherQcp = qcp[i];
			uint32_t dedicatedQueuesInFamily = queuesInfo.compute.dedicatedQueueCount;
			if (otherQcp.familyIndex == queuesInfo.compute.index)
			{
				if (dedicatedQueuesInFamily >= 1)
				{
					queuesIndexInFamily[InitOutput::EQT_COMPUTE] = otherQcp.count + 0u;
				}
				else
				{
					queuesIndexInFamily[InitOutput::EQT_COMPUTE] = 0u;
				}
				otherQcp.count += dedicatedQueuesInFamily;
				foundComputeInOtherFamily = true;
				break; // If works correctly no need to check other family indices as they are unique
			}
		}
		if (!foundComputeInOtherFamily)
		{
			uint32_t dedicatedQueuesInFamily = queuesInfo.compute.dedicatedQueueCount;
			assert(dedicatedQueuesInFamily == 1u);

			queuesIndexInFamily[InitOutput::EQT_COMPUTE] = 0u;

			auto& computeQcp = qcp[actualQueueParamsCount];
			computeQcp.familyIndex = queuesInfo.compute.index;
			computeQcp.count = dedicatedQueuesInFamily;
			computeQcp.flags = static_cast<nbl::video::IGPUQueue::E_CREATE_FLAGS>(0);
			computeQcp.priorities = queuePriorities;
			actualQueueParamsCount++;
		}

		// Transfer Queue
		bool foundTransferInOtherFamily = false;
		for (uint32_t i = 0; i < actualQueueParamsCount; ++i)
		{
			auto& otherQcp = qcp[i];
			uint32_t dedicatedQueuesInFamily = queuesInfo.transfer.dedicatedQueueCount;
			if (otherQcp.familyIndex == queuesInfo.transfer.index)
			{
				if (dedicatedQueuesInFamily >= 2u)
				{
					queuesIndexInFamily[InitOutput::EQT_TRANSFER_UP] = otherQcp.count + 0u;
					queuesIndexInFamily[InitOutput::EQT_TRANSFER_DOWN] = otherQcp.count + 1u;
				}
				else if (dedicatedQueuesInFamily >= 1u)
				{
					queuesIndexInFamily[InitOutput::EQT_TRANSFER_UP] = otherQcp.count + 0u;
					queuesIndexInFamily[InitOutput::EQT_TRANSFER_DOWN] = otherQcp.count + 0u;
				}
				else if (dedicatedQueuesInFamily == 0u)
				{
					queuesIndexInFamily[InitOutput::EQT_TRANSFER_UP] = 0u;
					queuesIndexInFamily[InitOutput::EQT_TRANSFER_DOWN] = 0u;
				}
				otherQcp.count += dedicatedQueuesInFamily;
				foundTransferInOtherFamily = true;
				break; // If works correctly no need to check other family indices as they are unique
			}
		}
		if (!foundTransferInOtherFamily)
		{
			uint32_t dedicatedQueuesInFamily = queuesInfo.transfer.dedicatedQueueCount;
			assert(dedicatedQueuesInFamily >= 1u);

			if (dedicatedQueuesInFamily >= 2u)
			{
				queuesIndexInFamily[InitOutput::EQT_TRANSFER_UP] = 0u;
				queuesIndexInFamily[InitOutput::EQT_TRANSFER_DOWN] = 1u;
			}
			else if (dedicatedQueuesInFamily >= 1u)
			{
				queuesIndexInFamily[InitOutput::EQT_TRANSFER_UP] = 0u;
				queuesIndexInFamily[InitOutput::EQT_TRANSFER_DOWN] = 0u;
			}
			else
			{
				assert(false);
			}

			auto& transferQcp = qcp[actualQueueParamsCount];
			transferQcp.familyIndex = queuesInfo.transfer.index;
			transferQcp.count = dedicatedQueuesInFamily;
			transferQcp.flags = static_cast<nbl::video::IGPUQueue::E_CREATE_FLAGS>(0);
			transferQcp.priorities = queuePriorities;
			actualQueueParamsCount++;
		}

		// Present Queue
		if (!headlessCompute)
		{
			bool foundPresentInOtherFamily = false;
			for (uint32_t i = 0; i < actualQueueParamsCount; ++i)
			{
				auto& otherQcp = qcp[i];
				if (otherQcp.familyIndex == queuesInfo.present.index)
				{
					if (otherQcp.familyIndex == queuesInfo.graphics.index)
					{
						presentQueueIndexInFamily = 0u;
					}
					else
					{
						uint32_t dedicatedQueuesInFamily = queuesInfo.present.dedicatedQueueCount;

						if (dedicatedQueuesInFamily >= 1u)
						{
							presentQueueIndexInFamily = otherQcp.count + 0u;
						}
						else if (dedicatedQueuesInFamily == 0u)
						{
							presentQueueIndexInFamily = 0u;
						}
						otherQcp.count += dedicatedQueuesInFamily;
					}
					foundPresentInOtherFamily = true;
					break; // If works correctly no need to check other family indices as they are unique
				}
			}
			if (!foundPresentInOtherFamily)
			{
				uint32_t dedicatedQueuesInFamily = queuesInfo.present.dedicatedQueueCount;
				assert(dedicatedQueuesInFamily == 1u);
				presentQueueIndexInFamily = 0u;

				auto& presentQcp = qcp[actualQueueParamsCount];
				presentQcp.familyIndex = queuesInfo.present.index;
				presentQcp.count = dedicatedQueuesInFamily;
				presentQcp.flags = static_cast<nbl::video::IGPUQueue::E_CREATE_FLAGS>(0);
				presentQcp.priorities = queuePriorities;
				actualQueueParamsCount++;
			}
		}


		nbl::video::ILogicalDevice::SCreationParams dev_params;
		dev_params.queueParamsCount = actualQueueParamsCount;
		dev_params.queueParams = qcp;
		dev_params.featuresToEnable = params.physicalDeviceFilter.requiredFeatures;
		dev_params.compilerSet = result.compilerSet;
		result.logicalDevice = selectedPhysicalDevice->createLogicalDevice(std::move(dev_params));

		result.utilities = nbl::core::make_smart_refctd_ptr<nbl::video::IUtilities>(nbl::core::smart_refctd_ptr(result.logicalDevice));

		if (!headlessCompute)
			result.queues[InitOutput::EQT_GRAPHICS] = result.logicalDevice->getQueue(queuesInfo.graphics.index, queuesIndexInFamily[InitOutput::EQT_GRAPHICS]);
		result.queues[InitOutput::EQT_COMPUTE] = result.logicalDevice->getQueue(queuesInfo.compute.index, queuesIndexInFamily[InitOutput::EQT_COMPUTE]);

		// TEMP_FIX
	#ifdef EXAMPLES_CAN_LIVE_WITHOUT_GRAPHICS_QUEUE
		result.queues[InitOutput::EQT_TRANSFER_UP] = result.logicalDevice->getQueue(queuesInfo.transfer.index, queuesIndexInFamily[EQT_TRANSFER_UP]);
		result.queues[InitOutput::EQT_TRANSFER_DOWN] = result.logicalDevice->getQueue(queuesInfo.transfer.index, queuesIndexInFamily[EQT_TRANSFER_DOWN]);
	#else
		if(queuesInfo.graphics.index != QueueFamilyProps::InvalidIndex)
			result.queues[InitOutput::EQT_COMPUTE] = result.logicalDevice->getQueue(queuesInfo.graphics.index, 0u);
		if (!headlessCompute)
		{
			result.queues[InitOutput::EQT_TRANSFER_UP] = result.logicalDevice->getQueue(queuesInfo.graphics.index, 0u);
			result.queues[InitOutput::EQT_TRANSFER_DOWN] = result.logicalDevice->getQueue(queuesInfo.graphics.index, 0u);
		}
		else
		{
			result.queues[InitOutput::EQT_TRANSFER_UP] = result.logicalDevice->getQueue(queuesInfo.compute.index, queuesIndexInFamily[InitOutput::EQT_COMPUTE]);
			result.queues[InitOutput::EQT_TRANSFER_DOWN] = result.logicalDevice->getQueue(queuesInfo.compute.index, queuesIndexInFamily[InitOutput::EQT_COMPUTE]);
		}
	#endif
		if (!headlessCompute)
		{
			result.swapchainCreationParams = computeSwapchainCreationParams(
				params.swapchainImageCount,
				result.logicalDevice,
				result.surface,
				params.swapchainImageUsage,
				params.acceptableSurfaceFormats, params.acceptableSurfaceFormatCount,
				params.acceptableColorPrimaries, params.acceptableColorPrimaryCount,
				params.acceptableEotfs, params.acceptableEotfCount,
				params.acceptablePresentModes, params.acceptablePresentModeCount,
				params.acceptableSurfaceTransforms, params.acceptableSurfaceTransformCount
			);

			nbl::asset::E_FORMAT swapChainFormat = result.swapchainCreationParams.surfaceFormat.format;
			result.renderToSwapchainRenderpass = createRenderpass(result.logicalDevice, swapChainFormat, params.depthFormat);
		}

		uint32_t commandPoolsToCreate = core::max(params.framesInFlight, 1u);
		for (uint32_t i = 0; i < InitOutput::EQT_COUNT; ++i)
		{
			const IGPUQueue* queue = result.queues[i];
			if (queue != nullptr)
			{
				for (size_t j = 0; j < commandPoolsToCreate; j++)
				{
					result.commandPools[i][j] = result.logicalDevice->createCommandPool(queue->getFamilyIndex(), IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);
					assert(result.commandPools[i][j]);
				}
			}
		}

		result.physicalDevice = selectedPhysicalDevice;

		uint32_t mainQueueFamilyIndex = (headlessCompute) ? queuesInfo.compute.index : queuesInfo.graphics.index;
		result.cpu2gpuParams.assetManager = result.assetManager.get();
		result.cpu2gpuParams.device = result.logicalDevice.get();
		result.cpu2gpuParams.finalQueueFamIx = mainQueueFamilyIndex;
		result.cpu2gpuParams.limits = result.physicalDevice->getLimits();
		result.cpu2gpuParams.pipelineCache = nullptr;
		result.cpu2gpuParams.utilities = result.utilities.get();

		result.cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_TRANSFER].queue = result.queues[InitOutput::EQT_TRANSFER_UP];
		result.cpu2gpuParams.perQueue[nbl::video::IGPUObjectFromAssetConverter::EQU_COMPUTE].queue = result.queues[InitOutput::EQT_COMPUTE];

		const uint32_t transferUpQueueFamIndex = result.queues[InitOutput::EQT_TRANSFER_UP]->getFamilyIndex();
		const uint32_t computeQueueFamIndex = result.queues[InitOutput::EQT_COMPUTE]->getFamilyIndex();

		auto pool_transfer = result.logicalDevice->createCommandPool(transferUpQueueFamIndex, IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);
		nbl::core::smart_refctd_ptr<IGPUCommandPool> pool_compute;
		if (transferUpQueueFamIndex == computeQueueFamIndex)
			pool_compute = pool_transfer;
		else
			pool_compute = result.logicalDevice->createCommandPool(result.queues[InitOutput::EQT_COMPUTE]->getFamilyIndex(), IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);

		nbl::core::smart_refctd_ptr<IGPUCommandBuffer> transferCmdBuffer;
		nbl::core::smart_refctd_ptr<IGPUCommandBuffer> computeCmdBuffer;

		result.logicalDevice->createCommandBuffers(pool_transfer.get(), IGPUCommandBuffer::EL_PRIMARY, 1u, &transferCmdBuffer);
		result.logicalDevice->createCommandBuffers(pool_compute.get(), IGPUCommandBuffer::EL_PRIMARY, 1u, &computeCmdBuffer);

		result.cpu2gpuParams.perQueue[IGPUObjectFromAssetConverter::EQU_TRANSFER].cmdbuf = transferCmdBuffer;
		result.cpu2gpuParams.perQueue[IGPUObjectFromAssetConverter::EQU_COMPUTE].cmdbuf = computeCmdBuffer;
	}
};

#endif

#endif
