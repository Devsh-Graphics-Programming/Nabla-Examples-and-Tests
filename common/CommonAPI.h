#define _NBL_STATIC_LIB_
#include <nabla.h>

// TODO: get these all included by the appropriate namespace headers!
#include "nbl/system/IApplicationFramework.h"
#include "nbl/ui/CGraphicalApplicationAndroid.h"
#include "nbl/ui/CWindowManagerAndroid.h"
#include "nbl/ui/IGraphicalApplicationFramework.h"
#if defined(_NBL_PLATFORM_WINDOWS_)
#include <nbl/system/CColoredStdoutLoggerWin32.h>
#elif defined(_NBL_PLATFORM_ANDROID_)
#include <nbl/system/CStdoutLoggerAndroid.h>
#endif
#include "nbl/system/CSystemAndroid.h"
#include "nbl/system/CSystemLinux.h"
#include "nbl/system/CSystemWin32.h"
// TODO: make these include themselves via `nabla.h`




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

	class ICommonAPIEventCallback : public nbl::ui::IWindow::IEventCallback
	{
	public:
		virtual void setLogger(nbl::system::logger_opt_smart_ptr& logger) = 0;
	};
	class CommonAPIEventCallback : public ICommonAPIEventCallback
	{
	public:
		CommonAPIEventCallback(nbl::core::smart_refctd_ptr<InputSystem>&& inputSystem, nbl::system::logger_opt_smart_ptr&& logger) : m_inputSystem(std::move(inputSystem)), m_logger(std::move(logger)), m_gotWindowClosedMsg(false){}
		CommonAPIEventCallback() {}
		bool isWindowOpen() const {return !m_gotWindowClosedMsg;}
		void setLogger(nbl::system::logger_opt_smart_ptr& logger) override
		{
			m_logger = logger;
		}
		void setInputSystem(nbl::core::smart_refctd_ptr<InputSystem>&& inputSystem)
		{
			m_inputSystem = std::move(inputSystem);
		}
	private:
		bool onWindowShown_impl() override
		{
			m_logger.log("Window Shown");
			return true;
		}
		bool onWindowHidden_impl() override
		{
			m_logger.log("Window hidden");
			return true;
		}
		bool onWindowMoved_impl(int32_t x, int32_t y) override
		{
			m_logger.log("Window window moved to { %d, %d }", nbl::system::ILogger::ELL_WARNING, x, y);
			return true;
		}
		bool onWindowResized_impl(uint32_t w, uint32_t h) override
		{
			m_logger.log("Window resized to { %u, %u }", nbl::system::ILogger::ELL_DEBUG, w, h);
			return true;
		}
		bool onWindowMinimized_impl() override
		{
			m_logger.log("Window minimized", nbl::system::ILogger::ELL_ERROR);
			return true;
		}
		bool onWindowMaximized_impl() override
		{
			m_logger.log("Window maximized", nbl::system::ILogger::ELL_PERFORMANCE);
			return true;
		}
		void onGainedMouseFocus_impl() override
		{
			m_logger.log("Window gained mouse focus", nbl::system::ILogger::ELL_INFO);
		}
		void onLostMouseFocus_impl() override
		{
			m_logger.log("Window lost mouse focus", nbl::system::ILogger::ELL_INFO);
		}
		void onGainedKeyboardFocus_impl() override
		{
			m_logger.log("Window gained keyboard focus", nbl::system::ILogger::ELL_INFO);
		}
		void onLostKeyboardFocus_impl() override
		{
			m_logger.log("Window lost keyboard focus", nbl::system::ILogger::ELL_INFO);
		}
		
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

	static nbl::core::smart_refctd_ptr<nbl::system::ISystem> createSystem()
	{
		using namespace nbl;
		using namespace system;
#ifdef _NBL_PLATFORM_WINDOWS_
		return nbl::core::make_smart_refctd_ptr<nbl::system::CSystemWin32>();
#elif defined(_NBL_PLATFORM_ANDROID_)
#if 0
		return nbl::core::make_smart_refctd_ptr<nbl::system::CSystemAndroid>(std::move(caller));
#endif
#endif
		return nullptr;

	}
	
	// Used to help with queue selection
	struct QueueFamilyProps
	{
		static constexpr uint32_t InvalidIndex = ~0u;
		uint32_t index					= InvalidIndex;
		uint32_t dedicatedQueueCount	= 0u;
		uint32_t score					= 0u;
		bool supportsGraphics			: 1;
		bool supportsCompute			: 1;
		bool supportsTransfer			: 1;
		bool supportsSparseBinding		: 1;
		bool supportsPresent			: 1;
		bool supportsProtected			: 1;
	};

	struct GPUInfo
	{
		std::vector<nbl::video::ISurface::SFormat> availableSurfaceFormats;
		nbl::video::ISurface::E_PRESENT_MODE availablePresentModes;
		nbl::video::ISurface::SCapabilities surfaceCapabilities;

		struct
		{
			QueueFamilyProps graphics;
			QueueFamilyProps compute;
			QueueFamilyProps transfer;
			QueueFamilyProps present;
		} queueFamilyProps;

		bool hasSurfaceCapabilities = false;
		bool isSwapChainSupported = false;
	};
	
	static std::vector<GPUInfo> extractGPUInfos(
		nbl::core::SRange<nbl::video::IPhysicalDevice* const> gpus,
		nbl::core::smart_refctd_ptr<nbl::video::ISurface> surface,
		const bool headlessCompute = false)
	{
		using namespace nbl;
		using namespace nbl::video;

		std::vector<GPUInfo> extractedInfos = std::vector<GPUInfo>(gpus.size());

		for (size_t i = 0ull; i < gpus.size(); ++i)
		{
			auto& extractedInfo = extractedInfos[i];
			extractedInfo = {};
			auto gpu = gpus.begin()[i];

			// Find queue family indices
			{
				const auto& queueFamilyProperties = gpu->getQueueFamilyProperties();

				std::vector<uint32_t> remainingQueueCounts = std::vector<uint32_t>(queueFamilyProperties.size(), 0u);
				
				for (uint32_t familyIndex = 0u; familyIndex < queueFamilyProperties.size(); ++familyIndex)
				{
					const auto& familyProperty = queueFamilyProperties.begin()[familyIndex];
					remainingQueueCounts[familyIndex] = familyProperty.queueCount;
				}

				// Select Graphics Queue Family Index
				if(!headlessCompute)
				{
					// Select Graphics Queue Family Index
					for (uint32_t familyIndex = 0u; familyIndex < queueFamilyProperties.size(); ++familyIndex)
					{
						const auto& familyProperty = queueFamilyProperties.begin()[familyIndex];
						auto& outFamilyProp = extractedInfo.queueFamilyProps;
					
						const uint32_t currentFamilyQueueCount = familyProperty.queueCount;
						if(currentFamilyQueueCount <= 0)
							continue;

						bool supportsPresent = surface && surface->isSupportedForPhysicalDevice(gpu, familyIndex);
						bool hasGraphicsFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_GRAPHICS_BIT).value != 0;
						bool hasComputeFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_COMPUTE_BIT).value != 0;
						bool hasTransferFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_TRANSFER_BIT).value != 0;
						bool hasSparseBindingFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_SPARSE_BINDING_BIT).value != 0;
						bool hasProtectedFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_PROTECTED_BIT).value != 0;

						const uint32_t remainingQueueCount = remainingQueueCounts[familyIndex];
						const bool hasEnoughQueues = remainingQueueCount >= 1u;
						
						/*
						* Examples:
						*	-> score is 0 for every queueFam with no Graphics support
						*	-> If both queue families !hasEnoughtQueues -> score will be equal but this doesn't/shouldn't happen -> there should be a queueFamily with "enoughQueues" for graphics.
						*	-> if both queue families hasEnoughQueues and have similar support for present and compute: Queue Family with more remainingQueueCount is preferred.
						*	-> if both queue families hasEnoughQueues with the same number of remainingQueueCount -> "QueueFamily with present and no compute" >>>> "QueueFamily with compute and no present"
						*	-> if both queue families hasEnoughQueues -> "QueueFamily with compute and no present and 16 remainingQueues" ==== "QueueFamily with present and no compute and 1 remaining Queue"
						*	-> if both queue families hasEnoughQueues -> "QueueFamily with present and compute and 1 remaining Queue" ==== "QueueFamily with no compute and no present and 34 remaining Queues xD"
						*/
						uint32_t score = 0u;
						if(hasGraphicsFlag) {
							score++;
							if(hasEnoughQueues) {
								score += 1u * remainingQueueCount;

								if(supportsPresent) 
								{
									score += 32u; // more important to have present than compute (presentSupport is larger in scoring to 16 extra compute queues)
								}

								if(hasComputeFlag) 
								{
									score += 1u * remainingQueueCount;
								}
							}	
						}

						if(score > outFamilyProp.graphics.score)
						{
							outFamilyProp.graphics.index = familyIndex;
							outFamilyProp.graphics.supportsGraphics = hasGraphicsFlag;
							outFamilyProp.graphics.supportsCompute = hasComputeFlag;
							outFamilyProp.graphics.supportsTransfer = true; // Reporting this is optional for Vk Graphics-Capable QueueFam, but Its support is guaranteed.
							outFamilyProp.graphics.supportsSparseBinding = hasSparseBindingFlag;
							outFamilyProp.graphics.supportsPresent = supportsPresent;
							outFamilyProp.graphics.supportsProtected = hasProtectedFlag;
							outFamilyProp.graphics.dedicatedQueueCount = 1u;
							outFamilyProp.graphics.score = score;
						}
					}
					assert(extractedInfo.queueFamilyProps.graphics.index != QueueFamilyProps::InvalidIndex);
					remainingQueueCounts[extractedInfo.queueFamilyProps.graphics.index] -= extractedInfo.queueFamilyProps.graphics.dedicatedQueueCount;
				}

				// Select Compute Queue Family Index
				for (uint32_t familyIndex = 0u; familyIndex < queueFamilyProperties.size(); ++familyIndex)
				{
					const auto& familyProperty = queueFamilyProperties.begin()[familyIndex];
					auto& outFamilyProp = extractedInfo.queueFamilyProps;
					
					const uint32_t currentFamilyQueueCount = familyProperty.queueCount;
					if(currentFamilyQueueCount <= 0)
						continue;

					bool supportsPresent = surface && surface->isSupportedForPhysicalDevice(gpu, familyIndex);
					bool hasGraphicsFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_GRAPHICS_BIT).value != 0;
					bool hasComputeFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_COMPUTE_BIT).value != 0;
					bool hasTransferFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_TRANSFER_BIT).value != 0;
					bool hasSparseBindingFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_SPARSE_BINDING_BIT).value != 0;
					bool hasProtectedFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_PROTECTED_BIT).value != 0;
					
					const uint32_t remainingQueueCount = remainingQueueCounts[familyIndex];
					const bool hasExtraQueues = remainingQueueCount >= 1u;

					/*
					* Examples:
					*	-> If both !hasEnoughExtraQueues: "queue family that supports graphics" >>>> "queue family that doesn't support graphics"
					*	-> If both queueFams supports Graphics and hasEnoughExtraQueues: "Graphics-capable QueueFamily equal to the selected Graphics QueueFam" >>>> "Any other Graphics-capable QueueFamily"
					*	-> If both support Graphics (not equal to graphicsQueueFamIndex): "queue family that hasEnoughExtraQueues" >>>> "queue family that !hasEnoughExtraQueues"
					*	-> If both support Graphics and hasEnoughExtraQueues (not equal to graphicsQueueFamIndex):  both are adequate enough, depends on the order of the queueFams.
					*	-> "Compute-capable QueueFam with hasEnoughExtraQueues" >>>> "Compute-capable QueueFam with graphics capability and ==graphicsQueueFamIdx with no extra dedicated queues"
					*/
					uint32_t score = 0u;
					if(hasComputeFlag) {
						score++;

						if(hasExtraQueues) {
							score += 3;
						}
						
						if(!headlessCompute && hasGraphicsFlag) {
							score++;
							if(familyIndex == outFamilyProp.graphics.index) {
								score++;
							}
						}
					}

					if(score > outFamilyProp.compute.score)
					{
						outFamilyProp.compute.index = familyIndex;
						outFamilyProp.compute.supportsGraphics = hasGraphicsFlag;
						outFamilyProp.compute.supportsCompute = hasComputeFlag;
						outFamilyProp.compute.supportsTransfer = true; // Reporting this is optional for Vk Compute-Capable QueueFam, but Its support is guaranteed.
						outFamilyProp.compute.supportsSparseBinding = hasSparseBindingFlag;
						outFamilyProp.compute.supportsPresent = supportsPresent;
						outFamilyProp.compute.supportsProtected = hasProtectedFlag;
						outFamilyProp.compute.dedicatedQueueCount = (hasExtraQueues) ? 1u : 0u;
						outFamilyProp.compute.score = score;
					}
				}
				assert(extractedInfo.queueFamilyProps.compute.index != QueueFamilyProps::InvalidIndex);
				remainingQueueCounts[extractedInfo.queueFamilyProps.compute.index] -= extractedInfo.queueFamilyProps.compute.dedicatedQueueCount;

				// Select Transfer Queue Family Index
				for (uint32_t familyIndex = 0u; familyIndex < queueFamilyProperties.size(); ++familyIndex)
				{
					const auto& familyProperty = queueFamilyProperties.begin()[familyIndex];
					auto& outFamilyProp = extractedInfo.queueFamilyProps;
					
					const uint32_t currentFamilyQueueCount = familyProperty.queueCount;
					if(currentFamilyQueueCount <= 0)
						continue;

					bool supportsPresent = surface && surface->isSupportedForPhysicalDevice(gpu, familyIndex);
					bool hasGraphicsFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_GRAPHICS_BIT).value != 0;
					bool hasComputeFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_COMPUTE_BIT).value != 0;
					bool hasTransferFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_TRANSFER_BIT).value != 0;
					bool hasSparseBindingFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_SPARSE_BINDING_BIT).value != 0;
					bool hasProtectedFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_PROTECTED_BIT).value != 0;

					const uint32_t extraQueueCount = nbl::core::min(remainingQueueCounts[familyIndex], 2u); // UP + DOWN
					const bool hasExtraQueues = extraQueueCount >= 1u;
					
					/*
					* Examples:
					*	-> score is 0 for every queueFam with no Transfer support
					*	-> If both have similar hasEnoughExtraQueues, compute and graphics support: the one with more remainingQueueCount is preferred
					*	-> If both support Transfer: "QueueFam with >=1 extra queues and graphics and compute support" >>>> (less probable)"QueueFam with no extra queues and transfer-only(no compute and graphics support)"
					*	-> If both support Transfer: "QueueFam with >=0 extra queues and only compute" >>>> "QueueFam with >=0 extra queues and only graphics"
					*/
					uint32_t score = 0u;
					if(hasTransferFlag) {
						score += 1u;

						uint32_t notHavingComputeScore = 1u;
						uint32_t notHavingGraphicsScore = 2u;

						if(hasExtraQueues) { // Having extra queues to have seperate up/down transfer queues is more important
							score += 4u * extraQueueCount;
							notHavingComputeScore *= extraQueueCount;
							notHavingGraphicsScore *= extraQueueCount;
						}
						
						if(!hasGraphicsFlag) {
							score += notHavingGraphicsScore;
						}

						if(!hasComputeFlag) {
							score += notHavingComputeScore;
						}

					}

					if(score > outFamilyProp.transfer.score)
					{
						outFamilyProp.transfer.index = familyIndex;
						outFamilyProp.transfer.supportsGraphics = hasGraphicsFlag;
						outFamilyProp.transfer.supportsCompute = hasComputeFlag;
						outFamilyProp.transfer.supportsTransfer = hasTransferFlag;
						outFamilyProp.transfer.supportsSparseBinding = hasSparseBindingFlag;
						outFamilyProp.transfer.supportsPresent = supportsPresent;
						outFamilyProp.transfer.supportsProtected = hasProtectedFlag;
						outFamilyProp.transfer.dedicatedQueueCount = extraQueueCount;
						outFamilyProp.transfer.score = score;
					}
				}
				assert(extractedInfo.queueFamilyProps.transfer.index != QueueFamilyProps::InvalidIndex);
				remainingQueueCounts[extractedInfo.queueFamilyProps.transfer.index] -= extractedInfo.queueFamilyProps.transfer.dedicatedQueueCount;

				// Select Present Queue Family Index
				if(!headlessCompute)
				{
					if(extractedInfo.queueFamilyProps.graphics.supportsPresent && extractedInfo.queueFamilyProps.graphics.index != QueueFamilyProps::InvalidIndex)
					{
						extractedInfo.queueFamilyProps.present = extractedInfo.queueFamilyProps.graphics;
						extractedInfo.queueFamilyProps.present.dedicatedQueueCount = 0u;
					}
					else
					{
						const uint32_t maxNeededQueueCountForPresent = 1u;
						for (uint32_t familyIndex = 0u; familyIndex < queueFamilyProperties.size(); ++familyIndex)
						{
							const auto& familyProperty = queueFamilyProperties.begin()[familyIndex];
							auto& outFamilyProp = extractedInfo.queueFamilyProps;
					
							const uint32_t currentFamilyQueueCount = familyProperty.queueCount;
							if(currentFamilyQueueCount <= 0)
								continue;

							bool supportsPresent = surface && surface->isSupportedForPhysicalDevice(gpu, familyIndex);
							bool hasGraphicsFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_GRAPHICS_BIT).value != 0;
							bool hasComputeFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_COMPUTE_BIT).value != 0;
							bool hasTransferFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_TRANSFER_BIT).value != 0;
							bool hasSparseBindingFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_SPARSE_BINDING_BIT).value != 0;
							bool hasProtectedFlag = (familyProperty.queueFlags & IPhysicalDevice::EQF_PROTECTED_BIT).value != 0;

							const uint32_t remainingQueueCount = remainingQueueCounts[familyIndex];
							const bool hasEnoughExtraQueues = remainingQueueCount >= 1u;
							
							/* this will only lead here if selected graphics queue can't support present
							* Examples:
							*	-> score is 0 for every queueFam with no Present support
							*	-> If both queue families support Present -> "graphics support is preferred rather than extra dedicated queues"
							*		-> graphics support is equal in scoring to 100 extra queues with no graphics support
							*	-> If both queue families !hasEnoughExtraQueues -> "graphics support is preferred"
							*	-> If both queue families hasEnoughExtraQueues and have similar support for graphics -> "queue family with more remainingQueueCount is preferred"
							*/
							uint32_t score = 0u;
							if(supportsPresent) {
								score += 1u;
								
								uint32_t graphicsSupportScore = 100u;
								if(hasEnoughExtraQueues) {
									score += 1u * remainingQueueCount;
									graphicsSupportScore *= remainingQueueCount;
								}

								if(hasGraphicsFlag) {
									score += graphicsSupportScore; // graphics support is larger in scoring than 100 extra queues with no graphics support
								}
							}

							if(score > outFamilyProp.present.score)
							{
								outFamilyProp.present.index = familyIndex;
								outFamilyProp.present.supportsGraphics = hasGraphicsFlag;
								outFamilyProp.present.supportsCompute = hasComputeFlag;
								outFamilyProp.present.supportsTransfer = hasTransferFlag;
								outFamilyProp.present.supportsSparseBinding = hasSparseBindingFlag;
								outFamilyProp.present.supportsPresent = supportsPresent;
								outFamilyProp.present.supportsProtected = hasProtectedFlag;
								outFamilyProp.present.dedicatedQueueCount = (hasEnoughExtraQueues) ? 1u : 0u;
								outFamilyProp.present.score = score;
							}
						}
					}
					assert(extractedInfo.queueFamilyProps.present.index != QueueFamilyProps::InvalidIndex);
					remainingQueueCounts[extractedInfo.queueFamilyProps.present.index] -= extractedInfo.queueFamilyProps.present.dedicatedQueueCount;
				}

				if(!headlessCompute)
					assert(extractedInfo.queueFamilyProps.graphics.supportsTransfer && "This shouldn't happen");
				assert(extractedInfo.queueFamilyProps.compute.supportsTransfer && "This shouldn't happen");
			}

			extractedInfo.isSwapChainSupported = gpu->isSwapchainSupported();

			// Check if the surface is adequate
			if(surface)
			{
				uint32_t surfaceFormatCount;
				surface->getAvailableFormatsForPhysicalDevice(gpu, surfaceFormatCount, nullptr);
				extractedInfo.availableSurfaceFormats = std::vector<nbl::video::ISurface::SFormat>(surfaceFormatCount);
				surface->getAvailableFormatsForPhysicalDevice(gpu, surfaceFormatCount, extractedInfo.availableSurfaceFormats.data());

				extractedInfo.availablePresentModes = surface->getAvailablePresentModesForPhysicalDevice(gpu);

				// TODO: @achal OpenGL shouldn't fail this
				extractedInfo.surfaceCapabilities = {};
				if (surface->getSurfaceCapabilitiesForPhysicalDevice(gpu, extractedInfo.surfaceCapabilities))
					extractedInfo.hasSurfaceCapabilities = true;
			}
		}

		return extractedInfos;
	}
	
	// TODO: also implement a function:findBestGPU
	// Returns an index into gpus info vector
	static uint32_t findSuitableGPU(const std::vector<GPUInfo>& extractedInfos, const bool headlessCompute)
	{
		uint32_t ret = ~0u;
		for(uint32_t i = 0; i < extractedInfos.size(); ++i)
		{
			bool isGPUSuitable = false;
			const auto& extractedInfo = extractedInfos[i];

			if(!headlessCompute)
			{
				if ((extractedInfo.queueFamilyProps.graphics.index != QueueFamilyProps::InvalidIndex) &&
					(extractedInfo.queueFamilyProps.compute.index != QueueFamilyProps::InvalidIndex) &&
					(extractedInfo.queueFamilyProps.transfer.index != QueueFamilyProps::InvalidIndex) &&
					(extractedInfo.queueFamilyProps.present.index != QueueFamilyProps::InvalidIndex))
					isGPUSuitable = true;
				
				if(extractedInfo.isSwapChainSupported == false)
					isGPUSuitable = false;

				if(extractedInfo.hasSurfaceCapabilities == false)
					isGPUSuitable = false;
			}
			else
			{
				if ((extractedInfo.queueFamilyProps.compute.index != QueueFamilyProps::InvalidIndex) &&
					(extractedInfo.queueFamilyProps.transfer.index != QueueFamilyProps::InvalidIndex))
					isGPUSuitable = true;
			}

			if(isGPUSuitable)
			{
				// find the first suitable GPU
				ret = i;
				break;
			}
		}

		if(ret == ~0u)
		{
			//_NBL_DEBUG_BREAK_IF(true);
			ret = 0;
		}

		return ret;
	}

	template <typename FeatureType>
	struct SFeatureRequest
	{
		uint32_t count = 0u;
		FeatureType* features = nullptr;
	};

	struct InitParams {
		std::string_view appName;
		nbl::video::E_API_TYPE apiType = nbl::video::EAT_VULKAN;
		
		uint32_t framesInFlight = 5u;
		uint32_t windowWidth = 800u;
		uint32_t windowHeight = 600u;
		uint32_t scImageCount = 3u;

		SFeatureRequest<nbl::video::IAPIConnection::E_FEATURE> requiredInstanceFeatures = {};
		SFeatureRequest<nbl::video::IAPIConnection::E_FEATURE> optionalInstanceFeatures = {};
		SFeatureRequest<nbl::video::ILogicalDevice::E_FEATURE> requiredDeviceFeatures = {};
		SFeatureRequest<nbl::video::ILogicalDevice::E_FEATURE> optionalDeviceFeatures = {};
		
		nbl::asset::IImage::E_USAGE_FLAGS swapchainImageUsage = nbl::asset::IImage::E_USAGE_FLAGS::EUF_COLOR_ATTACHMENT_BIT;

		constexpr static inline std::array<nbl::asset::E_FORMAT, 4> defaultAcceptableSurfaceFormats = { nbl::asset::EF_R8G8B8A8_SRGB, nbl::asset::EF_R8G8B8A8_UNORM, nbl::asset::EF_B8G8R8A8_SRGB, nbl::asset::EF_B8G8R8A8_UNORM };
		constexpr static inline std::array<nbl::asset::E_COLOR_PRIMARIES, 1> defaultAcceptableColorPrimaries = { nbl::asset::ECP_SRGB };
		constexpr static inline std::array<nbl::asset::ELECTRO_OPTICAL_TRANSFER_FUNCTION, 1> defaultAcceptableEotfs = { nbl::asset::EOTF_sRGB };
		constexpr static inline std::array<nbl::video::ISurface::E_PRESENT_MODE, 1> defaultAcceptablePresentModes = { nbl::video::ISurface::EPM_FIFO_RELAXED };
		constexpr static inline std::array<nbl::video::ISurface::E_SURFACE_TRANSFORM_FLAGS, 2> defaultAcceptableSurfaceTransforms = { nbl::video::ISurface::EST_IDENTITY_BIT, nbl::video::ISurface::EST_HORIZONTAL_MIRROR_ROTATE_180_BIT };

		const nbl::asset::E_FORMAT* acceptableSurfaceFormats = &defaultAcceptableSurfaceFormats[0];
		uint32_t acceptableSurfaceFormatCount = defaultAcceptableSurfaceFormats.size();
		const nbl::asset::E_COLOR_PRIMARIES* acceptableColorPrimaries = &defaultAcceptableColorPrimaries[0];
		uint32_t acceptableColorPrimaryCount = defaultAcceptableColorPrimaries.size();
		const nbl::asset::ELECTRO_OPTICAL_TRANSFER_FUNCTION* acceptableEotfs = &defaultAcceptableEotfs[0];
		uint32_t acceptableEotfCount = defaultAcceptableEotfs.size();
		const nbl::video::ISurface::E_PRESENT_MODE* acceptablePresentModes = &defaultAcceptablePresentModes[0];
		uint32_t acceptablePresentModeCount = defaultAcceptablePresentModes.size();
		const nbl::video::ISurface::E_SURFACE_TRANSFORM_FLAGS* acceptableSurfaceTransforms = &defaultAcceptableSurfaceTransforms[0];
		uint32_t acceptableSurfaceTransformCount = defaultAcceptableSurfaceTransforms.size();

		nbl::asset::E_FORMAT depthFormat = nbl::asset::EF_UNKNOWN;

		nbl::core::smart_refctd_ptr<nbl::ui::IWindow> window = nullptr;
		nbl::core::smart_refctd_ptr<nbl::ui::IWindowManager> windowManager = nullptr;
		nbl::core::smart_refctd_ptr<CommonAPIEventCallback> windowCb = nullptr;
		nbl::core::bitflag<nbl::system::ILogger::E_LOG_LEVEL> logLevel =
			nbl::core::bitflag(nbl::system::ILogger::ELL_DEBUG) | nbl::system::ILogger::ELL_PERFORMANCE | nbl::system::ILogger::ELL_WARNING | nbl::system::ILogger::ELL_ERROR | nbl::system::ILogger::ELL_INFO;

		constexpr bool isHeadlessCompute()
		{
			return swapchainImageUsage == nbl::asset::IImage::E_USAGE_FLAGS(0);
		}
	};

	struct InitOutput
	{
		enum E_QUEUE_TYPE
		{
			EQT_GRAPHICS = 0,
			EQT_COMPUTE,
			EQT_TRANSFER_UP,
			EQT_TRANSFER_DOWN,
			EQT_COUNT
		};
		
		static constexpr uint32_t MaxQueuesInFamily = 32;
		static constexpr uint32_t MaxFramesInFlight = 10u;
		static constexpr uint32_t MaxQueuesCount = EQT_COUNT;
		static constexpr uint32_t MaxSwapChainImageCount = 4;

		nbl::core::smart_refctd_ptr<nbl::video::IAPIConnection> apiConnection;
		nbl::core::smart_refctd_ptr<nbl::video::ISurface> surface;
		nbl::core::smart_refctd_ptr<nbl::video::IUtilities> utilities;
		nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice;
		nbl::video::IPhysicalDevice* physicalDevice;
		std::array<nbl::video::IGPUQueue*, MaxQueuesCount> queues = { nullptr, nullptr, nullptr, nullptr };
		std::array<std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool>, MaxFramesInFlight>, MaxQueuesCount> commandPools; // TODO: Multibuffer and reset the commandpools
		nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderToSwapchainRenderpass;
		nbl::core::smart_refctd_ptr<nbl::system::ISystem> system;
		nbl::core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
		nbl::video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
		nbl::core::smart_refctd_ptr<InputSystem> inputSystem;
		nbl::video::ISwapchain::SCreationParams swapchainCreationParams;
		nbl::core::smart_refctd_ptr<nbl::system::ILogger> logger;
	};

	template<typename AppClassName>
	static void main(int argc, char** argv)
	{
#ifndef _NBL_PLATFORM_ANDROID_
		nbl::system::path CWD = nbl::system::path(argv[0]).parent_path().generic_string() + "/";
		nbl::system::path sharedInputCWD = CWD / "../../media/";
		nbl::system::path sharedOutputCWD = CWD / "../../tmp/";;
		nbl::system::path localInputCWD = CWD / "../assets";
		nbl::system::path localOutputCWD = CWD;
		auto app = nbl::core::make_smart_refctd_ptr<AppClassName>(localInputCWD, localOutputCWD, sharedInputCWD, sharedOutputCWD);

		for (size_t i = 0; i < argc; ++i)
			app->argv.push_back(std::string(argv[i]));

		app->onAppInitialized();
		while (app->keepRunning())
		{
			app->workLoopBody();
		}
		app->onAppTerminated();
#endif
	}
	
#ifdef _NBL_PLATFORM_ANDROID_
	static void recreateSurface(nbl::ui::CGraphicalApplicationAndroid* framework)
	{
		using namespace nbl;
		android_app* app = framework->getApp();
		auto apiConnection = framework->getAPIConnection();
		auto window = framework->getWindow();
		auto logicalDevice = framework->getLogicalDevice();
		auto surface = nbl::video::CSurfaceGLAndroid::create(nbl::core::smart_refctd_ptr<nbl::video::COpenGLESConnection>((nbl::video::COpenGLESConnection*)apiConnection), nbl::core::smart_refctd_ptr<nbl::ui::IWindowAndroid>(static_cast<nbl::ui::IWindowAndroid*>(window)));
		auto renderpass = framework->getRenderpass();
		nbl::asset::E_FORMAT depthFormat = framework->getDepthFormat();
		framework->setSurface(surface);
		uint32_t width = ANativeWindow_getWidth(app->window);
		uint32_t height = ANativeWindow_getHeight(app->window);
		uint32_t scImageCount = framework->getSwapchainImageCount();
		nbl::video::ISurface::SFormat requestedFormat;
		
		// Temporary to make previous examples work
		requestedFormat.format = nbl::asset::EF_R8G8B8A8_SRGB;
		requestedFormat.colorSpace.eotf = nbl::asset::EOTF_sRGB;
		requestedFormat.colorSpace.primary = nbl::asset::ECP_SRGB;
		
		auto gpus = apiConnection->getPhysicalDevices();
		assert(!gpus.empty());
		auto extractedInfos = extractGPUInfos(gpus, surface);
		auto suitableGPUIndex = findSuitableGPU(extractedInfos, true);
		auto gpu = gpus.begin()[suitableGPUIndex];
		const auto& gpuInfo = extractedInfos[suitableGPUIndex];
		
		auto swapchain = createSwapchain(nbl::video::EAT_OPENGL_ES,
			gpuInfo,
			scImageCount,
			width,
			height,
			nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice>(logicalDevice),
			surface,
			nbl::asset::IImage::E_USAGE_FLAGS::EUF_NONE,
			nbl::video::ISurface::EPM_FIFO_RELAXED,
			requestedFormat
			);
		auto fbo = [&]()
		{
			auto fbos = createFBOWithSwapchainImages(scImageCount, 
				width, 
				height, 
				nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice>(logicalDevice), 
				swapchain,
				nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass>(renderpass),
				depthFormat
			);
			
			std::vector<nbl::core::smart_refctd_ptr<video::IGPUFramebuffer>> data;
			for(auto it : fbos)
				data.push_back(it);
			
			return data;
		}();
		
		framework->setSwapchain(std::move(swapchain));
		framework->setFBOs(fbo);
	}
#endif

	template<bool gpuInit = true, class EventCallback = CommonAPIEventCallback>
	static InitOutput Init(InitParams&& params)
	{
		using namespace nbl;
		using namespace nbl::video;

		InitOutput result;

		bool headlessCompute = params.isHeadlessCompute();

#ifdef _NBL_PLATFORM_WINDOWS_
		result.system = createSystem();
#endif

#ifdef _NBL_PLATFORM_WINDOWS_
		result.logger = nbl::core::make_smart_refctd_ptr<system::CColoredStdoutLoggerWin32>(params.logLevel);
#elif defined(_NBL_PLATFORM_ANDROID_)
		result.logger = nbl::core::make_smart_refctd_ptr<system::CStdoutLoggerAndroid>(params.logLevel);
#endif

		result.inputSystem = nbl::core::make_smart_refctd_ptr<InputSystem>(system::logger_opt_smart_ptr(nbl::core::smart_refctd_ptr(result.logger)));
		result.assetManager = nbl::core::make_smart_refctd_ptr<nbl::asset::IAssetManager>(nbl::core::smart_refctd_ptr(result.system)); // we should let user choose it?

		if (!headlessCompute)
		{
#ifndef _NBL_PLATFORM_ANDROID_
			auto windowManager = nbl::core::make_smart_refctd_ptr<nbl::ui::CWindowManagerWin32>(); // should we store it in result?
			params.windowCb = nbl::core::make_smart_refctd_ptr<EventCallback>(nbl::core::smart_refctd_ptr(result.inputSystem), system::logger_opt_smart_ptr(nbl::core::smart_refctd_ptr(result.logger)));

			nbl::ui::IWindow::SCreationParams windowsCreationParams;
			windowsCreationParams.width = params.windowWidth;
			windowsCreationParams.height = params.windowHeight;
			windowsCreationParams.x = 64u;
			windowsCreationParams.y = 64u;
			windowsCreationParams.system = nbl::core::smart_refctd_ptr(result.system);
			windowsCreationParams.flags = nbl::ui::IWindow::ECF_NONE;
			windowsCreationParams.windowCaption = params.appName.data();
			windowsCreationParams.callback = params.windowCb;

			params.window = windowManager->createWindow(std::move(windowsCreationParams));
			params.windowCb->setInputSystem(nbl::core::smart_refctd_ptr(result.inputSystem));
#else
			params.windowCb = nbl::core::smart_refctd_ptr<EventCallback>((CommonAPIEventCallback*)params.window->getEventCallback());
			params.windowCb->setInputSystem(nbl::core::smart_refctd_ptr(result.inputSystem));
#endif
		}

		if constexpr (gpuInit)
		{
			if (params.apiType == EAT_VULKAN)
			{
				auto _apiConnection = nbl::video::CVulkanConnection::create(
					nbl::core::smart_refctd_ptr(result.system),
					0,
					params.appName.data(),
					params.requiredInstanceFeatures.count,
					params.requiredInstanceFeatures.features,
					params.optionalInstanceFeatures.count,
					params.optionalInstanceFeatures.features,
					nbl::core::smart_refctd_ptr(result.logger),
					true);

				if (!headlessCompute)
				{
#ifdef _NBL_PLATFORM_WINDOWS_
					result.surface = nbl::video::CSurfaceVulkanWin32::create(nbl::core::smart_refctd_ptr(_apiConnection), nbl::core::smart_refctd_ptr<nbl::ui::IWindowWin32>(static_cast<nbl::ui::IWindowWin32*>(params.window.get())));
#elif defined(_NBL_PLATFORM_ANDROID_)
					////result.surface = nbl::video::CSurfaceVulkanAndroid::create(nbl::core::smart_refctd_ptr(_apiConnection), nbl::core::smart_refctd_ptr<nbl::ui::IWindowAndroid>(static_cast<nbl::ui::IWindowAndroid*>(params.window.get())));
#endif
				}
				result.apiConnection = _apiConnection;
			}
			else if (params.apiType == EAT_OPENGL)
			{
				auto _apiConnection = nbl::video::COpenGLConnection::create(nbl::core::smart_refctd_ptr(result.system), 0, params.appName.data(), nbl::video::COpenGLDebugCallback(nbl::core::smart_refctd_ptr(result.logger)));

				if (!headlessCompute)
				{
#ifdef _NBL_PLATFORM_WINDOWS_
					result.surface = nbl::video::CSurfaceGLWin32::create(nbl::core::smart_refctd_ptr(_apiConnection), nbl::core::smart_refctd_ptr<nbl::ui::IWindowWin32>(static_cast<nbl::ui::IWindowWin32*>(params.window.get())));
#elif defined(_NBL_PLATFORM_ANDROID_)
					result.surface = nbl::video::CSurfaceGLAndroid::create(nbl::core::smart_refctd_ptr(_apiConnection), nbl::core::smart_refctd_ptr<nbl::ui::IWindowAndroid>(static_cast<nbl::ui::IWindowAndroid*>(params.window.get())));
#endif
				}

				result.apiConnection = _apiConnection;
			}
			else if (params.apiType	 == EAT_OPENGL_ES)
			{
				auto _apiConnection = nbl::video::COpenGLESConnection::create(nbl::core::smart_refctd_ptr(result.system), 0, params.appName.data(), nbl::video::COpenGLDebugCallback(nbl::core::smart_refctd_ptr(result.logger)));

				if (!headlessCompute)
				{
#ifdef _NBL_PLATFORM_WINDOWS_
					result.surface = nbl::video::CSurfaceGLWin32::create(nbl::core::smart_refctd_ptr(_apiConnection), nbl::core::smart_refctd_ptr<nbl::ui::IWindowWin32>(static_cast<nbl::ui::IWindowWin32*>(params.window.get())));
#elif defined(_NBL_PLATFORM_ANDROID_)
					result.surface = nbl::video::CSurfaceGLAndroid::create(nbl::core::smart_refctd_ptr(_apiConnection), nbl::core::smart_refctd_ptr<nbl::ui::IWindowAndroid>(static_cast<nbl::ui::IWindowAndroid*>(params.window.get())));
#endif
				}

				result.apiConnection = _apiConnection;
			}
			else
			{
				_NBL_TODO();
			}

			auto gpus = result.apiConnection->getPhysicalDevices();
			assert(!gpus.empty());
			auto extractedInfos = extractGPUInfos(gpus, result.surface, headlessCompute);
			auto suitableGPUIndex = findSuitableGPU(extractedInfos, headlessCompute);
			auto gpu = gpus.begin()[suitableGPUIndex];
			const auto& gpuInfo = extractedInfos[suitableGPUIndex];

			// Fill QueueCreationParams
			constexpr uint32_t MaxQueuesInFamily = 32;
			float queuePriorities[MaxQueuesInFamily];
			std::fill(queuePriorities, queuePriorities + MaxQueuesInFamily, IGPUQueue::DEFAULT_QUEUE_PRIORITY);

			constexpr uint32_t MaxQueueFamilyCount = 4;
			nbl::video::ILogicalDevice::SQueueCreationParams qcp[MaxQueueFamilyCount] = {};

			uint32_t actualQueueParamsCount = 0u;

			uint32_t queuesIndexInFamily[InitOutput::EQT_COUNT];
			uint32_t presentQueueIndexInFamily = 0u;

			// Graphics Queue
			if (!headlessCompute)
			{
				uint32_t dedicatedQueuesInFamily = gpuInfo.queueFamilyProps.graphics.dedicatedQueueCount;
				assert(dedicatedQueuesInFamily >= 1u);

				qcp[0].familyIndex = gpuInfo.queueFamilyProps.graphics.index;
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
				uint32_t dedicatedQueuesInFamily = gpuInfo.queueFamilyProps.compute.dedicatedQueueCount;
				if (otherQcp.familyIndex == gpuInfo.queueFamilyProps.compute.index)
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
				uint32_t dedicatedQueuesInFamily = gpuInfo.queueFamilyProps.compute.dedicatedQueueCount;
				assert(dedicatedQueuesInFamily == 1u);

				queuesIndexInFamily[InitOutput::EQT_COMPUTE] = 0u;

				auto& computeQcp = qcp[actualQueueParamsCount];
				computeQcp.familyIndex = gpuInfo.queueFamilyProps.compute.index;
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
				uint32_t dedicatedQueuesInFamily = gpuInfo.queueFamilyProps.transfer.dedicatedQueueCount;
				if (otherQcp.familyIndex == gpuInfo.queueFamilyProps.transfer.index)
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
				uint32_t dedicatedQueuesInFamily = gpuInfo.queueFamilyProps.transfer.dedicatedQueueCount;
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
				transferQcp.familyIndex = gpuInfo.queueFamilyProps.transfer.index;
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
					if (otherQcp.familyIndex == gpuInfo.queueFamilyProps.present.index)
					{
						if (otherQcp.familyIndex == gpuInfo.queueFamilyProps.graphics.index)
						{
							presentQueueIndexInFamily = 0u;
						}
						else
						{
							uint32_t dedicatedQueuesInFamily = gpuInfo.queueFamilyProps.present.dedicatedQueueCount;

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
					uint32_t dedicatedQueuesInFamily = gpuInfo.queueFamilyProps.present.dedicatedQueueCount;
					assert(dedicatedQueuesInFamily == 1u);
					presentQueueIndexInFamily = 0u;

					auto& presentQcp = qcp[actualQueueParamsCount];
					presentQcp.familyIndex = gpuInfo.queueFamilyProps.present.index;
					presentQcp.count = dedicatedQueuesInFamily;
					presentQcp.flags = static_cast<nbl::video::IGPUQueue::E_CREATE_FLAGS>(0);
					presentQcp.priorities = queuePriorities;
					actualQueueParamsCount++;
				}
			}

			nbl::video::ILogicalDevice::SCreationParams dev_params;
			dev_params.queueParamsCount = actualQueueParamsCount;
			dev_params.queueParams = qcp;
			dev_params.requiredFeatureCount = params.requiredDeviceFeatures.count;
			dev_params.requiredFeatures = params.requiredDeviceFeatures.features;
			dev_params.optionalFeatureCount = params.optionalDeviceFeatures.count;
			dev_params.optionalFeatures = params.optionalDeviceFeatures.features;
			result.logicalDevice = gpu->createLogicalDevice(dev_params);

			result.utilities = nbl::core::make_smart_refctd_ptr<nbl::video::IUtilities>(nbl::core::smart_refctd_ptr(result.logicalDevice));

			if (!headlessCompute)
				result.queues[InitOutput::EQT_GRAPHICS] = result.logicalDevice->getQueue(gpuInfo.queueFamilyProps.graphics.index, queuesIndexInFamily[InitOutput::EQT_GRAPHICS]);
			result.queues[InitOutput::EQT_COMPUTE] = result.logicalDevice->getQueue(gpuInfo.queueFamilyProps.compute.index, queuesIndexInFamily[InitOutput::EQT_COMPUTE]);

			// TEMP_FIX
#ifdef EXAMPLES_CAN_HANDLE_TRANSFER_WITHOUT_GRAPHICS 
			result.queues[InitOutput::EQT_TRANSFER_UP] = result.logicalDevice->getQueue(gpuInfo.queueFamilyProps.transfer.index, queuesIndexInFamily[EQT_TRANSFER_UP]);
			result.queues[InitOutput::EQT_TRANSFER_DOWN] = result.logicalDevice->getQueue(gpuInfo.queueFamilyProps.transfer.index, queuesIndexInFamily[EQT_TRANSFER_DOWN]);
#else
			if (!headlessCompute)
			{
				result.queues[InitOutput::EQT_TRANSFER_UP] = result.logicalDevice->getQueue(gpuInfo.queueFamilyProps.graphics.index, 0u);
				result.queues[InitOutput::EQT_TRANSFER_DOWN] = result.logicalDevice->getQueue(gpuInfo.queueFamilyProps.graphics.index, 0u);
			}
			else
			{
				result.queues[InitOutput::EQT_TRANSFER_UP] = result.logicalDevice->getQueue(gpuInfo.queueFamilyProps.compute.index, queuesIndexInFamily[InitOutput::EQT_COMPUTE]);
				result.queues[InitOutput::EQT_TRANSFER_DOWN] = result.logicalDevice->getQueue(gpuInfo.queueFamilyProps.compute.index, queuesIndexInFamily[InitOutput::EQT_COMPUTE]);
			}
#endif
			if (!headlessCompute)
			{

				result.swapchainCreationParams = computeSwapchainCreationParams(
					gpuInfo, 
					params.scImageCount, 
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
			for(uint32_t i = 0; i < InitOutput::EQT_COUNT; ++i)
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

			result.physicalDevice = gpu;

			uint32_t mainQueueFamilyIndex = (headlessCompute) ? gpuInfo.queueFamilyProps.compute.index : gpuInfo.queueFamilyProps.graphics.index;
			result.cpu2gpuParams.assetManager = result.assetManager.get();
			result.cpu2gpuParams.device = result.logicalDevice.get();
			result.cpu2gpuParams.finalQueueFamIx = mainQueueFamilyIndex;
			result.cpu2gpuParams.limits = result.physicalDevice->getLimits();
			result.cpu2gpuParams.pipelineCache = nullptr;
			result.cpu2gpuParams.sharingMode = nbl::asset::ESM_EXCLUSIVE;
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
		else
		{
			result.cpu2gpuParams.device = nullptr;
			result.cpu2gpuParams.finalQueueFamIx = 0u;
			result.cpu2gpuParams.limits = {};
			result.cpu2gpuParams.pipelineCache = nullptr;
			result.cpu2gpuParams.sharingMode = nbl::asset::ESM_EXCLUSIVE;
			result.cpu2gpuParams.utilities = nullptr;
		}

		return result;
	}
	
	static nbl::video::ISwapchain::SCreationParams computeSwapchainCreationParams(
		const GPUInfo& gpuInfo, uint32_t& imageCount,
		const nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice>& device,
		const nbl::core::smart_refctd_ptr<nbl::video::ISurface>& surface,
		nbl::asset::IImage::E_USAGE_FLAGS imageUsage,
		// Acceptable settings, ordered by preference.
		const nbl::asset::E_FORMAT* acceptableSurfaceFormats, uint32_t acceptableSurfaceFormatCount,
		const nbl::asset::E_COLOR_PRIMARIES* acceptableColorPrimaries, uint32_t acceptableColorPrimaryCount,
		const nbl::asset::ELECTRO_OPTICAL_TRANSFER_FUNCTION* acceptableEotfs, uint32_t acceptableEotfCount,
		const nbl::video::ISurface::E_PRESENT_MODE* acceptablePresentModes, uint32_t acceptablePresentModeCount,
		const nbl::video::ISurface::E_SURFACE_TRANSFORM_FLAGS* acceptableSurfaceTransforms, uint32_t acceptableSurfaceTransformsCount
	)
	{
		using namespace nbl;

		asset::E_SHARING_MODE imageSharingMode;
		if (gpuInfo.queueFamilyProps.graphics.index == gpuInfo.queueFamilyProps.present.index)
			imageSharingMode = asset::ESM_EXCLUSIVE;
		else
			imageSharingMode = asset::ESM_CONCURRENT;

		nbl::video::ISurface::SFormat surfaceFormat;
		nbl::video::ISurface::E_PRESENT_MODE presentMode = nbl::video::ISurface::EPM_UNKNOWN;
		nbl::video::ISurface::E_SURFACE_TRANSFORM_FLAGS surfaceTransform = nbl::video::ISurface::EST_FLAG_BITS_MAX_ENUM;

		if(device->getAPIType() == nbl::video::EAT_VULKAN)
		{
			// Deduce format features from imageUsage param
			nbl::video::IPhysicalDevice::SFormatImageUsage requiredFormatUsages = {};
			if (imageUsage & asset::IImage::EUF_STORAGE_BIT)
				requiredFormatUsages.storageImage = 1;

			nbl::video::ISurface::SCapabilities capabilities;
			surface->getSurfaceCapabilitiesForPhysicalDevice(device->getPhysicalDevice(), capabilities);

			for (uint32_t i = 0; i < acceptableSurfaceFormatCount; i++)
			{
				auto testSurfaceTransform = acceptableSurfaceTransforms[i];
				if (capabilities.currentTransform == testSurfaceTransform)
				{
					surfaceTransform = testSurfaceTransform;
					break;
				}
			}
			assert(surfaceTransform != nbl::video::ISurface::EST_FLAG_BITS_MAX_ENUM); // currentTransform must be supported in acceptableSurfaceTransforms

			auto availablePresentModes = surface->getAvailablePresentModesForPhysicalDevice(device->getPhysicalDevice());
			for (uint32_t i = 0; i < acceptablePresentModeCount; i++)
			{
				auto testPresentMode = acceptablePresentModes[i];
				if ((availablePresentModes & testPresentMode) == testPresentMode)
				{
					presentMode = testPresentMode;
					break;
				}
			}
			assert(presentMode != nbl::video::ISurface::EST_FLAG_BITS_MAX_ENUM);

			constexpr uint32_t MAX_SURFACE_FORMAT_COUNT = 1000u;
			uint32_t availableFormatCount;
			nbl::video::ISurface::SFormat availableFormats[MAX_SURFACE_FORMAT_COUNT];
			surface->getAvailableFormatsForPhysicalDevice(device->getPhysicalDevice(), availableFormatCount, availableFormats);

			for (uint32_t i = 0; i < availableFormatCount; ++i)
			{
				// TODO verify if acceptableSurfaceFormats, acceptableColorPrimaries & acceptableEotfs
				// allow for supportedFormat
				const auto& supportedFormat = availableFormats[i];
				if (true)
				{
					surfaceFormat = supportedFormat;
					break;
				}
			}
			// Require at least one of the acceptable options to be present
			assert(surfaceFormat.format != nbl::asset::EF_UNKNOWN &&
				surfaceFormat.colorSpace.primary != nbl::asset::ECP_COUNT &&
				surfaceFormat.colorSpace.eotf != nbl::asset::EOTF_UNKNOWN);
		}
		else
		{
			// Temporary path until OpenGL reports properly!
			surfaceFormat = nbl::video::ISurface::SFormat(acceptableSurfaceFormats[0], acceptableColorPrimaries[0], acceptableEotfs[0]);
			presentMode = nbl::video::ISurface::EPM_IMMEDIATE;
			surfaceTransform = nbl::video::ISurface::EST_HORIZONTAL_MIRROR_ROTATE_180_BIT;
		}

		nbl::video::ISwapchain::SCreationParams sc_params = {};
		sc_params.arrayLayers = 1u;
		sc_params.minImageCount = imageCount;
		sc_params.presentMode = presentMode;
		sc_params.imageUsage = imageUsage;
		sc_params.surface = surface;
		sc_params.imageSharingMode = imageSharingMode;
		sc_params.preTransform = surfaceTransform;
		sc_params.compositeAlpha = nbl::video::ISurface::ECA_OPAQUE_BIT;
		sc_params.surfaceFormat = surfaceFormat;

		return sc_params;
	}

	static bool createSwapchain(
		const nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice>& device,
		nbl::video::ISwapchain::SCreationParams& params,
		uint32_t width, uint32_t height,
		// nullptr for initial creation, old swapchain for eventual resizes
		nbl::core::smart_refctd_ptr<nbl::video::ISwapchain>& swapchain
	)
	{
		nbl::video::ISwapchain::SCreationParams paramsCp = params;
		paramsCp.width = width;
		paramsCp.height = height;
		paramsCp.oldSwapchain = swapchain;
		swapchain = device->createSwapchain(std::move(paramsCp));
		assert(swapchain);

		return true;
	}

	template<bool gpuInit = true, class EventCallback = CommonAPIEventCallback>
	static InitOutput InitWithDefaultExt(InitParams&& params)
	{
#ifndef _NBL_PLATFORM_ANDROID_
		nbl::video::IAPIConnection::E_FEATURE requiredFeatures_Instance[] = { nbl::video::IAPIConnection::EF_SURFACE };
		params.requiredInstanceFeatures.features = requiredFeatures_Instance;
		params.requiredInstanceFeatures.count = 1u;
		nbl::video::ILogicalDevice::E_FEATURE requiredFeatures_Device[] = { nbl::video::ILogicalDevice::EF_SWAPCHAIN };
		params.requiredDeviceFeatures.features = requiredFeatures_Device;
		params.requiredDeviceFeatures.count = 1u;
#endif
		return CommonAPI::Init<gpuInit, EventCallback>(std::move(params));
	}

	template<bool gpuInit = true, class EventCallback = CommonAPIEventCallback>
	static InitOutput InitWithRaytracingExt(InitParams&& params)
	{
#ifndef _NBL_PLATFORM_ANDROID_
		nbl::video::IAPIConnection::E_FEATURE requiredFeatures_Instance[] = { nbl::video::IAPIConnection::EF_SURFACE };
		params.requiredInstanceFeatures.features = requiredFeatures_Instance;
		params.requiredInstanceFeatures.count = 1u;

		nbl::video::ILogicalDevice::E_FEATURE requiredFeatures_Device[] =
		{
			nbl::video::ILogicalDevice::EF_SWAPCHAIN,
			nbl::video::ILogicalDevice::EF_ACCELERATION_STRUCTURE,
			nbl::video::ILogicalDevice::EF_RAY_QUERY
		};
		params.requiredDeviceFeatures.features = requiredFeatures_Device;
		params.requiredDeviceFeatures.count = 3u;
#endif
		return CommonAPI::Init<gpuInit, EventCallback>(std::move(params));
	}

	static nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> createRenderpass(const nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice>& device, nbl::asset::E_FORMAT colorAttachmentFormat, nbl::asset::E_FORMAT baseDepthFormat)
	{
		using namespace nbl;

		bool useDepth = baseDepthFormat != nbl::asset::EF_UNKNOWN;
		nbl::asset::E_FORMAT depthFormat = nbl::asset::EF_UNKNOWN;
		if (useDepth)
		{
			depthFormat = device->getPhysicalDevice()->promoteImageFormat(
				{ baseDepthFormat, nbl::video::IPhysicalDevice::SFormatImageUsage(nbl::asset::IImage::EUF_DEPTH_STENCIL_ATTACHMENT_BIT) },
				nbl::asset::IImage::ET_OPTIMAL
			);
			// TODO error reporting
			assert(depthFormat != nbl::asset::EF_UNKNOWN);
		}

		nbl::video::IGPURenderpass::SCreationParams::SAttachmentDescription attachments[2];
		attachments[0].initialLayout = asset::EIL_UNDEFINED;
		attachments[0].finalLayout = asset::EIL_PRESENT_SRC;
		attachments[0].format = colorAttachmentFormat;
		attachments[0].samples = asset::IImage::ESCF_1_BIT;
		attachments[0].loadOp = nbl::video::IGPURenderpass::ELO_CLEAR;
		attachments[0].storeOp = nbl::video::IGPURenderpass::ESO_STORE;

		attachments[1].initialLayout = asset::EIL_UNDEFINED;
		attachments[1].finalLayout = asset::EIL_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		attachments[1].format = depthFormat;
		attachments[1].samples = asset::IImage::ESCF_1_BIT;
		attachments[1].loadOp = nbl::video::IGPURenderpass::ELO_CLEAR;
		attachments[1].storeOp = nbl::video::IGPURenderpass::ESO_STORE;

		nbl::video::IGPURenderpass::SCreationParams::SSubpassDescription::SAttachmentRef colorAttRef;
		colorAttRef.attachment = 0u;
		colorAttRef.layout = asset::EIL_COLOR_ATTACHMENT_OPTIMAL;

		nbl::video::IGPURenderpass::SCreationParams::SSubpassDescription::SAttachmentRef depthStencilAttRef;
		depthStencilAttRef.attachment = 1u;
		depthStencilAttRef.layout = asset::EIL_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		nbl::video::IGPURenderpass::SCreationParams::SSubpassDescription sp;
		sp.pipelineBindPoint = asset::EPBP_GRAPHICS;
		sp.colorAttachmentCount = 1u;
		sp.colorAttachments = &colorAttRef;
		if(useDepth) {
			sp.depthStencilAttachment = &depthStencilAttRef;
		} else {
			sp.depthStencilAttachment = nullptr;
		}
		sp.flags = nbl::video::IGPURenderpass::ESDF_NONE;
		sp.inputAttachmentCount = 0u;
		sp.inputAttachments = nullptr;
		sp.preserveAttachmentCount = 0u;
		sp.preserveAttachments = nullptr;
		sp.resolveAttachments = nullptr;

		nbl::video::IGPURenderpass::SCreationParams rp_params;
		rp_params.attachmentCount = (useDepth) ? 2u : 1u;
		rp_params.attachments = attachments;
		rp_params.dependencies = nullptr;
		rp_params.dependencyCount = 0u;
		rp_params.subpasses = &sp;
		rp_params.subpassCount = 1u;

		return device->createRenderpass(rp_params);
	}

	static auto createFBOWithSwapchainImages(
		size_t imageCount, uint32_t width, uint32_t height,
		const nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice>& device,
		nbl::core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain,
		nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass, 
		nbl::asset::E_FORMAT baseDepthFormat = nbl::asset::EF_UNKNOWN
	) -> nbl::core::smart_refctd_dynamic_array <nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>>
	{
		using namespace nbl;

		auto depthFormat = device->getPhysicalDevice()->promoteImageFormat(
			{ baseDepthFormat, nbl::video::IPhysicalDevice::SFormatImageUsage(nbl::asset::IImage::EUF_DEPTH_STENCIL_ATTACHMENT_BIT) },
			nbl::asset::IImage::ET_OPTIMAL
		);
		bool useDepth = depthFormat != nbl::asset::EF_UNKNOWN;

		auto sc_images = swapchain->getImages();
		assert(sc_images.size() == imageCount);
		auto fbo = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>>>(imageCount);
		for (uint32_t i = 0u; i < imageCount; ++i)
		{
			nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> view[2] = {};
			
			auto img = sc_images.begin()[i];
			{
				nbl::video::IGPUImageView::SCreationParams view_params;
				view_params.format = img->getCreationParameters().format;
				view_params.viewType = asset::IImageView<nbl::video::IGPUImage>::ET_2D;
				view_params.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
				view_params.subresourceRange.baseMipLevel = 0u;
				view_params.subresourceRange.levelCount = 1u;
				view_params.subresourceRange.baseArrayLayer = 0u;
				view_params.subresourceRange.layerCount = 1u;
				view_params.image = std::move(img);

				view[0] = device->createImageView(std::move(view_params));
				assert(view[0]);
			}
			
			if(useDepth) {
				nbl::video::IGPUImage::SCreationParams imgParams;
				imgParams.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);
				imgParams.type = asset::IImage::ET_2D;
				imgParams.format = depthFormat;
				imgParams.extent = {width, height, 1};
				imgParams.usage = asset::IImage::E_USAGE_FLAGS::EUF_DEPTH_STENCIL_ATTACHMENT_BIT;
				imgParams.mipLevels = 1u;
				imgParams.arrayLayers = 1u;
				imgParams.samples = asset::IImage::ESCF_1_BIT;

				auto depthImg = device->createImage(std::move(imgParams));
				auto depthImgMemReqs = depthImg->getMemoryReqs();
				depthImgMemReqs.memoryTypeBits &= device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
				auto depthImgMem = device->allocate(depthImgMemReqs, depthImg.get());

				nbl::video::IGPUImageView::SCreationParams view_params;
				view_params.format = depthFormat;
				view_params.viewType = asset::IImageView<nbl::video::IGPUImage>::ET_2D;
				view_params.subresourceRange.aspectMask = asset::IImage::EAF_DEPTH_BIT;
				view_params.subresourceRange.baseMipLevel = 0u;
				view_params.subresourceRange.levelCount = 1u;
				view_params.subresourceRange.baseArrayLayer = 0u;
				view_params.subresourceRange.layerCount = 1u;
				view_params.image = std::move(depthImg);

				view[1] = device->createImageView(std::move(view_params));
				assert(view[1]);
			}

			nbl::video::IGPUFramebuffer::SCreationParams fb_params;
			fb_params.width = width;
			fb_params.height = height;
			fb_params.layers = 1u;
			fb_params.renderpass = renderpass;
			fb_params.flags = static_cast<nbl::video::IGPUFramebuffer::E_CREATE_FLAGS>(0);
			fb_params.attachmentCount = (useDepth) ? 2u : 1u;
			fb_params.attachments = view;

			fbo->begin()[i] = device->createFramebuffer(std::move(fb_params));
			assert(fbo->begin()[i]);
		}
		return fbo;
	}

	static constexpr nbl::asset::E_PIPELINE_STAGE_FLAGS DefaultSubmitWaitStage = nbl::asset::EPSF_COLOR_ATTACHMENT_OUTPUT_BIT;
	static void Submit(
		nbl::video::ILogicalDevice* device,
		nbl::video::IGPUCommandBuffer* cmdbuf,
		nbl::video::IGPUQueue* queue,
		nbl::video::IGPUSemaphore* const waitSemaphore, // usually the image acquire semaphore
		nbl::video::IGPUSemaphore* const renderFinishedSemaphore,
		nbl::video::IGPUFence* fence=nullptr,
		const nbl::core::bitflag<nbl::asset::E_PIPELINE_STAGE_FLAGS> waitDstStageMask=DefaultSubmitWaitStage // only matters if `waitSemaphore` not null
	)
	{
		using namespace nbl;
		nbl::video::IGPUQueue::SSubmitInfo submit;
		{
			submit.commandBufferCount = 1u;
			submit.commandBuffers = &cmdbuf;
			nbl::video::IGPUSemaphore* signalsem = renderFinishedSemaphore;
			submit.signalSemaphoreCount = waitSemaphore ? 1u:0u;
			submit.pSignalSemaphores = &signalsem;
			nbl::video::IGPUSemaphore* waitsem = waitSemaphore;
			asset::E_PIPELINE_STAGE_FLAGS dstWait = waitDstStageMask.value;
			submit.waitSemaphoreCount = 1u;
			submit.pWaitSemaphores = &waitsem;
			submit.pWaitDstStageMask = &dstWait;

			queue->submit(1u,&submit,fence);
		}
	}

	static void Present(nbl::video::ILogicalDevice* device,
		nbl::video::ISwapchain* sc,
		nbl::video::IGPUQueue* queue,
		nbl::video::IGPUSemaphore* waitSemaphore, // usually the render finished semaphore
		uint32_t imageNum)
	{
		using namespace nbl;
		nbl::video::IGPUQueue::SPresentInfo present;
		{
			present.swapchainCount = 1u;
			present.imgIndices = &imageNum;
			nbl::video::ISwapchain* swapchain = sc;
			present.swapchains = &swapchain;
			present.waitSemaphoreCount = waitSemaphore ? 1u:0u;
			present.waitSemaphores = &waitSemaphore;

			queue->present(present);
		}
	}

	static std::pair<nbl::core::smart_refctd_ptr<nbl::video::IGPUImage>, nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView>> createEmpty2DTexture(
		const nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice>& device,
		uint32_t width,
		uint32_t height,
		nbl::asset::E_FORMAT format)
	{
		nbl::video::IGPUImage::SCreationParams gpu_image_params;
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

		nbl::video::IGPUImageView::SCreationParams creation_params;
		creation_params.format = image->getCreationParameters().format;
		creation_params.image = image;
		creation_params.viewType = nbl::video::IGPUImageView::ET_2D;
		creation_params.subresourceRange = { static_cast<nbl::asset::IImage::E_ASPECT_FLAGS>(0u), 0, 1, 0, 1 };
		creation_params.flags = static_cast<nbl::video::IGPUImageView::E_CREATE_FLAGS>(0u);
		nbl::core::smart_refctd_ptr image_view = device->createImageView(std::move(creation_params));
		return std::pair(image, image_view);
	}

	static int getQueueFamilyIndex(const nbl::video::IPhysicalDevice* gpu, nbl::core::bitflag<nbl::video::IPhysicalDevice::E_QUEUE_FLAGS> requiredQueueFlags)
	{
		auto props = gpu->getQueueFamilyProperties();
		int currentIndex = 0;
		for (const auto& property : props)
		{
			if ((property.queueFlags.value & requiredQueueFlags.value) == requiredQueueFlags.value)
			{
				return currentIndex;
			}
			++currentIndex;
		}
		return -1;
	}
};


#ifndef _NBL_PLATFORM_ANDROID_
class GraphicalApplication : public nbl::system::IApplicationFramework, public nbl::ui::IGraphicalApplicationFramework
{
protected:
	~GraphicalApplication() {}
public:
	GraphicalApplication(
		const std::filesystem::path& _localInputCWD,
		const std::filesystem::path& _localOutputCWD,
		const std::filesystem::path& _sharedInputCWD,
		const std::filesystem::path& _sharedOutputCWD
	) : nbl::system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}
	void recreateSurface() override
	{
	}
};
#else
class GraphicalApplication : public nbl::ui::CGraphicalApplicationAndroid
{
protected:
	~GraphicalApplication() {}
public:
	GraphicalApplication(
		android_app* app, JNIEnv* env,
		const std::filesystem::path& _localInputCWD,
		const std::filesystem::path& _localOutputCWD,
		const std::filesystem::path& _sharedInputCWD,
		const std::filesystem::path& _sharedOutputCWD
	) : nbl::ui::CGraphicalApplicationAndroid(app, env, _localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}
	void recreateSurface() override
	{
		CommonAPI::recreateSurface(this);
	}
};
#endif
//***** Application framework macros ******
#ifdef _NBL_PLATFORM_ANDROID_
using ApplicationBase = GraphicalApplication;
using NonGraphicalApplicationBase = nbl::system::CApplicationAndroid;
#define APP_CONSTRUCTOR(type) type(android_app* app, JNIEnv* env, const nbl::system::path& _localInputCWD,\
const nbl::system::path& _localOutputCWD,\
const nbl::system::path& _sharedInputCWD,\
const nbl::system::path& _sharedOutputCWD) : ApplicationBase(app, env, _localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

#define NON_GRAPHICAL_APP_CONSTRUCTOR(type) type(android_app* app, JNIEnv* env, const nbl::system::path& _localInputCWD,\
const nbl::system::path& _localOutputCWD,\
const nbl::system::path& _sharedInputCWD,\
const nbl::system::path& _sharedOutputCWD) : NonGraphicalApplicationBase(app, env, _localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

#define NBL_COMMON_API_MAIN(android_app_class) NBL_ANDROID_MAIN_FUNC(android_app_class, CommonAPI::CommonAPIEventCallback)
#else
using ApplicationBase = GraphicalApplication;
using NonGraphicalApplicationBase = nbl::system::IApplicationFramework;
#define APP_CONSTRUCTOR(type) type(const nbl::system::path& _localInputCWD,\
const nbl::system::path& _localOutputCWD,\
const nbl::system::path& _sharedInputCWD,\
const nbl::system::path& _sharedOutputCWD) : ApplicationBase(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

#define NON_GRAPHICAL_APP_CONSTRUCTOR(type) type(const nbl::system::path& _localInputCWD,\
const nbl::system::path& _localOutputCWD,\
const nbl::system::path& _sharedInputCWD,\
const nbl::system::path& _sharedOutputCWD) : NonGraphicalApplicationBase(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}
#define NBL_COMMON_API_MAIN(app_class) int main(int argc, char** argv){\
CommonAPI::main<app_class>(argc, argv);\
}
#endif
//***** Application framework macros ******

