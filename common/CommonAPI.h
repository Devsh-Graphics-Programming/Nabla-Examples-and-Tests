#ifndef __NBL_COMMON_API_H_INCLUDED__
#define __NBL_COMMON_API_H_INCLUDED__

#define _NBL_STATIC_LIB_
#include <nabla.h>

// TODO: get these all included by the appropriate namespace headers!
#include "nbl/system/IApplicationFramework.h"
#include "nbl/video/IDeviceMemoryBacked.h"
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

		void setApplication(nbl::ui::IGraphicalApplicationFramework* app)
		{
			m_app = std::move(app);
		}
	protected:
		nbl::ui::IGraphicalApplicationFramework* m_app = nullptr;
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
			if (m_app) m_app->onResize(w, h);
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
		const bool headlessCompute = false);
	
	// TODO: also implement a function:findBestGPU
	// Returns an index into gpus info vector
	static uint32_t findSuitableGPU(const std::vector<GPUInfo>& extractedInfos, const bool headlessCompute);

	template <typename FeatureType>
	struct SFeatureRequest
	{
		uint32_t count = 0u;
		FeatureType* features = nullptr;
	};

	struct InitParams
	{
		std::string_view appName;
		nbl::video::E_API_TYPE apiType = nbl::video::EAT_VULKAN;
		
		uint32_t framesInFlight = 5u;
		uint32_t windowWidth = 800u;
		uint32_t windowHeight = 600u;
		uint32_t swapchainImageCount = 3u;

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
		nbl::core::smart_refctd_ptr<nbl::ui::IWindowManager> windowManager;
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
		// Will handle android properly later
		_NBL_TODO();
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
			if (!params.window)
			{
				nbl::core::smart_refctd_ptr<EventCallback> windowCallback = nbl::core::make_smart_refctd_ptr<EventCallback>(nbl::core::smart_refctd_ptr(result.inputSystem), system::logger_opt_smart_ptr(nbl::core::smart_refctd_ptr(result.logger)));
				result.windowManager = nbl::core::make_smart_refctd_ptr<nbl::ui::CWindowManagerWin32>(); // on the Android path

				nbl::ui::IWindow::SCreationParams windowsCreationParams;
				windowsCreationParams.width = params.windowWidth;
				windowsCreationParams.height = params.windowHeight;
				windowsCreationParams.x = 64u;
				windowsCreationParams.y = 64u;
				windowsCreationParams.system = nbl::core::smart_refctd_ptr(result.system);
				windowsCreationParams.flags = nbl::ui::IWindow::ECF_RESIZABLE;
				windowsCreationParams.windowCaption = params.appName.data();
				windowsCreationParams.callback = windowCallback;

				params.window = result.windowManager->createWindow(std::move(windowsCreationParams));
			}
			params.windowCb = nbl::core::smart_refctd_ptr<CommonAPIEventCallback>((CommonAPIEventCallback*) params.window->getEventCallback());
			params.windowCb->setInputSystem(nbl::core::smart_refctd_ptr(result.inputSystem));
		}

		if constexpr (gpuInit)
		{
			performGpuInit(params, result);
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
	);


	class IRetiredSwapchainResources: public nbl::video::ICleanup
	{
	public:
		// nbl::core::smart_refctd_ptr<nbl::video::ISwapchain> oldSwapchain = nullptr; // this gets dropped along with the images
		uint64_t retiredFrameId = 0;
	};

	static void dropRetiredSwapchainResources(nbl::core::deque<IRetiredSwapchainResources*>& qRetiredSwapchainResources, const uint64_t completedFrameId);
	static void retireSwapchainResources(nbl::core::deque<IRetiredSwapchainResources*>& qRetiredSwapchainResources, IRetiredSwapchainResources* retired);

	static bool createSwapchain(
		const nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice>&& device,
		nbl::video::ISwapchain::SCreationParams& params,
		uint32_t width, uint32_t height,
		// nullptr for initial creation, old swapchain for eventual resizes
		nbl::core::smart_refctd_ptr<nbl::video::ISwapchain>& swapchain
	);

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

	static nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> createRenderpass(const nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice>& device, nbl::asset::E_FORMAT colorAttachmentFormat, nbl::asset::E_FORMAT baseDepthFormat);

	static nbl::core::smart_refctd_dynamic_array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>> createFBOWithSwapchainImages(
		size_t imageCount, uint32_t width, uint32_t height,
		const nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice>& device,
		nbl::core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain,
		nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass,
		nbl::asset::E_FORMAT baseDepthFormat = nbl::asset::EF_UNKNOWN
	);

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
		nbl::video::ISwapchain::SPresentInfo present;
		{
			present.imgIndex = imageNum;
			present.waitSemaphoreCount = waitSemaphore ? 1u:0u;
			present.waitSemaphores = &waitSemaphore;
			
			sc->present(queue, present);
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
protected:
	static void performGpuInit(InitParams& params, InitOutput& result);
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
	void onResize(uint32_t w, uint32_t h) override
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

#endif
