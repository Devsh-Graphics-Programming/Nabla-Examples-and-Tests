#ifndef __NBL_COMMON_API_H_INCLUDED__
#define __NBL_COMMON_API_H_INCLUDED__

#define _NBL_STATIC_LIB_
#include <nabla.h>

// TODO: get these all included by the appropriate namespace headers!
#include "nbl/system/IApplicationFramework.h"

#include "nbl/ui/CGraphicalApplicationAndroid.h"
#include "nbl/ui/CWindowManagerAndroid.h"
#include "nbl/ui/IGraphicalApplicationFramework.h"

// TODO: see TODO below
#if defined(_NBL_PLATFORM_WINDOWS_)
#include <nbl/system/CColoredStdoutLoggerWin32.h>
#elif defined(_NBL_PLATFORM_ANDROID_)
#include <nbl/system/CStdoutLoggerAndroid.h>
#endif
#include "nbl/system/CSystemAndroid.h"
#include "nbl/system/CSystemLinux.h"
#include "nbl/system/CSystemWin32.h"
// TODO: make these include themselves via `nabla.h`

#include "nbl/video/utilities/SPhysicalDeviceFilter.h"

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
			inline void consumeEvents(F&& processFunc, nbl::system::logger_opt_ptr logger = nullptr)
			{
				auto events = channel->getEvents();
				const auto frontBufferCapacity = channel->getFrontBufferCapacity();
				if (events.size() > consumedCounter + frontBufferCapacity)
				{
					logger.log(
						"Detected overflow, %d unconsumed events in channel of size %d!",
						nbl::system::ILogger::ELL_ERROR, events.size() - consumedCounter, frontBufferCapacity
					);
					consumedCounter = events.size() - frontBufferCapacity;
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
			getDefault(m_mouse, reader);
		}
		void getDefaultKeyboard(ChannelReader<nbl::ui::IKeyboardEventChannel>* reader)
		{
			getDefault(m_keyboard, reader);
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
				channels.channels.begin(), channels.channels.end(), [channel](const auto& chan)->bool {return chan.get() == channel; }
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
				m_logger.log("Waiting For Input Device to be connected...", nbl::system::ILogger::ELL_INFO);
				channels.added.wait(lock);
			}

			uint64_t consumedCounter = 0ull;

			using namespace std::chrono;
			constexpr long long DefaultChannelTimeoutInMicroSeconds = 100 * 1e3; // 100 mili-seconds
			auto nowTimeStamp = duration_cast<microseconds>(steady_clock::now().time_since_epoch());

			// Update Timestamp of all channels
			for (uint32_t ch = 0u; ch < channels.channels.size(); ++ch) {
				auto& channel = channels.channels[ch];
				auto& timeStamp = channels.timeStamps[ch];
				auto events = channel->getEvents();
				if (events.size() > 0) {
					auto lastEventTimeStamp = (*(events.end() - 1)).timeStamp; // last event timestamp
					timeStamp = lastEventTimeStamp;
				}
			}

			auto defaultIdx = channels.defaultChannelIndex;
			if (defaultIdx >= channels.channels.size()) {
				defaultIdx = 0;
			}
			auto defaultChannel = channels.channels[defaultIdx];
			auto defaultChannelEvents = defaultChannel->getEvents();
			auto timeDiff = (nowTimeStamp - channels.timeStamps[defaultIdx]).count();

			constexpr size_t RewindBackEvents = 50u;

			// If the current one hasn't been active for a while
			if (defaultChannel->empty()) {
				if (timeDiff > DefaultChannelTimeoutInMicroSeconds) {
					// Look for the most active channel (the channel which has got the most events recently)
					auto newDefaultIdx = defaultIdx;
					microseconds maxEventTimeStamp = microseconds(0);

					for (uint32_t chIdx = 0; chIdx < channels.channels.size(); ++chIdx) {
						if (defaultIdx != chIdx)
						{
							auto channelTimeDiff = (nowTimeStamp - channels.timeStamps[chIdx]).count();
							// Check if was more recently active than the current most active
							if (channelTimeDiff < DefaultChannelTimeoutInMicroSeconds)
							{
								auto& channel = channels.channels[chIdx];
								auto channelEvents = channel->getEvents();
								auto channelEventSize = channelEvents.size();
								const auto frontBufferCapacity = channel->getFrontBufferCapacity();

								size_t rewindBack = std::min(RewindBackEvents, frontBufferCapacity);
								rewindBack = std::min(rewindBack, channelEventSize);

								auto oldEvent = *(channelEvents.end() - rewindBack);

								// Which oldEvent of channels are most recent.
								if (oldEvent.timeStamp > maxEventTimeStamp) {
									maxEventTimeStamp = oldEvent.timeStamp;
									newDefaultIdx = chIdx;
								}
							}
						}
					}

					if (defaultIdx != newDefaultIdx) {
						m_logger.log("Default InputChannel for ChannelType changed from %u to %u", nbl::system::ILogger::ELL_INFO, defaultIdx, newDefaultIdx);

						defaultIdx = newDefaultIdx;
						channels.defaultChannelIndex = newDefaultIdx;
						defaultChannel = channels.channels[newDefaultIdx];

						consumedCounter = defaultChannel->getEvents().size() - defaultChannel->getFrontBufferCapacity(); // to not get overflow in reader when consuming.
					}
				}
			}

			if (reader->channel == defaultChannel)
				return;

			reader->channel = defaultChannel;
			reader->consumedCounter = consumedCounter;
		}

		nbl::system::logger_opt_smart_ptr m_logger;
		Channels<nbl::ui::IMouseEventChannel> m_mouse;
		Channels<nbl::ui::IKeyboardEventChannel> m_keyboard;
	};

	class CommonAPIEventCallback : public virtual nbl::ui::IWindow::IEventCallback
	{
	public:
		CommonAPIEventCallback(nbl::core::smart_refctd_ptr<InputSystem>&& inputSystem, nbl::system::logger_opt_smart_ptr&& logger) : m_inputSystem(std::move(inputSystem)), m_logger(std::move(logger)), m_gotWindowClosedMsg(false) {}
		CommonAPIEventCallback() {}
		bool isWindowOpen() const { return !m_gotWindowClosedMsg; }
		void setLogger(nbl::system::logger_opt_smart_ptr& logger)
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
			m_inputSystem.get()->add(m_inputSystem.get()->m_mouse, std::move(mch));
		}
		void onMouseDisconnected_impl(nbl::ui::IMouseEventChannel* mch) override
		{
			m_logger.log("A mouse %p has been disconnected", nbl::system::ILogger::ELL_INFO, mch);
			m_inputSystem.get()->remove(m_inputSystem.get()->m_mouse, mch);
		}
		void onKeyboardConnected_impl(nbl::core::smart_refctd_ptr<nbl::ui::IKeyboardEventChannel>&& kbch) override
		{
			m_logger.log("A keyboard %p has been connected", nbl::system::ILogger::ELL_INFO, kbch.get());
			m_inputSystem.get()->add(m_inputSystem.get()->m_keyboard, std::move(kbch));
		}
		void onKeyboardDisconnected_impl(nbl::ui::IKeyboardEventChannel* kbch) override
		{
			m_logger.log("A keyboard %p has been disconnected", nbl::system::ILogger::ELL_INFO, kbch);
			m_inputSystem.get()->remove(m_inputSystem.get()->m_keyboard, kbch);
		}

	private:
		nbl::core::smart_refctd_ptr<InputSystem> m_inputSystem = nullptr;
		nbl::system::logger_opt_smart_ptr m_logger = nullptr;
		bool m_gotWindowClosedMsg = false;
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

	class IPhysicalDeviceSelector
	{
	public:
		// ! this will get called after all physical devices go through filtering via `InitParams::physicalDeviceFilter`
		virtual nbl::video::IPhysicalDevice* selectPhysicalDevice(const nbl::core::set<nbl::video::IPhysicalDevice*>& suitablePhysicalDevices) = 0;
	};

	class CDefaultPhysicalDeviceSelector : public CommonAPI::IPhysicalDeviceSelector
	{
	protected:
		const nbl::video::IPhysicalDevice::E_DRIVER_ID preferredDriver = nbl::video::IPhysicalDevice::EDI_NVIDIA_PROPRIETARY;

	public:

		CDefaultPhysicalDeviceSelector(nbl::video::IPhysicalDevice::E_DRIVER_ID preferredDriver)
			: preferredDriver(preferredDriver)
		{}

		// ! this will get called after all physical devices go through filtering via `InitParams::physicalDevicesFilter`
		nbl::video::IPhysicalDevice* selectPhysicalDevice(const nbl::core::set<nbl::video::IPhysicalDevice*>& suitablePhysicalDevices) override;
	};

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

		nbl::video::IAPIConnection::SFeatures apiFeaturesToEnable = {};
		//! Optional: Physical Device Requirements include features, limits, memory size, queue count, etc. requirements
		nbl::video::SPhysicalDeviceFilter physicalDeviceFilter = {};
		//! Optional: PhysicalDevices that meet all the requirements of `physicalDeviceFilter` will go through `physicalDeviceSelector` to select one from the suitable physical devices
		IPhysicalDeviceSelector* physicalDeviceSelector = nullptr;

		nbl::asset::IImage::E_USAGE_FLAGS swapchainImageUsage = nbl::asset::IImage::E_USAGE_FLAGS::EUF_NONE;

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
			return swapchainImageUsage == nbl::asset::IImage::EUF_NONE;
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
		nbl::core::smart_refctd_ptr<nbl::asset::CCompilerSet> compilerSet;
		nbl::core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
		nbl::video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
		nbl::core::smart_refctd_ptr<InputSystem> inputSystem;
		nbl::video::ISwapchain::SCreationParams swapchainCreationParams;
		nbl::core::smart_refctd_ptr<nbl::system::ILogger> logger;
		nbl::core::smart_refctd_ptr<nbl::ui::IWindowManager> windowManager;
	};

#ifdef _NBL_PLATFORM_ANDROID_
	static void recreateSurface(nbl::ui::CGraphicalApplicationAndroid* framework)
	{
		// Will handle android properly later
		_NBL_TODO();
	}
#endif

	template<bool gpuInit = true, class EventCallback = CommonAPIEventCallback, nbl::video::DeviceFeatureDependantClass... device_feature_dependant_t>
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

		result.compilerSet = nbl::core::make_smart_refctd_ptr<nbl::asset::CCompilerSet>(nbl::core::smart_refctd_ptr(result.system));

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

				if (params.window->getWidth() != params.windowWidth || params.window->getHeight() != params.windowHeight)
				{
					std::stringstream ss;
					ss << "Requested window size ";
					ss << '(' << params.windowWidth << 'x' << params.windowHeight << ')';
					ss << " could not be applied, actual size: ";
					ss << '(' << params.window->getWidth() << 'x' << params.window->getHeight() << ")";

					result.logger->log(ss.str(), system::ILogger::ELL_INFO);
				}
			}
			params.windowCb = nbl::core::smart_refctd_ptr<CommonAPIEventCallback>(dynamic_cast<CommonAPIEventCallback*>(params.window->getEventCallback()));
		}

		if constexpr (gpuInit)
		{
			performGpuInit<device_feature_dependant_t...>(params, result);
		}
		else
		{
			result.cpu2gpuParams.device = nullptr;
			result.cpu2gpuParams.finalQueueFamIx = 0u;
			result.cpu2gpuParams.pipelineCache = nullptr;
			result.cpu2gpuParams.utilities = nullptr;
		}

		return result;
	}

	static nbl::video::ISwapchain::SCreationParams computeSwapchainCreationParams(
		uint32_t& imageCount,
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


	class IRetiredSwapchainResources : public nbl::video::ICleanup
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

	template<class EventCallback = CommonAPIEventCallback, nbl::video::DeviceFeatureDependantClass... device_feature_dependant_t>
	static InitOutput InitWithDefaultExt(InitParams&& params)
	{
#ifndef _NBL_PLATFORM_ANDROID_
		const auto swapChainMode = nbl::video::E_SWAPCHAIN_MODE::ESM_SURFACE;
		nbl::video::IAPIConnection::SFeatures apiFeaturesToEnable;
		apiFeaturesToEnable.swapchainMode = swapChainMode;
		apiFeaturesToEnable.validations = true;
		apiFeaturesToEnable.debugUtils = true;
		params.apiFeaturesToEnable = apiFeaturesToEnable;

		params.physicalDeviceFilter.requiredFeatures.swapchainMode = swapChainMode;
#endif

		return CommonAPI::Init<true, EventCallback, device_feature_dependant_t...>(std::move(params));
	}

	template<class EventCallback = CommonAPIEventCallback, nbl::video::DeviceFeatureDependantClass... device_feature_dependant_t>
	static InitOutput InitWithRaytracingExt(InitParams&& params)
	{
#ifndef _NBL_PLATFORM_ANDROID_
		const auto swapChainMode = nbl::video::E_SWAPCHAIN_MODE::ESM_SURFACE;
		nbl::video::IAPIConnection::SFeatures apiFeaturesToEnable;
		apiFeaturesToEnable.swapchainMode = swapChainMode;
		apiFeaturesToEnable.validations = true;
		apiFeaturesToEnable.debugUtils = true;
		params.apiFeaturesToEnable = apiFeaturesToEnable;

		params.physicalDeviceFilter.requiredFeatures.swapchainMode = swapChainMode;
		params.physicalDeviceFilter.requiredFeatures.rayQuery = true;
		params.physicalDeviceFilter.requiredFeatures.accelerationStructure = true;
#elif
		return {};
#endif
		return CommonAPI::Init<true, EventCallback, device_feature_dependant_t...>(std::move(params));
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
		nbl::video::IGPUFence* fence = nullptr,
		const nbl::core::bitflag<nbl::asset::E_PIPELINE_STAGE_FLAGS> waitDstStageMask = DefaultSubmitWaitStage // only matters if `waitSemaphore` not null
	)
	{
		using namespace nbl;
		nbl::video::IGPUQueue::SSubmitInfo submit;
		{
			submit.commandBufferCount = 1u;
			submit.commandBuffers = &cmdbuf;
			nbl::video::IGPUSemaphore* signalsem = renderFinishedSemaphore;
			submit.signalSemaphoreCount = signalsem ? 1u : 0u;
			submit.pSignalSemaphores = &signalsem;
			nbl::video::IGPUSemaphore* waitsem = waitSemaphore;
			asset::E_PIPELINE_STAGE_FLAGS dstWait = waitDstStageMask.value;
			submit.waitSemaphoreCount = waitsem ? 1u : 0u;
			submit.pWaitSemaphores = &waitsem;
			submit.pWaitDstStageMask = &dstWait;

			queue->submit(1u, &submit, fence);
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
			present.waitSemaphoreCount = waitSemaphore ? 1u : 0u;
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
	static nbl::core::set<nbl::video::IPhysicalDevice*> getFilteredPhysicalDevices(nbl::core::SRange<nbl::video::IPhysicalDevice* const> physicalDevices, const nbl::video::SPhysicalDeviceFilter& filter)
	{
		using namespace nbl;
		using namespace nbl::video;

		core::set<nbl::video::IPhysicalDevice*> ret;
		for (auto& physDev : physicalDevices) {
			if (filter.meetsRequirements(physDev))
				ret.insert(physDev);
		}
		return ret;
	}

	// Used to help with queue selection
	struct QueueFamilyProps
	{
		static constexpr uint32_t InvalidIndex = ~0u;
		uint32_t index = InvalidIndex;
		uint32_t dedicatedQueueCount = 0u;
		uint32_t score = 0u;
		bool supportsGraphics : 1;
		bool supportsCompute : 1;
		bool supportsTransfer : 1;
		bool supportsSparseBinding : 1;
		bool supportsPresent : 1;
		bool supportsProtected : 1;
	};

	struct PhysicalDeviceQueuesInfo
	{
		QueueFamilyProps graphics;
		QueueFamilyProps compute;
		QueueFamilyProps transfer;
		QueueFamilyProps present;
	};

	// TODO: (Erfan) no need to extract or hold any memory, we can construct the queue info while scoring and selecting queues; then code will be much more readable
	static PhysicalDeviceQueuesInfo extractPhysicalDeviceQueueInfos(
		nbl::video::IPhysicalDevice* const physicalDevice,
		nbl::core::smart_refctd_ptr<nbl::video::ISurface> surface,
		bool headlessCompute)
	{
		using namespace nbl;
		using namespace nbl::video;

		PhysicalDeviceQueuesInfo queuesInfo = {};

		// Find queue family indices
		{
			const auto& queueFamilyProperties = physicalDevice->getQueueFamilyProperties();

			std::vector<uint32_t> remainingQueueCounts = std::vector<uint32_t>(queueFamilyProperties.size(), 0u);

			for (uint32_t familyIndex = 0u; familyIndex < queueFamilyProperties.size(); ++familyIndex)
			{
				const auto& familyProperty = queueFamilyProperties[familyIndex];
				remainingQueueCounts[familyIndex] = familyProperty.queueCount;
			}

			// Select Graphics Queue Family Index
			if (!headlessCompute)
			{
				// Select Graphics Queue Family Index
				for (uint32_t familyIndex = 0u; familyIndex < queueFamilyProperties.size(); ++familyIndex)
				{
					const auto& familyProperty = queueFamilyProperties[familyIndex];

					const uint32_t currentFamilyQueueCount = familyProperty.queueCount;
					if (currentFamilyQueueCount <= 0)
						continue;

					bool supportsPresent = surface && surface->isSupportedForPhysicalDevice(physicalDevice, familyIndex);
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
					if (hasGraphicsFlag) {
						score++;
						if (hasEnoughQueues) {
							score += 1u * remainingQueueCount;

							if (supportsPresent)
							{
								score += 32u; // more important to have present than compute (presentSupport is larger in scoring to 16 extra compute queues)
							}

							if (hasComputeFlag)
							{
								score += 1u * remainingQueueCount;
							}
						}
					}

					if (score > queuesInfo.graphics.score)
					{
						queuesInfo.graphics.index = familyIndex;
						queuesInfo.graphics.supportsGraphics = hasGraphicsFlag;
						queuesInfo.graphics.supportsCompute = hasComputeFlag;
						queuesInfo.graphics.supportsTransfer = true; // Reporting this is optional for Vk Graphics-Capable QueueFam, but Its support is guaranteed.
						queuesInfo.graphics.supportsSparseBinding = hasSparseBindingFlag;
						queuesInfo.graphics.supportsPresent = supportsPresent;
						queuesInfo.graphics.supportsProtected = hasProtectedFlag;
						queuesInfo.graphics.dedicatedQueueCount = 1u;
						queuesInfo.graphics.score = score;
					}
				}
				assert(queuesInfo.graphics.index != QueueFamilyProps::InvalidIndex);
				remainingQueueCounts[queuesInfo.graphics.index] -= queuesInfo.graphics.dedicatedQueueCount;
			}

			// Select Compute Queue Family Index
			for (uint32_t familyIndex = 0u; familyIndex < queueFamilyProperties.size(); ++familyIndex)
			{
				const auto& familyProperty = queueFamilyProperties.begin()[familyIndex];

				const uint32_t currentFamilyQueueCount = familyProperty.queueCount;
				if (currentFamilyQueueCount <= 0)
					continue;

				bool supportsPresent = surface && surface->isSupportedForPhysicalDevice(physicalDevice, familyIndex);
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
				if (hasComputeFlag) {
					score++;

					if (hasExtraQueues) {
						score += 3;
					}

					if (!headlessCompute && hasGraphicsFlag) {
						score++;
						if (familyIndex == queuesInfo.graphics.index) {
							score++;
						}
					}
				}

				if (score > queuesInfo.compute.score)
				{
					queuesInfo.compute.index = familyIndex;
					queuesInfo.compute.supportsGraphics = hasGraphicsFlag;
					queuesInfo.compute.supportsCompute = hasComputeFlag;
					queuesInfo.compute.supportsTransfer = true; // Reporting this is optional for Vk Compute-Capable QueueFam, but Its support is guaranteed.
					queuesInfo.compute.supportsSparseBinding = hasSparseBindingFlag;
					queuesInfo.compute.supportsPresent = supportsPresent;
					queuesInfo.compute.supportsProtected = hasProtectedFlag;
					queuesInfo.compute.dedicatedQueueCount = (hasExtraQueues) ? 1u : 0u;
					queuesInfo.compute.score = score;
				}
			}
			assert(queuesInfo.compute.index != QueueFamilyProps::InvalidIndex);
			remainingQueueCounts[queuesInfo.compute.index] -= queuesInfo.compute.dedicatedQueueCount;

			// Select Transfer Queue Family Index
			for (uint32_t familyIndex = 0u; familyIndex < queueFamilyProperties.size(); ++familyIndex)
			{
				const auto& familyProperty = queueFamilyProperties.begin()[familyIndex];

				const uint32_t currentFamilyQueueCount = familyProperty.queueCount;
				if (currentFamilyQueueCount <= 0)
					continue;

				bool supportsPresent = surface && surface->isSupportedForPhysicalDevice(physicalDevice, familyIndex);
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
				if (hasTransferFlag) {
					score += 1u;

					uint32_t notHavingComputeScore = 1u;
					uint32_t notHavingGraphicsScore = 2u;

					if (hasExtraQueues) { // Having extra queues to have seperate up/down transfer queues is more important
						score += 4u * extraQueueCount;
						notHavingComputeScore *= extraQueueCount;
						notHavingGraphicsScore *= extraQueueCount;
					}

					if (!hasGraphicsFlag) {
						score += notHavingGraphicsScore;
					}

					if (!hasComputeFlag) {
						score += notHavingComputeScore;
					}

				}

				if (score > queuesInfo.transfer.score)
				{
					queuesInfo.transfer.index = familyIndex;
					queuesInfo.transfer.supportsGraphics = hasGraphicsFlag;
					queuesInfo.transfer.supportsCompute = hasComputeFlag;
					queuesInfo.transfer.supportsTransfer = hasTransferFlag;
					queuesInfo.transfer.supportsSparseBinding = hasSparseBindingFlag;
					queuesInfo.transfer.supportsPresent = supportsPresent;
					queuesInfo.transfer.supportsProtected = hasProtectedFlag;
					queuesInfo.transfer.dedicatedQueueCount = extraQueueCount;
					queuesInfo.transfer.score = score;
				}
			}
			assert(queuesInfo.transfer.index != QueueFamilyProps::InvalidIndex);
			remainingQueueCounts[queuesInfo.transfer.index] -= queuesInfo.transfer.dedicatedQueueCount;

			// Select Present Queue Family Index
			if (!headlessCompute)
			{
				if (queuesInfo.graphics.supportsPresent && queuesInfo.graphics.index != QueueFamilyProps::InvalidIndex)
				{
					queuesInfo.present = queuesInfo.graphics;
					queuesInfo.present.dedicatedQueueCount = 0u;
				}
				else
				{
					const uint32_t maxNeededQueueCountForPresent = 1u;
					for (uint32_t familyIndex = 0u; familyIndex < queueFamilyProperties.size(); ++familyIndex)
					{
						const auto& familyProperty = queueFamilyProperties.begin()[familyIndex];

						const uint32_t currentFamilyQueueCount = familyProperty.queueCount;
						if (currentFamilyQueueCount <= 0)
							continue;

						bool supportsPresent = surface && surface->isSupportedForPhysicalDevice(physicalDevice, familyIndex);
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
						if (supportsPresent) {
							score += 1u;

							uint32_t graphicsSupportScore = 100u;
							if (hasEnoughExtraQueues) {
								score += 1u * remainingQueueCount;
								graphicsSupportScore *= remainingQueueCount;
							}

							if (hasGraphicsFlag) {
								score += graphicsSupportScore; // graphics support is larger in scoring than 100 extra queues with no graphics support
							}
						}

						if (score > queuesInfo.present.score)
						{
							queuesInfo.present.index = familyIndex;
							queuesInfo.present.supportsGraphics = hasGraphicsFlag;
							queuesInfo.present.supportsCompute = hasComputeFlag;
							queuesInfo.present.supportsTransfer = hasTransferFlag;
							queuesInfo.present.supportsSparseBinding = hasSparseBindingFlag;
							queuesInfo.present.supportsPresent = supportsPresent;
							queuesInfo.present.supportsProtected = hasProtectedFlag;
							queuesInfo.present.dedicatedQueueCount = (hasEnoughExtraQueues) ? 1u : 0u;
							queuesInfo.present.score = score;
						}
					}
				}
				assert(queuesInfo.present.index != QueueFamilyProps::InvalidIndex);
				remainingQueueCounts[queuesInfo.present.index] -= queuesInfo.present.dedicatedQueueCount;
			}

			if (!headlessCompute)
				assert(queuesInfo.graphics.supportsTransfer && "This shouldn't happen");
			assert(queuesInfo.compute.supportsTransfer && "This shouldn't happen");
		}

		return queuesInfo;
	}


	template<nbl::video::DeviceFeatureDependantClass... device_feature_dependant_t>
	static void performGpuInit(InitParams& params, InitOutput& result)
	{
		using namespace nbl;
		using namespace nbl::video;

		bool headlessCompute = params.isHeadlessCompute();

		if (params.apiType == EAT_VULKAN)
		{
			auto _apiConnection = nbl::video::CVulkanConnection::create(
				nbl::core::smart_refctd_ptr(result.system),
				0,
				params.appName.data(),
				nbl::core::smart_refctd_ptr(result.logger),
				params.apiFeaturesToEnable
			);
			assert(_apiConnection);

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
		else
		{
			_NBL_TODO();
		}

		auto gpus = result.apiConnection->getPhysicalDevices();
		assert(!gpus.empty());

		(device_feature_dependant_t::enableRequiredFeautres(params.physicalDeviceFilter.requiredFeatures), ...);

		auto filteredPhysicalDevices = getFilteredPhysicalDevices(gpus, params.physicalDeviceFilter);

		if (filteredPhysicalDevices.empty() && result.logger)
		{
			result.logger->log("No available PhysicalDevice met the requirements.", nbl::system::ILogger::ELL_ERROR);
			assert(false);
			return;
		}

		CDefaultPhysicalDeviceSelector defaultPhysicalDeviceSelector(nbl::video::IPhysicalDevice::EDI_NVIDIA_PROPRIETARY);  // EDI_INTEL_PROPRIETARY_WINDOWS, EDI_NVIDIA_PROPRIETARY, EDI_AMD_PROPRIETARY
		if (params.physicalDeviceSelector == nullptr)
			params.physicalDeviceSelector = &defaultPhysicalDeviceSelector;

		auto selectedPhysicalDevice = params.physicalDeviceSelector->selectPhysicalDevice(filteredPhysicalDevices);

		if (selectedPhysicalDevice == nullptr)
		{
			result.logger->log("Physical Device selection callback returned no physical device.", nbl::system::ILogger::ELL_ERROR);
			assert(false);
			return;
		}

		(device_feature_dependant_t::enablePreferredFeatures(selectedPhysicalDevice->getFeatures(), params.physicalDeviceFilter.requiredFeatures), ...);

		auto queuesInfo = extractPhysicalDeviceQueueInfos(selectedPhysicalDevice, result.surface, headlessCompute);

		// Fill QueueCreationParams
		constexpr uint32_t MaxQueuesInFamily = video::ILogicalDevice::SQueueCreationParams::MaxQueuesInFamily;
		std::array<float, MaxQueuesInFamily> queuePriorities;
		queuePriorities.fill(IGPUQueue::DEFAULT_QUEUE_PRIORITY);

		constexpr uint32_t MaxQueueFamilyCount = nbl::video::ILogicalDevice::SCreationParams::MaxQueueFamilies;
		std::array<nbl::video::ILogicalDevice::SQueueCreationParams, MaxQueueFamilyCount> qcp;

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
		if (queuesInfo.graphics.index != QueueFamilyProps::InvalidIndex)
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

#ifndef _NBL_PLATFORM_ANDROID_
class GraphicalApplication : public virtual CommonAPI::CommonAPIEventCallback, public nbl::system::IApplicationFramework, public virtual nbl::ui::IGraphicalApplicationFramework
{
protected:
	~GraphicalApplication() {}

	static constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;

	uint32_t m_frameIx = 0;
	nbl::core::deque<CommonAPI::IRetiredSwapchainResources*> m_qRetiredSwapchainResources;
	uint32_t m_swapchainIteration = 0;
	std::array<uint32_t, CommonAPI::InitOutput::MaxSwapChainImageCount> m_imageSwapchainIterations;
	std::mutex m_swapchainPtrMutex;

	// Returns retired resources
	virtual std::unique_ptr<CommonAPI::IRetiredSwapchainResources> onCreateResourcesWithSwapchain(const uint32_t imageIndex) { return nullptr; }

	bool tryAcquireImage(nbl::video::ISwapchain* swapchain, nbl::video::IGPUSemaphore* waitSemaphore, uint32_t* imgnum)
	{
		if (swapchain->acquireNextImage(MAX_TIMEOUT, waitSemaphore, nullptr, imgnum) == nbl::video::ISwapchain::EAIR_SUCCESS)
		{
			if (m_swapchainIteration > m_imageSwapchainIterations[*imgnum])
			{
				auto retiredResources = onCreateResourcesWithSwapchain(*imgnum).release();
				m_imageSwapchainIterations[*imgnum] = m_swapchainIteration;
				if (retiredResources) CommonAPI::retireSwapchainResources(m_qRetiredSwapchainResources, retiredResources);
			}

			return true;
		}
		return false;
	}

	std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUImage>, 2> m_tripleBufferRenderTargets;

	struct PresentedFrameInfo
	{
		uint64_t width : 14;
		uint64_t height : 14;
		uint64_t resourceIx : 8; // (Frame in flight)
		uint64_t frameIx : 28; // (Total amount of frames rendered so far / frame index)
	};

	std::atomic_uint64_t m_lastPresentedFrame;

	PresentedFrameInfo getLastPresentedFrame()
	{
		uint64_t value = m_lastPresentedFrame.load();
		PresentedFrameInfo frame;
		frame.width = value >> 50;
		frame.height = (value >> 36) & ((1 << 14) - 1);
		frame.resourceIx = (value >> 28) & ((1 << 8) - 1);
		frame.frameIx = value & ((1 << 28) - 1);

		return frame;
	}

	void setLastPresentedFrame(PresentedFrameInfo frame)
	{
		uint64_t value = 0;
		value |= frame.width << 50;
		value |= frame.height << 36;
		value |= frame.resourceIx << 28;
		value |= frame.frameIx;
		m_lastPresentedFrame.store(value);
	}

	virtual void onCreateResourcesWithTripleBufferTarget(nbl::core::smart_refctd_ptr<nbl::video::IGPUImage>& image, uint32_t bufferIx) {}

	nbl::video::IGPUImage* getTripleBufferTarget(
		uint32_t frameIx, uint32_t w, uint32_t h,
		nbl::asset::E_FORMAT surfaceFormat,
		nbl::core::bitflag<nbl::asset::IImage::E_USAGE_FLAGS> imageUsageFlags)
	{
		uint32_t bufferIx = frameIx % 2;
		auto& image = m_tripleBufferRenderTargets.begin()[bufferIx];

		if (!image || image->getCreationParameters().extent.width < w || image->getCreationParameters().extent.height < h)
		{
			auto logicalDevice = getLogicalDevice();
			nbl::video::IGPUImage::SCreationParams creationParams;
			creationParams.type = nbl::asset::IImage::ET_2D;
			creationParams.samples = nbl::asset::IImage::ESCF_1_BIT;
			creationParams.format = surfaceFormat;
			creationParams.extent = { w, h, 1 };
			creationParams.mipLevels = 1;
			creationParams.arrayLayers = 1;
			creationParams.usage = imageUsageFlags | nbl::asset::IImage::EUF_TRANSFER_SRC_BIT;

			image = logicalDevice->createImage(std::move(creationParams));
			auto memReqs = image->getMemoryReqs();
			memReqs.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
			logicalDevice->allocate(memReqs, image.get());

			onCreateResourcesWithTripleBufferTarget(image, bufferIx);
		}

		return image.get();
	}
public:
	GraphicalApplication(
		const std::filesystem::path& _localInputCWD,
		const std::filesystem::path& _localOutputCWD,
		const std::filesystem::path& _sharedInputCWD,
		const std::filesystem::path& _sharedOutputCWD
	) : nbl::system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD),
		CommonAPI::CommonAPIEventCallback(nullptr, nullptr),
		m_qRetiredSwapchainResources(),
		m_imageSwapchainIterations{}
	{}

	std::unique_lock<std::mutex> recreateSwapchain(
		uint32_t w, uint32_t h,
		nbl::video::ISwapchain::SCreationParams& swapchainCreationParams,
		nbl::core::smart_refctd_ptr<nbl::video::ISwapchain>& swapchainRef)
	{
		auto logicalDevice = getLogicalDevice();
		std::unique_lock guard(m_swapchainPtrMutex);
		CommonAPI::createSwapchain(
			nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice>(logicalDevice),
			swapchainCreationParams,
			w, h,
			swapchainRef);
		assert(swapchainRef);
		m_swapchainIteration++;

		return guard;
	}

	void waitForFrame(
		uint32_t framesInFlight,
		nbl::core::smart_refctd_ptr<nbl::video::IGPUFence>& fence)
	{
		auto logicalDevice = getLogicalDevice();
		if (fence)
		{
			logicalDevice->blockForFences(1u, &fence.get());
			if (m_frameIx >= framesInFlight) CommonAPI::dropRetiredSwapchainResources(m_qRetiredSwapchainResources, m_frameIx - framesInFlight);
		}
		else
			fence = logicalDevice->createFence(static_cast<nbl::video::IGPUFence::E_CREATE_FLAGS>(0));
	}

	nbl::core::smart_refctd_ptr<nbl::video::ISwapchain> acquire(
		nbl::core::smart_refctd_ptr<nbl::video::ISwapchain>& swapchainRef,
		nbl::video::IGPUSemaphore* waitSemaphore,
		uint32_t* imgnum)
	{
		nbl::core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain;
		while (true)
		{
			std::unique_lock guard(m_swapchainPtrMutex);
			swapchain = swapchainRef;
			if (tryAcquireImage(swapchain.get(), waitSemaphore, imgnum))
			{
				return swapchain;
			}
		}
	}

	void immediateImagePresent(nbl::video::IGPUQueue* queue, nbl::video::ISwapchain* swapchain, nbl::core::smart_refctd_ptr<nbl::video::IGPUImage>* swapchainImages, uint32_t frameIx, uint32_t lastRenderW, uint32_t lastRenderH)
	{
		using namespace nbl;

		uint32_t bufferIx = frameIx % 2;
		auto image = m_tripleBufferRenderTargets.begin()[bufferIx];
		auto logicalDevice = getLogicalDevice();

		auto imageAcqToSubmit = logicalDevice->createSemaphore();
		auto submitToPresent = logicalDevice->createSemaphore();

		// acquires image, allocates one shot fences, commandpool and commandbuffer to do a blit, submits and presents
		uint32_t imgnum = 0;
		bool acquireResult = tryAcquireImage(swapchain, imageAcqToSubmit.get(), &imgnum);
		assert(acquireResult);

		auto& swapchainImage = swapchainImages[imgnum]; // tryAcquireImage will have this image be recreated
		auto fence = logicalDevice->createFence(static_cast<nbl::video::IGPUFence::E_CREATE_FLAGS>(0));;
		auto commandPool = logicalDevice->createCommandPool(queue->getFamilyIndex(), nbl::video::IGPUCommandPool::ECF_NONE);
		nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandBuffer> commandBuffer;
		logicalDevice->createCommandBuffers(commandPool.get(), nbl::video::IGPUCommandBuffer::EL_PRIMARY, 1u, &commandBuffer);

		commandBuffer->begin(nbl::video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);

		const uint32_t numBarriers = 2;
		video::IGPUCommandBuffer::SImageMemoryBarrier layoutTransBarrier[numBarriers] = {};
		for (uint32_t i = 0; i < numBarriers; i++) {
			layoutTransBarrier[i].srcQueueFamilyIndex = ~0u;
			layoutTransBarrier[i].dstQueueFamilyIndex = ~0u;
			layoutTransBarrier[i].subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
			layoutTransBarrier[i].subresourceRange.baseMipLevel = 0u;
			layoutTransBarrier[i].subresourceRange.levelCount = 1u;
			layoutTransBarrier[i].subresourceRange.baseArrayLayer = 0u;
			layoutTransBarrier[i].subresourceRange.layerCount = 1u;
		}

		layoutTransBarrier[0].barrier.srcAccessMask = asset::EAF_NONE;
		layoutTransBarrier[0].barrier.dstAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
		layoutTransBarrier[0].oldLayout = asset::IImage::EL_UNDEFINED;
		layoutTransBarrier[0].newLayout = asset::IImage::EL_TRANSFER_DST_OPTIMAL;
		layoutTransBarrier[0].image = swapchainImage;

		layoutTransBarrier[1].barrier.srcAccessMask = asset::EAF_NONE;
		layoutTransBarrier[1].barrier.dstAccessMask = asset::EAF_TRANSFER_READ_BIT;
		layoutTransBarrier[1].oldLayout = asset::IImage::EL_GENERAL;
		layoutTransBarrier[1].newLayout = asset::IImage::EL_TRANSFER_SRC_OPTIMAL;
		layoutTransBarrier[1].image = image;

		commandBuffer->pipelineBarrier(
			asset::EPSF_TOP_OF_PIPE_BIT,
			asset::EPSF_TRANSFER_BIT,
			static_cast<asset::E_DEPENDENCY_FLAGS>(0u),
			0u, nullptr,
			0u, nullptr,
			numBarriers, &layoutTransBarrier[0]);

		nbl::asset::SImageBlit blit;
		blit.srcSubresource.aspectMask = nbl::video::IGPUImage::EAF_COLOR_BIT;
		blit.srcSubresource.layerCount = 1;
		blit.srcOffsets[0] = { 0, 0, 0 };
		blit.srcOffsets[1] = { lastRenderW, lastRenderH, 1 };
		blit.dstSubresource.aspectMask = nbl::video::IGPUImage::EAF_COLOR_BIT;
		blit.dstSubresource.layerCount = 1;
		blit.dstOffsets[0] = { 0, 0, 0 };
		blit.dstOffsets[1] = { swapchain->getCreationParameters().width, swapchain->getCreationParameters().height, 1 };

		printf(
			"Blitting from frame %i buffer %i with last render dimensions %ix%i and output %ix%i\n",
			frameIx, bufferIx,
			lastRenderW, lastRenderH,
			image->getCreationParameters().extent.width, image->getCreationParameters().extent.height
		);
		commandBuffer->blitImage(
			image.get(), nbl::asset::IImage::EL_TRANSFER_SRC_OPTIMAL,
			swapchainImage.get(), nbl::asset::IImage::EL_TRANSFER_DST_OPTIMAL,
			1, &blit, nbl::asset::ISampler::ETF_LINEAR
		);

		layoutTransBarrier[0].barrier.srcAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
		layoutTransBarrier[0].barrier.dstAccessMask = asset::EAF_NONE;
		layoutTransBarrier[0].oldLayout = asset::IImage::EL_TRANSFER_DST_OPTIMAL;
		layoutTransBarrier[0].newLayout = asset::IImage::EL_PRESENT_SRC;

		layoutTransBarrier[1].barrier.srcAccessMask = asset::EAF_TRANSFER_READ_BIT;
		layoutTransBarrier[1].barrier.dstAccessMask = asset::EAF_NONE;
		layoutTransBarrier[1].oldLayout = asset::IImage::EL_TRANSFER_SRC_OPTIMAL;
		layoutTransBarrier[1].newLayout = asset::IImage::EL_GENERAL;

		commandBuffer->pipelineBarrier(
			asset::EPSF_TRANSFER_BIT,
			asset::EPSF_BOTTOM_OF_PIPE_BIT,
			static_cast<asset::E_DEPENDENCY_FLAGS>(0u),
			0u, nullptr,
			0u, nullptr,
			numBarriers, &layoutTransBarrier[0]);

		commandBuffer->end();

		CommonAPI::Submit(
			logicalDevice, commandBuffer.get(), queue,
			imageAcqToSubmit.get(),
			submitToPresent.get(),
			fence.get());
		CommonAPI::Present(
			logicalDevice,
			swapchain,
			queue,
			submitToPresent.get(),
			imgnum);

		logicalDevice->blockForFences(1u, &fence.get());
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
class NonGraphicalApplicationBase : public nbl::system::IApplicationFramework, public nbl::core::IReferenceCounted
{
public:
	using Base = nbl::system::IApplicationFramework;
	using Base::Base;
};
#define APP_CONSTRUCTOR(type) type(const nbl::system::path& _localInputCWD,\
const nbl::system::path& _localOutputCWD,\
const nbl::system::path& _sharedInputCWD,\
const nbl::system::path& _sharedOutputCWD) : ApplicationBase(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

#define NON_GRAPHICAL_APP_CONSTRUCTOR(type) type(const nbl::system::path& _localInputCWD,\
const nbl::system::path& _localOutputCWD,\
const nbl::system::path& _sharedInputCWD,\
const nbl::system::path& _sharedOutputCWD) : NonGraphicalApplicationBase(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}
#endif
//***** Application framework macros ******

#endif