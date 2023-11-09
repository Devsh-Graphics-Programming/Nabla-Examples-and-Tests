// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

// always include nabla first before std:: headers
#include "nabla.h"

#include "nbl/system/CStdoutLogger.h"
#include "nbl/system/CFileLogger.h"
#include "nbl/system/CColoredStdoutLoggerWin32.h"
#include "nbl/system/IApplicationFramework.h"

#include <iostream>
#include <cstdio>

//! builtin resources archive test
#ifdef NBL_EMBED_BUILTIN_RESOURCES
#include "nbl/builtin/CArchive.h"
#include "yourNamespace/builtin/CArchive.h"
#endif

using namespace nbl;
using namespace core;
using namespace ui;
using namespace system;
using namespace asset;

class WindowEventCallback;

// a basic input system which detects connection of keyboards and mice, then defaults to the most recently active one
class InputSystem : public IReferenceCounted
{
	public:
		template <class ChannelType>
		struct Channels
		{
			core::mutex lock;
			std::condition_variable added;
			core::vector<core::smart_refctd_ptr<ChannelType>> channels;
		};
		// TODO: move to "nbl/ui/InputEventChannel.h" once the interface of this utility struct matures, also maybe rename to `Consumer` ?
		template <class ChannelType>
		struct ChannelReader
		{
			template<typename F>
			inline void consumeEvents(F&& processFunc, system::logger_opt_ptr logger=nullptr)
			{
				auto events = channel->getEvents();
				const auto frontBufferCapacity = channel->getFrontBufferCapacity();
				if (events.size()>consumedCounter+frontBufferCapacity)
				{
					logger.log(
						"Detected overflow, %d unconsumed events in channel of size %d!",
						system::ILogger::ELL_ERROR,events.size()-consumedCounter,frontBufferCapacity
					);
					consumedCounter = events.size()-frontBufferCapacity;
				}
				processFunc(ChannelType::range_t(events.begin()+consumedCounter,events.end()), channel);
				consumedCounter = events.size();
			}

			core::smart_refctd_ptr<ChannelType> channel = nullptr;
			uint64_t consumedCounter = 0ull;
		};
		
		InputSystem(system::logger_opt_smart_ptr&& logger) : m_logger(std::move(logger)) {}

		void getDefaultMouse(ChannelReader<IMouseEventChannel>* reader)
		{
			getDefault(m_mouse,reader);
		}
		void getDefaultKeyboard(ChannelReader<IKeyboardEventChannel>* reader)
		{
			getDefault(m_keyboard,reader);
		}

	private:
		friend class WindowEventCallback;
		template<class ChannelType>
		void add(Channels<ChannelType>& channels, core::smart_refctd_ptr<ChannelType>&& channel)
		{
			std::unique_lock lock(channels.lock);
			channels.channels.push_back(std::move(channel));
			channels.added.notify_all();
		}
		template<class ChannelType>
		void remove(Channels<ChannelType>& channels, const ChannelType* channel)
		{
			std::unique_lock lock(channels.lock);
			channels.channels.erase(
				std::find_if(
					channels.channels.begin(),channels.channels.end(),[channel](const auto& chan)->bool{return chan.get()==channel;}
				)
			);
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
				m_logger.log("Waiting For Input Device to be connected...",system::ILogger::ELL_INFO);
				channels.added.wait(lock);
			}

			auto current_default = channels.channels.front();
			if (reader->channel==current_default)
				return;

			reader->channel = current_default;
			reader->consumedCounter = 0u;
		}

		system::logger_opt_smart_ptr m_logger;
		Channels<IMouseEventChannel> m_mouse;
		Channels<IKeyboardEventChannel> m_keyboard;
};

// this is a callback necessary to handle a window
class WindowEventCallback : public IWindow::IEventCallback
{
public:
	WindowEventCallback(core::smart_refctd_ptr<InputSystem>&& inputSystem, system::logger_opt_smart_ptr&& logger) : m_inputSystem(std::move(inputSystem)), m_logger(std::move(logger)), m_gotWindowClosedMsg(false) {}

	bool isWindowOpen() const {return !m_gotWindowClosedMsg;}

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
		m_logger.log("Window window moved to { %d, %d }", system::ILogger::ELL_WARNING, x, y);
		return true;
	}
	bool onWindowResized_impl(uint32_t w, uint32_t h) override
	{
		m_logger.log("Window resized to { %u, %u }", system::ILogger::ELL_DEBUG, w, h);
		return true;
	}
	bool onWindowMinimized_impl() override
	{
		m_logger.log("Window minimized", system::ILogger::ELL_ERROR);
		return true;
	}
	bool onWindowMaximized_impl() override
	{
		m_logger.log("Window maximized", system::ILogger::ELL_PERFORMANCE);
		return true;
	}
	void onGainedMouseFocus_impl() override
	{
		m_logger.log("Window gained mouse focus", system::ILogger::ELL_INFO);
	}
	void onLostMouseFocus_impl() override
	{
		m_logger.log("Window lost mouse focus", system::ILogger::ELL_INFO);
	}
	void onGainedKeyboardFocus_impl() override
	{
		m_logger.log("Window gained keyboard focus", system::ILogger::ELL_INFO);
	}
	void onLostKeyboardFocus_impl() override
	{
		m_logger.log("Window lost keyboard focus", system::ILogger::ELL_INFO);
	}
	bool onWindowClosed_impl() override
	{
		m_logger.log("Window closed");
		m_gotWindowClosedMsg = true;
		return true;
	}

	void onMouseConnected_impl(core::smart_refctd_ptr<IMouseEventChannel>&& mch) override
	{
		m_logger.log("A mouse %p has been connected", system::ILogger::ELL_INFO, mch);
		m_inputSystem.get()->add(m_inputSystem.get()->m_mouse,std::move(mch));
	}
	void onMouseDisconnected_impl(IMouseEventChannel* mch) override
	{
		m_logger.log("A mouse %p has been disconnected", system::ILogger::ELL_INFO, mch);
		m_inputSystem.get()->remove(m_inputSystem.get()->m_mouse,mch);
	}
	void onKeyboardConnected_impl(core::smart_refctd_ptr<IKeyboardEventChannel>&& kbch) override
	{
		m_logger.log("A keyboard %p has been connected", system::ILogger::ELL_INFO, kbch);
		m_inputSystem.get()->add(m_inputSystem.get()->m_keyboard,std::move(kbch));
	}
	void onKeyboardDisconnected_impl(IKeyboardEventChannel* kbch) override
	{
		m_logger.log("A keyboard %p has been disconnected", system::ILogger::ELL_INFO, kbch);
		m_inputSystem.get()->remove(m_inputSystem.get()->m_keyboard,kbch);
	}

private:
	core::smart_refctd_ptr<InputSystem> m_inputSystem;
	system::logger_opt_smart_ptr m_logger;
	bool m_gotWindowClosedMsg;
};

int main(int argc, char** argv)
{
	// the application only needs to call this to delay-load Shared Libraries, if you have a static build, it will do nothing
	IApplicationFramework::GlobalsInit();
	// we will actually use `IApplicationFramework` in later samples

	const path CWD = path(argv[0]).parent_path().generic_string() + "/";
	const path mediaWD = CWD.generic_string() + "../../media/";

	auto system = IApplicationFramework::createSystem();
	// TODO: system->deleteFile("log.txt");

	// in this sample we will write some of the logger output to a file
	core::smart_refctd_ptr<system::ILogger> logger;
	{
		system::ISystem::future_t<smart_refctd_ptr<system::IFile>> future;
		system->createFile(future, CWD/"log.txt", nbl::system::IFile::ECF_READ_WRITE);
		if (future.wait())
			logger = core::make_smart_refctd_ptr<system::CFileLogger>(future.copy(), false);
	}

	auto assetManager = core::make_smart_refctd_ptr<IAssetManager>(smart_refctd_ptr(system));

	auto winManager = IWindowManagerWin32::create();
	

	IWindow::SCreationParams params;
	params.callback = nullptr;
	params.width = 720;
	params.height = 480;
	params.x = 500;
	params.y = 300;
	params.flags = IWindow::ECF_NONE;
	params.windowCaption = "Test Window";

	auto input = make_smart_refctd_ptr<InputSystem>(system::logger_opt_smart_ptr(smart_refctd_ptr(logger)));
	auto windowCb = make_smart_refctd_ptr<WindowEventCallback>(core::smart_refctd_ptr(input),system::logger_opt_smart_ptr(smart_refctd_ptr(logger)));
	params.callback = windowCb;
	// *********************************
	auto window = winManager->createWindow(std::move(params));
	auto* cursorControl = window->getCursorControl();

	ISystem::future_t<smart_refctd_ptr<system::IFile>> future;
	system->createFile(future, CWD/"testFile.txt", core::bitflag(nbl::system::IFile::ECF_READ_WRITE)/*Growing mappable files are a TODO |IFile::ECF_MAPPABLE*/);
	if (auto pFile = future.acquire())
	{
		auto& file = *pFile;
		const std::string fileData = "Test file data!";

		system::IFile::success_t writeSuccess;
		file->write(writeSuccess, fileData.data(), 0, fileData.length());
		{
			const bool success = bool(writeSuccess);
			assert(success);
		}

		std::string readStr(fileData.length(), '\0');
		system::IFile::success_t readSuccess;
		file->read(readSuccess, readStr.data(), 0, readStr.length());
		{
			const bool success = bool(readSuccess);
			assert(success);
		}
		assert(readStr == fileData);
	}
	else
	{
		assert(false);
	}


	// monitor input for N seconds
	using namespace std::chrono;
	auto timeout = seconds(~0u);
	if (argc>=3 && core::string("-timeout_seconds")==argv[1])
		timeout = seconds(std::atoi(argv[2]));
	for (auto start=steady_clock::now(); windowCb->isWindowOpen() && duration_cast<decltype(timeout)>(steady_clock::now()-start)<timeout;)
	{
		input->getDefaultMouse(&mouse);
		input->getDefaultKeyboard(&keyboard);

		mouse.consumeEvents(mouseProcess,logger.get());
		keyboard.consumeEvents(keyboardProcess,logger.get());
	}
	return 0;
}
