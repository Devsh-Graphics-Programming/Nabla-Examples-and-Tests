// Regression test for `InputSystem::tryGetDefault*`: it must never block, whether devices are absent, connected,
// or disconnected while a frame is acquiring them. The blocking `getDefault*` must keep waiting for a device.
// Exit codes: 0 pass, 1 failed check, 2 timed out (blocked).
#include "nabla.h"
#include "nbl/examples/common/InputSystem.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <thread>

using namespace nbl;
using namespace nbl::ui;
using namespace nbl::examples;

namespace
{
constexpr auto Timeout = std::chrono::seconds(10);
constexpr size_t ChannelCapacity = 64u;

std::atomic_bool g_failed = false;
#define NBL_CHECK(COND) if (!(COND)) { std::fprintf(stderr,"FAILED: %s (%s:%d)\n",#COND,__FILE__,__LINE__); g_failed = true; }

// Runs a case on its own thread, a case still running after `Timeout` is blocked and can't be joined
template<typename F>
void runBounded(const char* name, F&& testCase)
{
	std::packaged_task<void()> task(std::forward<F>(testCase));
	auto done = task.get_future();
	std::thread(std::move(task)).detach();
	if (done.wait_for(Timeout)!=std::future_status::ready)
	{
		std::fprintf(stderr,"TIMED OUT: %s blocked for more than %lld s\n",name,static_cast<long long>(Timeout.count()));
		std::fflush(stderr);
		std::_Exit(2);
	}
	done.get();
	std::printf("%s: done\n",name);
}

auto makeMouse() { return core::make_smart_refctd_ptr<IMouseEventChannel>(ChannelCapacity); }
auto makeKeyboard() { return core::make_smart_refctd_ptr<IKeyboardEventChannel>(ChannelCapacity); }
auto makeInputSystem() { return core::make_smart_refctd_ptr<InputSystem>(system::logger_opt_smart_ptr(nullptr)); }
}

int main()
{
	runBounded("absent devices",[]() -> void
	{
		auto input = makeInputSystem();
		InputSystem::ChannelReader<IMouseEventChannel> mouse;
		InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;
		NBL_CHECK(!input->tryGetDefaultMouse(&mouse));
		NBL_CHECK(!input->tryGetDefaultKeyboard(&keyboard));
		NBL_CHECK(!mouse.channel && !keyboard.channel);
	});

	runBounded("connected devices",[]() -> void
	{
		auto input = makeInputSystem();
		auto mouseChannel = makeMouse();
		auto keyboardChannel = makeKeyboard();
		input->add(input->m_mouse,core::smart_refctd_ptr(mouseChannel));
		input->add(input->m_keyboard,core::smart_refctd_ptr(keyboardChannel));

		InputSystem::ChannelReader<IMouseEventChannel> mouse;
		InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;
		NBL_CHECK(input->tryGetDefaultMouse(&mouse) && input->tryGetDefaultKeyboard(&keyboard));
		NBL_CHECK(mouse.channel==mouseChannel && keyboard.channel==keyboardChannel);
		// a repeat selection of the same channel keeps the reader's position
		mouse.consumedCounter = 7u;
		NBL_CHECK(input->tryGetDefaultMouse(&mouse) && mouse.consumedCounter==7u);
	});

	runBounded("disconnect after acquisition",[]() -> void
	{
		auto input = makeInputSystem();
		input->add(input->m_mouse,makeMouse());
		InputSystem::ChannelReader<IMouseEventChannel> mouse;
		NBL_CHECK(input->tryGetDefaultMouse(&mouse));
		const IMouseEventChannel* const acquired = mouse.channel.get();

		// the last mouse goes away, the next frame must fall back instead of waiting
		input->remove(input->m_mouse,acquired);
		NBL_CHECK(!input->tryGetDefaultMouse(&mouse));
		// the reader's reference keeps the removed channel alive and readable
		NBL_CHECK(mouse.channel.get()==acquired && mouse.channel->getEvents().size()==0u);

		// reconnecting binds the new device
		auto reconnected = makeMouse();
		input->add(input->m_mouse,core::smart_refctd_ptr(reconnected));
		NBL_CHECK(input->tryGetDefaultMouse(&mouse) && mouse.channel==reconnected);
	});

	// The NAB-16 interleaving: the event thread drops the last mouse while the frame acquires it, and nothing
	// reconnects until that frame is done. The old check-then-`getDefault` sequence waits forever when it loses.
	runBounded("disconnect during acquisition",[]() -> void
	{
		constexpr uint32_t Rounds = 20000u;
		auto input = makeInputSystem();
		input->add(input->m_keyboard,makeKeyboard());

		std::atomic_uint32_t go = 0u, removed = 0u;
		std::atomic<const IMouseEventChannel*> pending = nullptr;
		std::thread eventThread([&]() -> void
		{
			for (uint32_t round=1u; round<=Rounds; round++)
			{
				while (go.load()!=round)
					std::this_thread::yield();
				input->remove(input->m_mouse,pending.load());
				removed = round;
			}
		});

		uint32_t withInput = 0u;
		InputSystem::ChannelReader<IMouseEventChannel> mouse;
		InputSystem::ChannelReader<IKeyboardEventChannel> keyboard;
		for (uint32_t round=1u; round<=Rounds; round++)
		{
			auto channel = makeMouse();
			pending = channel.get();
			input->add(input->m_mouse,std::move(channel));
			go = round;
			// sweep the interleaving, so the disconnect lands before, during and after the acquisition
			for (volatile uint32_t spin=(round*2654435761u)%2048u; spin; spin--) {}
			const bool hasInput = input->tryGetDefaultMouse(&mouse) && input->tryGetDefaultKeyboard(&keyboard);
			if (hasInput)
			{
				withInput++;
				NBL_CHECK(mouse.channel && keyboard.channel);
			}
			while (removed.load()!=round)
				std::this_thread::yield();
			NBL_CHECK(!input->tryGetDefaultMouse(&mouse));
		}
		eventThread.join();
		std::printf("  %u rounds, %u acquired before the disconnect\n",Rounds,withInput);
	});

	runBounded("blocking getDefault still waits",[]() -> void
	{
		auto input = makeInputSystem();
		auto mouseChannel = makeMouse();
		std::atomic_bool added = false;
		std::thread connector([&]() -> void
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(200));
			added = true;
			input->add(input->m_mouse,core::smart_refctd_ptr(mouseChannel));
		});
		InputSystem::ChannelReader<IMouseEventChannel> mouse;
		input->getDefaultMouse(&mouse);
		NBL_CHECK(added && mouse.channel==mouseChannel);
		connector.join();
	});

	if (g_failed)
		return 1;
	std::printf("PASSED\n");
	return 0;
}
