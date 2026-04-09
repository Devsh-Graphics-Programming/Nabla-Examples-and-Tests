#include "app/App.hpp"

namespace
{

template<typename ChannelReader, typename EventContainer>
inline void appendFocusedChannelEvents(
	ChannelReader& reader,
	IWindow& window,
	EventContainer& outEvents,
	ILogger* logger)
{
	reader.consumeEvents([&](const auto& events) -> void
	{
		if (!window.hasInputFocus())
			return;
		outEvents.insert(outEvents.end(), events.begin(), events.end());
	}, logger);
}

inline void scaleCapturedMouseEvents(
	std::vector<SMouseEvent>& mouseEvents,
	const CameraControlSettings& cameraControls)
{
	for (auto& event : mouseEvents)
	{
		if (event.type == ui::SMouseEvent::EET_SCROLL)
		{
			event.scrollEvent.verticalScroll *= cameraControls.mouseScrollScale;
			event.scrollEvent.horizontalScroll *= cameraControls.mouseScrollScale;
			continue;
		}

		if (event.type == ui::SMouseEvent::EET_MOVEMENT)
		{
			event.movementEvent.relativeMovementX *= cameraControls.mouseMoveScale;
			event.movementEvent.relativeMovementY *= cameraControls.mouseMoveScale;
		}
	}
}

} // namespace

void App::updatePresentationTiming()
{
	m_inputSystem->getDefaultMouse(&mouse);
	m_inputSystem->getDefaultKeyboard(&keyboard);

	oracle.reportEndFrameRecord();
	const auto timestamp = oracle.getNextPresentationTimeStamp();
	oracle.reportBeginFrameRecord();

	m_nextPresentationTimestamp = timestamp;
	if (m_presentationTiming.hasLastPresentationTimestamp)
	{
		const auto delta = m_nextPresentationTimestamp - m_presentationTiming.lastPresentationTimestamp;
		if (delta.count() < 0)
			m_presentationTiming.frameDeltaSec = 0.0;
		else
			m_presentationTiming.frameDeltaSec = std::chrono::duration<double>(delta).count();
	}
	m_presentationTiming.lastPresentationTimestamp = m_nextPresentationTimestamp;
	m_presentationTiming.hasLastPresentationTimestamp = true;
}

SCapturedUiEvents App::captureUiInputEvents()
{
	SCapturedUiEvents capturedEvents = {};
	appendFocusedChannelEvents(mouse, *m_window, capturedEvents.mouse, m_logger.get());
	appendFocusedChannelEvents(keyboard, *m_window, capturedEvents.keyboard, m_logger.get());
	return capturedEvents;
}

void App::buildCameraInputEvents(
	const SCapturedUiEvents& capturedEvents,
	std::vector<SKeyboardEvent>& outKeyboardEvents,
	std::vector<SMouseEvent>& outMouseEvents) const
{
	outKeyboardEvents = capturedEvents.keyboard;
	outMouseEvents = capturedEvents.mouse;
	scaleCapturedMouseEvents(outMouseEvents, m_cameraControls);
}

nbl::ext::imgui::UI::SUpdateParameters App::buildUiUpdateParameters(const SCapturedUiEvents& capturedEvents) const
{
	const auto cursorPosition = m_window->getCursorControl()->getPosition();
	return {
		.mousePosition = nbl::hlsl::float32_t2(cursorPosition.x, cursorPosition.y) - nbl::hlsl::float32_t2(m_window->getX(), m_window->getY()),
		.displaySize = { m_window->getWidth(), m_window->getHeight() },
		.mouseEvents = { capturedEvents.mouse.data(), capturedEvents.mouse.size() },
		.keyboardEvents = { capturedEvents.keyboard.data(), capturedEvents.keyboard.size() }
	};
}
