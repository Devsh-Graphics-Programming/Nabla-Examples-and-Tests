#include "app/App.hpp"

void App::workLoopBody()
{
	paceScriptedVisualDebugFrame();
	if (!waitForInflightFrameSlot())
		return;

	auto frameContext = tryBuildFrameSubmissionContext();
	if (!frameContext.has_value())
		return;

	update();
	if (!recordFramePasses(*frameContext))
		return;
	(void)submitAndPresentFrame(*frameContext);
}


