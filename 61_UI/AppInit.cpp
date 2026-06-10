#include "app/App.hpp"
#include "app/AppResourceUtilities.hpp"

bool App::onAppInitialized(smart_refctd_ptr<ISystem>&& system)
{
	argparse::ArgumentParser program("Virtual camera event system demo");

	program.add_argument<std::string>("--file")
		.help("Path to json file with camera inputs");
	program.add_argument("--ci")
		.help("Run in CI mode: capture a screenshot after a few frames and exit.")
		.default_value(false)
		.implicit_value(true);
	program.add_argument<std::string>("--script")
		.help("Path to json file with scripted input events");
	program.add_argument("--script-log")
		.help("Log scripted input and virtual events.")
		.default_value(false)
		.implicit_value(true);
	program.add_argument("--script-visual-debug")
		.help("Enable scripted visual debug overlay and fixed frame pacing.")
		.default_value(false)
		.implicit_value(true);
	program.add_argument("--no-screenshots")
		.help("Disable CI and scripted screenshot captures.")
		.default_value(false)
		.implicit_value(true);
	program.add_argument("--headless-camera-smoke")
		.help("Run a headless camera-only smoke test and exit after initialization.")
		.default_value(false)
		.implicit_value(true);

	try
	{
		program.parse_args({ argv.data(), argv.data() + argv.size() });
	}
	catch (const std::exception& err)
	{
		std::cerr << err.what() << std::endl << program;
		return false;
	}

	m_cliRuntime.headlessCameraSmokeMode = program.get<bool>("--headless-camera-smoke");
	if (m_cliRuntime.headlessCameraSmokeMode)
		return runHeadlessCameraSmoke(program, std::move(system));

	m_cliRuntime.ciMode = program.get<bool>("--ci");
	if (m_cliRuntime.ciMode)
	{
		m_cliRuntime.ciScreenshotPath = localOutputCWD / "cameraz_ci.png";
		m_cliRuntime.ciStartedAt = clock_t::now();
		m_viewports.useWindow = true;
	}
	m_scriptedInput.log = program.get<bool>("--script-log");
	m_cliRuntime.scriptVisualDebugCli = program.get<bool>("--script-visual-debug");
	m_cliRuntime.disableScreenshotsCli = program.get<bool>("--no-screenshots");

	m_inputSystem = make_smart_refctd_ptr<InputSystem>(logger_opt_smart_ptr(smart_refctd_ptr(m_logger)));
	m_logFormatter = core::make_smart_refctd_ptr<CUILogFormatter>();

	if (!base_t::onAppInitialized(smart_refctd_ptr(system)))
		return false;
	if (!initializeMountedCameraResources(std::move(system)))
		return false;

	if (!initializeCameraConfiguration(program))
		return false;
	if (!initializePresentationResources())
		return false;
	if (!initializeUiResources())
		return false;
	if (!initializeSceneResources())
		return false;

	oracle.reportBeginFrameRecord();

	if (base_t::argv.size() >= 3 && argv[1] == "-timeout_seconds")
		timeout = std::chrono::seconds(std::atoi(argv[2].c_str()));
	start = clock_t::now();
	return true;
}
