// Copyright (C) 2018-2024 - DevSH GrapMonoAssetManagerAndBuiltinResourceApplicationhics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"

#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include <functional>
#include <condition_variable>

#include "nlohmann/json.hpp"
#include "argparse/argparse.hpp"

using json = nlohmann::json;

using namespace nbl;
using namespace core;
using namespace hlsl;
using namespace system;
using namespace asset;
using namespace ui;
using namespace video;

class ThreadPool {
public:
	using task_t = std::function<void()>;

	ThreadPool() {
		for (size_t i = 0; i < m_maxWorkers; i++) {
			m_threads.emplace_back([this]() {
				while (true) {
					task_t task;

					{
						std::unique_lock<std::mutex> lock(m_queueMutex);
						m_taskAvailable.wait(lock, [&]() { return !m_tasks.empty() || m_stop; });

						if (m_stop && m_tasks.empty())
							return;

						task = std::move(m_tasks.front());
						m_tasks.pop();
					}

					task();
				}
				});
		}
	}

	~ThreadPool() {
		{
			std::lock_guard<std::mutex> lock(m_queueMutex);
			m_stop = true;
		}

		m_taskAvailable.notify_all();

		for (auto& thread : m_threads) {
			thread.join();
		}
	}

	void addTask(task_t task) {
		std::lock_guard<std::mutex> queueLock(m_queueMutex);
		m_tasks.emplace(std::move(task));
		m_taskAvailable.notify_one();
	}
private:
	const size_t m_maxWorkers = std::thread::hardware_concurrency();
	std::vector<std::thread> m_threads;
	std::condition_variable m_taskAvailable;
	bool m_stop = false; // stops the threads on object destruction
	
	std::queue<task_t> m_tasks;
	std::mutex m_queueMutex;
};

class JpegLoaderApp final : public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
	using base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;
	using clock_t = std::chrono::steady_clock;
	constexpr static inline uint32_t m_maxWorkers = 8;
public:
	using base_t::base_t;

	inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		argparse::ArgumentParser program("Color Space");

		program.add_argument<std::string>("--directory")
			.required()
			.help("Path to a directory where all JPEG files are stored (not recursive)");

		program.add_argument<std::string>("--output")
			.default_value("output.json")
			.help("Path to the file where the benchmark result will be stored");

		try
		{
			program.parse_args({ argv.data(), argv.data() + argv.size() });
		}
		catch (const std::exception& err)
		{
			std::cerr << err.what() << std::endl << program; // NOTE: std::cerr because logger isn't initialized yet
			return 1;
		}

		if (!base_t::onAppInitialized(std::move(system)))
			return false;

		options.directory = program.get<std::string>("--directory");
		options.outputFile = program.get<std::string>("--output");


		ThreadPool threadPool;
		// TODO: Load JPEGs
		// TODO: Measure the time

		return true;
	}

	inline bool keepRunning() override
	{
		return false;
	}

	inline void workLoopBody() override
	{

	}
private:
	struct NBL_APP_OPTIONS
	{
		std::string directory;
		std::string outputFile;
	} options;
};

NBL_MAIN_FUNC(JpegLoaderApp)