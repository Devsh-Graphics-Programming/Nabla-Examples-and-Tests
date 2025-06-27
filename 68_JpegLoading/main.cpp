// Copyright (C) 2018-2024 - DevSH GrapMonoAssetManagerAndBuiltinResourceApplicationhics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


#include "nbl/examples/examples.hpp"

#include <future>

#include "nlohmann/json.hpp"
#include "argparse/argparse.hpp"


using json = nlohmann::json;

using namespace nbl;
using namespace nbl::core;
using namespace nbl::hlsl;
using namespace nbl::system;
using namespace nbl::asset;
using namespace nbl::ui;
using namespace nbl::video;
using namespace nbl::examples;

class ThreadPool
{
using task_t = std::function<void()>;
public:
   ThreadPool(size_t workers = std::thread::hardware_concurrency())
   {
      for (size_t i = 0; i < workers; i++)
      {
         m_workers.emplace_back([this] {
            task_t task;

            while (1)
            {
               {
                  std::unique_lock<std::mutex> lock(m_queueLock);
                  m_taskAvailable.wait(lock, [this] { return !m_tasks.empty() || m_shouldStop; });

                  if (m_shouldStop && m_tasks.empty()) {
                     return;
                  }

                  task = std::move(m_tasks.front());
                  m_tasks.pop();
               }

               task();
            }
         });
      }
   }

   ~ThreadPool()
   {
      m_shouldStop = true;
      m_taskAvailable.notify_all();

      for (auto& worker : m_workers)
      {
         worker.join();
      }
   }

   void enqueue(task_t task)
   {
      {
         std::lock_guard<std::mutex> lock(m_queueLock);
         m_tasks.emplace(std::move(task));
      }
      m_taskAvailable.notify_one();
   }
   private:
   std::mutex m_queueLock;
   std::condition_variable m_taskAvailable;
   std::vector<std::thread> m_workers;
   std::queue<task_t> m_tasks;
   std::atomic<bool> m_shouldStop = false;
};

class JpegLoaderApp final : public BuiltinResourcesApplication
{
   using clock_t = std::chrono::steady_clock;
   using clock_resolution_t = std::chrono::milliseconds;
   using base_t = BuiltinResourcesApplication;
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
         return false;
      }

      if (!base_t::onAppInitialized(std::move(system)))
         return false;

      options.directory = program.get<std::string>("--directory");
      options.outputFile = program.get<std::string>("--output");

      // check if directory exists
      if (!std::filesystem::exists(options.directory)) 
      {
         logFail("Provided directory doesn't exist");
         return false;
      }

      auto start = clock_t::now();
      std::vector<std::string> files;

      {
         ThreadPool tp;

         constexpr auto cachingFlags = static_cast<IAssetLoader::E_CACHING_FLAGS>(IAssetLoader::ECF_DONT_CACHE_REFERENCES & IAssetLoader::ECF_DONT_CACHE_TOP_LEVEL);
         const IAssetLoader::SAssetLoadParams loadParams(0ull, nullptr, cachingFlags, IAssetLoader::ELPF_NONE, m_logger.get());

         for (auto& item : std::filesystem::directory_iterator(options.directory))
         {
            auto& path = item.path();
            if (path.has_extension() && path.extension() == ".jpg")
            {
               files.emplace_back(std::move(path.generic_string()));

               ISystem::future_t<smart_refctd_ptr<system::IFile>> future;
               m_system->createFile(future, path, IFile::ECF_READ | IFile::ECF_MAPPABLE);

               if (auto pFile = future.acquire(); pFile && pFile->get()) 
               {
                  auto& file = *pFile;
                  tp.enqueue([=] {
                     m_logger->log("Loading %S", ILogger::ELL_INFO, path.c_str());
                     m_assetMgr->getAsset(file.get(), path.generic_string(), loadParams);
                  });
               }
            }
         }
      }

      auto stop = clock_t::now();
      auto time = std::chrono::duration_cast<clock_resolution_t>(stop - start).count();

      m_logger->log("Process took %llu ms", ILogger::ELL_INFO, time);

      // Dump data to JSON
      json j;
      j["loaded_files"] = files;
      j["duration_ms"] = time;
      
      std::ofstream output(options.outputFile);
      if (!output.good())
      {
         logFail("Failed to open %S", options.outputFile);
         return false;
      }

      output << j;

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