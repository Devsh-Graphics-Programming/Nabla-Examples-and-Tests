// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/examples/examples.hpp"
#include "nbl/ext/MitsubaLoader/CMitsubaLoader.h"
#include "nbl/ext/MitsubaLoader/CSerializedLoader.h"

#include <cctype>
#include <functional>


using namespace nbl;
using namespace nbl::core;
using namespace nbl::hlsl;
using namespace nbl::system;
using namespace nbl::asset;
using namespace nbl::ui;
using namespace nbl::video;
using namespace nbl::examples;

// Testing our Mitsuba Loader
class MitsubaLoaderTest final : public BuiltinResourcesApplication
{
		using base_t = BuiltinResourcesApplication;

		bool test(const system::path& listPath)
		{
			smart_refctd_ptr<const IFile> file;
			{
				ISystem::future_t<smart_refctd_ptr<IFile>> future;
				using create_flags_t = IFileBase::E_CREATE_FLAGS;
				m_system->createFile(future,listPath,create_flags_t::ECF_READ|create_flags_t::ECF_MAPPABLE);
				if (!future.wait())
					return logFail("Failed to list of scenes to test with path %s",listPath.string().c_str());
				smart_refctd_ptr<IFile> tmp;
				future.acquire().move_into(tmp);
				file = std::move(tmp);
			}
			if (!file)
				return logFail("Failed to open list of scenes to test with path %s",listPath.string().c_str());

			const auto base = file->getFileName().parent_path();
			const void* const ptr = file->getMappedPointer();
			const auto end = reinterpret_cast<const char*>(ptr)+file->getSize();
			for (auto cursor=reinterpret_cast<const char*>(ptr); cursor<end;)
			{
				cursor = std::find_if(cursor,end,std::not_fn<int(int)>(std::isspace));
				if (cursor==end)
					break;
				auto nextLine = [&]()->const char*
				{
					constexpr std::array<const char,2> newlines = {'\r','\n'};
					auto retval = std::find_first_of(cursor,end,newlines.begin(),newlines.end());
					while (++retval<end && std::find(newlines.begin(),newlines.end(),*retval)!=newlines.end()) {}
					return retval;
				};
				if (*cursor=='\"')
				{
					const auto begin = ++cursor;
					// find first unescaped "
					while (cursor!=end)
					{
						cursor = std::find(cursor,end,'\"');
						if (*(cursor-1)!='\\')
							break;
					}
					const auto relPath = (base/std::string_view(begin,cursor)).lexically_normal().string();
					//
					IAssetLoader::SAssetLoadParams params = {};
					params.logger = m_logger.get();
					auto asset = m_assetMgr->getAsset(relPath,params);
					if (asset.getContents().empty() || asset.getAssetType()!=IAsset::E_TYPE::ET_SCENE)
						return logFail("Failed To Load %s",relPath.c_str());
					m_logger->log("Loaded %s",ILogger::ELL_INFO,relPath.c_str());
					// TODO: print True Material IR
					// so we don't run out of RAM during testing
					m_assetMgr->clearAllAssetCache();
				}
				else if (*cursor!=';')
				{
					const char chr[2] = {*cursor,0};
					return logFail("Parser Error, encountered unsupprted character %s near line start",chr);
				}
				cursor = nextLine();
			}
			return true;
		}

	public:
		// Yay thanks to multiple inheritance we cannot forward ctors anymore
		MitsubaLoaderTest(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
			system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

		// we stuff all our work here because its a "single shot" app
		bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			if (!base_t::onAppInitialized(std::move(system)))
				return false;

			m_assetMgr->addAssetLoader(make_smart_refctd_ptr<ext::MitsubaLoader::CMitsubaLoader>(core::smart_refctd_ptr(m_system)));
			// some of our test scenes won't load without the `.serialized` support
			m_assetMgr->addAssetLoader(make_smart_refctd_ptr<ext::MitsubaLoader::CSerializedLoader>());

			// public batch
			if (!test(localInputCWD/"test_scenes.txt"))
				return false;
//			if (!test(sharedInputCWD/"Ditt-Reference-Scenes/private_test_scenes.txt"))
//				return false;

			return true;
		}

		// One-shot App
		bool keepRunning() override { return false; }

		// One-shot App
		void workLoopBody() override {}

		// Cleanup
		bool onAppTerminated() override
		{
			return base_t::onAppTerminated();
		}
};


NBL_MAIN_FUNC(MitsubaLoaderTest)