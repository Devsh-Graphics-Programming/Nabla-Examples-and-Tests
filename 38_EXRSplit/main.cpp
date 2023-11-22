// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


#include "../common/MonoSystemMonoLoggerApplication.hpp"

using namespace nbl;
using namespace core;
using namespace asset;
using namespace system;


// instead of defining our own `int main()` we derive from `nbl::system::IApplicationFramework` to play "nice" wil all platofmrs
class HelloComputeApp final : public nbl::examples::MonoSystemMonoLoggerApplication
{
	using base_t = examples::MonoSystemMonoLoggerApplication;
public:
	// Generally speaking because certain platforms delay initialization from main object construction you should just forward and not do anything in the ctor
	using base_t::base_t;

	// we stuff all our work here because its a "single shot" app
	bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		// Remember to call the base class initialization!
		if (!base_t::onAppInitialized(std::move(system)))
			return false;


		// Nabla's virtual filesystem has no notion of a Current Working Directory as its inherently thread-unsafe
		// everything operates on "absolute" paths
		const nbl::system::path CWD = path(argv[0]).parent_path().generic_string() + "/";

		// we assume you'll run the example `../..` relative to our media dir
		path mediaWD = CWD.generic_string() + "../../media/";

		auto assetManager = make_smart_refctd_ptr<nbl::asset::IAssetManager>(smart_refctd_ptr(m_system));

		// when assets are retrieved you need to provide parameters that control the loading process
		nbl::asset::IAssetLoader::SAssetLoadParams lp;
		// at the very least you need to provide the `workingDirectory` if your asset depends on others
		// this helps resolve relative paths for things such as textures
		lp.workingDirectory = mediaWD;
		

		auto checkedLoad = [&]<class T>(const string & key)->smart_refctd_ptr<T>
		{
			nbl::asset::SAssetBundle bundle = assetManager->getAsset(key, lp);
			if (bundle.getContents().empty())
			{
				m_logger->log("Asset %s failed to load! Are you sure it exists?", ILogger::ELL_ERROR, key.c_str());
				return nullptr;
			}
			// All assets derive from `nbl::asset::IAsset`, and can be casted down if the type matches
			static_assert(std::is_base_of_v<nbl::asset::IAsset, T>);
			// The type of the root assets in the bundle is not known until runtime, so this is kinda like a `dynamic_cast` which will return nullptr on type mismatch
			auto typedAsset = IAsset::castDown<T>(bundle.getContents()[0]); // just grab the first asset in the bundle
			if (!typedAsset)
				m_logger->log("Asset type mismatch want %d got %d !", ILogger::ELL_ERROR, T::AssetType, bundle.getAssetType());
			return typedAsset;
		};

		if (auto cpuImage = checkedLoad.operator() < nbl::asset::ICPUImage > ("noises/spp_benchmark_4k_512.exr"))
		{
			ICPUImageView::SCreationParams imgViewParams;
			imgViewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
			imgViewParams.format = cpuImage->getCreationParameters().format;
			imgViewParams.image = core::smart_refctd_ptr<ICPUImage>(cpuImage);
			imgViewParams.viewType = ICPUImageView::ET_2D;
			imgViewParams.subresourceRange = { static_cast<IImage::E_ASPECT_FLAGS>(0u),0u,1u,0u,1u };
			smart_refctd_ptr<nbl::asset::ICPUImageView> imageView = ICPUImageView::create(std::move(imgViewParams));

			nbl::asset::IAssetWriter::SAssetWriteParams wp(imageView.get());
			wp.workingDirectory = CWD;
			assetManager->writeAsset("sample_out.exr", wp);
		}


		return true;
	}

	void workLoopBody() override {}

	bool keepRunning() override { return false; }

};


NBL_MAIN_FUNC(HelloComputeApp)