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
	using base_t::base_t;

	bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		if (!base_t::onAppInitialized(std::move(system)))
			return false;



		const bool isItDefaultImage = argc == 1;

		if (isItDefaultImage)
			m_logger->log("No image specified - loading a default OpenEXR image placed in CWD", ILogger::ELL_INFO);

		constexpr std::string_view defaultImagePath = "spp_benchmark_4k_512.exr";
		const std::string filePath = std::string(isItDefaultImage ? defaultImagePath.data() : argv[1]);


		auto assetManager = make_smart_refctd_ptr<nbl::asset::IAssetManager>(smart_refctd_ptr(m_system));

		nbl::asset::IAssetLoader::SAssetLoadParams lp;
	

		const asset::COpenEXRMetadata* meta;
		auto image_bundle = assetManager->getAsset(filePath, lp);
		auto contents = image_bundle.getContents();
		{
			bool status = !contents.empty();
			assert(status);
			status = meta = image_bundle.getMetadata()->selfCast<const COpenEXRMetadata>();
			assert(status);
		}

		uint32_t i = 0u;
		for (auto asset : contents)
		{
			auto image = IAsset::castDown<ICPUImage>(asset);
			const auto* metadata = static_cast<const COpenEXRMetadata::CImage*>(meta->getAssetSpecificMetadata(image.get()));

			ICPUImageView::SCreationParams imgViewParams;
			imgViewParams.flags = static_cast<ICPUImageView::E_CREATE_FLAGS>(0u);
			imgViewParams.image = std::move(image);
			imgViewParams.format = imgViewParams.image->getCreationParameters().format;
			imgViewParams.viewType = ICPUImageView::ET_2D;
			imgViewParams.subresourceRange = { static_cast<IImage::E_ASPECT_FLAGS>(0u),0u,1u,0u,1u };
			auto imageView = ICPUImageView::create(std::move(imgViewParams));

			auto channelsName = metadata->m_name;

			std::filesystem::path filename, extension;
			core::splitFilename(filePath.c_str(), nullptr, &filename, &extension);
			const std::string finalFileNameWithExtension = filename.string() + extension.string();
			const std::string finalOutputPath = channelsName.empty() ? (filename.string() + "_" + std::to_string(i++) + extension.string()) : (filename.string() + "_" + channelsName + extension.string());

			const auto writeParams = IAssetWriter::SAssetWriteParams(imageView.get(), EWF_BINARY);
			{
				bool status = assetManager->writeAsset(finalOutputPath, writeParams);
				assert(status);
			}
		}

		return true;
	}

	void workLoopBody() override {}

	bool keepRunning() override { return false; }

};


NBL_MAIN_FUNC(HelloComputeApp)