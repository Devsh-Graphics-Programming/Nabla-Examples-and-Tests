// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/application_templates/MonoSystemMonoLoggerApplication.hpp"

using namespace nbl;
using namespace core;
using namespace asset;
using namespace system;

// instead of defining our own `int main()` we derive from `nbl::system::IApplicationFramework` to play "nice" wil all platofmrs
class HelloComputeApp final : public nbl::application_templates::MonoSystemMonoLoggerApplication
{
	using base_t = application_templates::MonoSystemMonoLoggerApplication;
public:
	using base_t::base_t;

	bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		if (!base_t::onAppInitialized(std::move(system)))
			return false;

		constexpr std::string_view defaultImagePath = "../../media/noises/spp_benchmark_4k_512.exr";

		const auto targetFilePath = [&]() -> std::string_view
		{
			const auto argc = argv.size();
			const bool isDefaultImageRequested = argc == 1;

			if (isDefaultImageRequested)
			{
				m_logger->log("No image specified, loading default \"%s\" OpenEXR image from media directory!", ILogger::ELL_INFO, defaultImagePath.data());
				return defaultImagePath;
			}
			else if (argc == 2)
			{
				const std::string_view target(argv[1]);
				m_logger->log("Requested \"%s\"", ILogger::ELL_INFO, target.data());
				return { target };
			}
			else
			{
				m_logger->log("To many arguments! Pass a single filename to an OpenEXR image w.r.t CWD.", ILogger::ELL_ERROR);
				return {};
			}
		}();

		if (targetFilePath.empty())
			return false;
			
		auto assetManager = make_smart_refctd_ptr<nbl::asset::IAssetManager>(smart_refctd_ptr(m_system));

		nbl::asset::IAssetLoader::SAssetLoadParams lp;
		const asset::COpenEXRMetadata* meta;

		auto image_bundle = assetManager->getAsset(targetFilePath.data(), lp);
		auto contents = image_bundle.getContents();
		{
			if (contents.empty())
			{
				m_logger->log("Could not load \"%s\"", ILogger::ELL_ERROR, targetFilePath.data());
				return false;
			}

			meta = image_bundle.getMetadata()->selfCast<const COpenEXRMetadata>();

			if (!meta)
			{
				m_logger->log("Could not selfCast \"%s\" asset's metadata to COpenEXRMetadata, the tool expects valid OpenEXR input image, terminating!", ILogger::ELL_ERROR, targetFilePath.data());
				return false;
			}
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
			core::splitFilename(targetFilePath.data(), nullptr, &filename, &extension);

			const std::string finalFileNameWithExtension = filename.string() + extension.string();
			const std::string finalOutputPath = channelsName.empty() ? (filename.string() + "_" + std::to_string(i++) + extension.string()) : (filename.string() + "_" + channelsName + extension.string());

			const auto writeParams = IAssetWriter::SAssetWriteParams(imageView.get(), EWF_BINARY);
			if (assetManager->writeAsset(finalOutputPath, writeParams))
				m_logger->log("Saved \"%s\"!", ILogger::ELL_INFO, finalOutputPath.c_str());
			else
			{
				m_logger->log("Could not save \"%s\", terminating!", ILogger::ELL_ERROR, finalOutputPath.c_str());
				return false;
			}		
		}

		return true;
	}

	void workLoopBody() override {}

	bool keepRunning() override { return false; }

};

NBL_MAIN_FUNC(HelloComputeApp)