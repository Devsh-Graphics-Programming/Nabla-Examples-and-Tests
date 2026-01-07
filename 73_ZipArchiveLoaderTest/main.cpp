// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include <nbl/examples/common/BuiltinResourcesApplication.hpp>

using namespace nbl;
using namespace nbl::core;
using namespace nbl::system;
using namespace nbl::asset;
using namespace nbl::examples;

class ZipArchiveLoaderTest final : public BuiltinResourcesApplication
{
    using asset_base_t = BuiltinResourcesApplication;

public:
    ZipArchiveLoaderTest(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
        IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {
    }

    bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
    {
        if (!asset_base_t::onAppInitialized(std::move(system)))
            return false;

        const std::filesystem::path zipPath = sharedInputCWD / "mitsuba/bedroom.zip";
        auto archive = m_system->openFileArchive(zipPath);

        auto archiveFiles = IFileArchive::SFileList::span_t(archive->listAssets());

        std::stringstream ss;
        for (const auto& file : archiveFiles)
        {
            ss << "ID: " << file.ID;
            ss << " offset: " << file.offset;
            ss << " path relative od archive: " << file.pathRelativeToArchive;
            ss << " size: " << file.size << '\n';
        }

        m_logger->log(ss.str().c_str(), ILogger::ELL_PERFORMANCE);

        // TODO: test GZIP files and ZIP files with AES encryption

        return true;
    }

    void onAppTerminated_impl() override
    {
    }

    void workLoopBody() override
    {
    }

    bool keepRunning() override
    {
        return false;
    }
};

NBL_MAIN_FUNC(ZipArchiveLoaderTest)