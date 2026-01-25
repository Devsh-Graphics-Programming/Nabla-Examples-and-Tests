// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include <nbl/examples/common/BuiltinResourcesApplication.hpp>

#include <algorithm>
#include <array>
#include <deque>
#include <string>
#include <string_view>
#include <vector>

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

        const path zipPath = sharedInputCWD / "mitsuba/bedroom.zip";
        auto archive = m_system->openFileArchive(zipPath);
        if (!archive)
        {
            m_logger->log("Failed to open zip archive: %s", ILogger::ELL_ERROR, zipPath.string().c_str());
            return false;
        }

        auto archiveFiles = IFileArchive::SFileList::span_t(archive->listAssets());
        if (archiveFiles.empty())
        {
            m_logger->log("Zip archive is empty: %s", ILogger::ELL_ERROR, zipPath.string().c_str());
            return false;
        }

        const path scenePath = "scene.xml";
        auto sceneIt = std::find_if(archiveFiles.begin(), archiveFiles.end(), [&scenePath](const auto& entry)
            {
                return entry.pathRelativeToArchive == scenePath;
            });
        if (sceneIt == archiveFiles.end())
        {
            m_logger->log("Zip archive missing scene.xml: %s", ILogger::ELL_ERROR, zipPath.string().c_str());
            return false;
        }

        auto sceneFile = archive->getFile(scenePath, IFileBase::ECF_READ, "");
        if (!sceneFile)
        {
            m_logger->log("Failed to open scene.xml from zip: %s", ILogger::ELL_ERROR, zipPath.string().c_str());
            return false;
        }

        if (sceneIt->size == 0 || sceneFile->getSize() != sceneIt->size)
        {
            m_logger->log("scene.xml size mismatch in zip: %s", ILogger::ELL_ERROR, zipPath.string().c_str());
            return false;
        }

        const size_t probeSize = std::min<size_t>(sceneIt->size, 64u);
        std::array<char, 64> probe{};
        IFile::success_t probeRead;
        sceneFile->read(probeRead, probe.data(), 0, probeSize);
        if (!probeRead)
        {
            m_logger->log("Failed to read scene.xml from zip: %s", ILogger::ELL_ERROR, zipPath.string().c_str());
            return false;
        }

        const std::string_view probeView(probe.data(), probeSize);
        if (probeView.find("<?xml") == std::string_view::npos)
        {
            m_logger->log("scene.xml header is unexpected in zip: %s", ILogger::ELL_ERROR, zipPath.string().c_str());
            return false;
        }

        const size_t linesToPrint = 6u;
        const char* mapped = static_cast<const char*>(sceneFile->getMappedPointer());
        if (mapped)
        {
            std::vector<std::string_view> headLines;
            headLines.reserve(linesToPrint);
            std::deque<std::string_view> tailLines;

            size_t lineStart = 0;
            for (size_t i = 0; i < sceneIt->size; ++i)
            {
                if (mapped[i] != '\n')
                    continue;

                size_t lineLen = i - lineStart;
                if (lineLen && mapped[i - 1] == '\r')
                    --lineLen;

                const std::string_view line(mapped + lineStart, lineLen);
                if (headLines.size() < linesToPrint)
                    headLines.push_back(line);
                if (tailLines.size() == linesToPrint)
                    tailLines.pop_front();
                tailLines.push_back(line);
                lineStart = i + 1;
            }
            if (lineStart < sceneIt->size)
            {
                size_t lineLen = sceneIt->size - lineStart;
                if (lineLen && mapped[sceneIt->size - 1] == '\r')
                    --lineLen;
                const std::string_view line(mapped + lineStart, lineLen);
                if (headLines.size() < linesToPrint)
                    headLines.push_back(line);
                if (tailLines.size() == linesToPrint)
                    tailLines.pop_front();
                tailLines.push_back(line);
            }

            std::string head;
            for (const auto& line : headLines)
            {
                head.append(line);
                head.push_back('\n');
            }
            std::string tail;
            for (const auto& line : tailLines)
            {
                tail.append(line);
                tail.push_back('\n');
            }

            m_logger->log("scene.xml head (%u lines):\n%s", ILogger::ELL_INFO, static_cast<uint32_t>(headLines.size()), head.c_str());
            m_logger->log("scene.xml tail (%u lines):\n%s", ILogger::ELL_INFO, static_cast<uint32_t>(tailLines.size()), tail.c_str());
        }
        else
        {
            std::vector<std::string> headLines;
            headLines.reserve(linesToPrint);
            std::deque<std::string> tailLines;
            std::string carry;
            const size_t chunkSize = 64u * 1024u;
            std::string buffer(chunkSize, '\0');
            size_t offset = 0;
            while (offset < sceneIt->size)
            {
                const size_t toRead = std::min(chunkSize, sceneIt->size - offset);
                IFile::success_t chunkRead;
                sceneFile->read(chunkRead, buffer.data(), offset, toRead);
                if (!chunkRead)
                {
                    m_logger->log("Failed to read scene.xml from zip: %s", ILogger::ELL_ERROR, zipPath.string().c_str());
                    return false;
                }

                size_t lineStart = 0;
                for (size_t i = 0; i < toRead; ++i)
                {
                    if (buffer[i] != '\n')
                        continue;

                    size_t lineEnd = i;
                    if (lineEnd > lineStart && buffer[lineEnd - 1] == '\r')
                        --lineEnd;

                    std::string line;
                    if (!carry.empty())
                    {
                        line = carry;
                        if (lineEnd > lineStart)
                            line.append(buffer.data() + lineStart, lineEnd - lineStart);
                        if (!line.empty() && line.back() == '\r')
                            line.pop_back();
                        carry.clear();
                    }
                    else
                    {
                        line.assign(buffer.data() + lineStart, lineEnd - lineStart);
                    }

                    if (headLines.size() < linesToPrint)
                        headLines.push_back(line);
                    if (tailLines.size() == linesToPrint)
                        tailLines.pop_front();
                    tailLines.push_back(std::move(line));
                    lineStart = i + 1;
                }

                if (lineStart < toRead)
                {
                    const size_t tailSize = toRead - lineStart;
                    if (carry.empty())
                        carry.assign(buffer.data() + lineStart, tailSize);
                    else
                        carry.append(buffer.data() + lineStart, tailSize);
                }

                offset += toRead;
            }
            if (!carry.empty())
            {
                if (!carry.empty() && carry.back() == '\r')
                    carry.pop_back();
                if (headLines.size() < linesToPrint)
                    headLines.push_back(carry);
                if (tailLines.size() == linesToPrint)
                    tailLines.pop_front();
                tailLines.push_back(carry);
                carry.clear();
            }

            std::string head;
            for (const auto& line : headLines)
            {
                head.append(line);
                head.push_back('\n');
            }
            std::string tail;
            for (const auto& line : tailLines)
            {
                tail.append(line);
                tail.push_back('\n');
            }

            m_logger->log("scene.xml head (%u lines):\n%s", ILogger::ELL_INFO, static_cast<uint32_t>(headLines.size()), head.c_str());
            m_logger->log("scene.xml tail (%u lines):\n%s", ILogger::ELL_INFO, static_cast<uint32_t>(tailLines.size()), tail.c_str());
        }

        std::stringstream ss;
        for (const auto& file : archiveFiles)
        {
            ss << "ID: " << file.ID;
            ss << " offset: " << file.offset;
            ss << " path relative od archive: " << file.pathRelativeToArchive;
            ss << " size: " << file.size << '\n';
        }

        m_logger->log(ss.str().c_str(), ILogger::ELL_PERFORMANCE);

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
