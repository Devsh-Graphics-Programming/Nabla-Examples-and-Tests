// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_THIS_EXAMPLE_FILE_UTILITIES_HPP_INCLUDED_
#define _NBL_THIS_EXAMPLE_FILE_UTILITIES_HPP_INCLUDED_

#include <string>
#include <string_view>

#include "nbl/asset/ICPUBuffer.h"
#include "nbl/system/IFile.h"
#include "nbl/system/ISystem.h"

/// @brief Whole-file read/write helpers shared by the camera persistence, the scripted-runtime loader and the app's resource loaders.
///
/// Nothing here is camera specific; the helpers only give that code one consistent way to
/// open a file through `ISystem` and to report why that failed.
struct CFileUtilities final
{
public:
    /// @brief Read a whole file into a new CPU buffer, returns `nullptr` when the file cannot be opened or read.
    static inline nbl::core::smart_refctd_ptr<nbl::asset::ICPUBuffer> readBinaryFile(
        nbl::system::ISystem& system,
        const nbl::system::path& filePath,
        std::string* error = nullptr,
        const std::string_view openError = {})
    {
        nbl::system::ISystem::future_t<nbl::core::smart_refctd_ptr<nbl::system::IFile>> future;
        system.createFile(future, filePath, nbl::system::IFile::ECF_READ | nbl::system::IFile::ECF_MAPPABLE);
        auto file = future.acquire();
        if (!file || !file->get())
        {
            if (error && !openError.empty())
                *error = std::string(openError);
            return nullptr;
        }

        auto& input = *file->get();
        const auto fileSize = input.getSize();

        nbl::asset::ICPUBuffer::SCreationParams params = {};
        params.size = fileSize;
        auto buffer = nbl::asset::ICPUBuffer::create(std::move(params));
        if (!buffer)
        {
            if (error && !openError.empty())
                *error = std::string(openError);
            return nullptr;
        }
        if (fileSize == 0ull)
            return buffer;

        nbl::system::IFile::success_t readResult;
        input.read(readResult, buffer->getPointer(), 0, fileSize);
        if (!static_cast<bool>(readResult))
        {
            if (error && !openError.empty())
                *error = std::string(openError);
            return nullptr;
        }
        return buffer;
    }

    /// @brief Read a whole file and interpret its payload as UTF-8 text.
    static inline bool readTextFile(
        nbl::system::ISystem& system,
        const nbl::system::path& filePath,
        std::string& outText,
        std::string* error = nullptr,
        const std::string_view openError = {})
    {
        const auto payload = readBinaryFile(system, filePath, error, openError);
        if (!payload)
            return false;

        outText.assign(reinterpret_cast<const char*>(payload->getPointer()), payload->getSize());
        return true;
    }

    /// @brief Overwrite a file with the provided text payload.
    static inline bool writeTextFile(
        nbl::system::ISystem& system,
        const nbl::system::path& filePath,
        const std::string_view text)
    {
        nbl::system::ISystem::future_t<nbl::core::smart_refctd_ptr<nbl::system::IFile>> future;
        system.createFile(future, filePath, nbl::system::IFile::ECF_WRITE);
        auto file = future.acquire();
        if (!file || !file->get())
            return false;
        if (text.empty())
            return true;

        nbl::system::IFile::success_t writeResult;
        (*file)->write(writeResult, text.data(), 0, text.size());
        return static_cast<bool>(writeResult);
    }
};

#endif // _NBL_THIS_EXAMPLE_FILE_UTILITIES_HPP_INCLUDED_
