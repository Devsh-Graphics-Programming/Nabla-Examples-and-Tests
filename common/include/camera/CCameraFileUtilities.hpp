#ifndef _C_CAMERA_FILE_UTILITIES_HPP_
#define _C_CAMERA_FILE_UTILITIES_HPP_

#include <string>
#include <string_view>
#include <vector>

#include "nbl/system/ISystem.h"

namespace nbl::system
{

inline bool readBinaryFile(
    ISystem& system,
    const path& filePath,
    std::vector<uint8_t>& outPayload,
    std::string* error = nullptr,
    const std::string_view openError = {})
{
    ISystem::future_t<core::smart_refctd_ptr<IFile>> future;
    system.createFile(future, filePath, IFile::ECF_READ | IFile::ECF_MAPPABLE);
    auto file = future.acquire();
    if (!file || !file->get())
    {
        if (error && !openError.empty())
            *error = std::string(openError);
        return false;
    }

    auto& input = *file->get();
    const auto fileSize = input.getSize();
    outPayload.resize(fileSize);
    if (outPayload.empty())
        return true;

    IFile::success_t readResult;
    input.read(readResult, outPayload.data(), 0, fileSize);
    if (!static_cast<bool>(readResult))
    {
        if (error && !openError.empty())
            *error = std::string(openError);
        return false;
    }
    return true;
}

inline bool readTextFile(
    ISystem& system,
    const path& filePath,
    std::string& outText,
    std::string* error = nullptr,
    const std::string_view openError = {})
{
    std::vector<uint8_t> payload;
    if (!readBinaryFile(system, filePath, payload, error, openError))
        return false;

    outText.assign(reinterpret_cast<const char*>(payload.data()), payload.size());
    return true;
}

inline bool writeTextFile(
    ISystem& system,
    const path& filePath,
    const std::string_view text)
{
    ISystem::future_t<core::smart_refctd_ptr<IFile>> future;
    system.createFile(future, filePath, IFile::ECF_WRITE);
    auto file = future.acquire();
    if (!file || !file->get())
        return false;
    if (text.empty())
        return true;

    IFile::success_t writeResult;
    (*file)->write(writeResult, text.data(), 0, text.size());
    return static_cast<bool>(writeResult);
}

} // namespace nbl::system

#endif // _C_CAMERA_FILE_UTILITIES_HPP_
