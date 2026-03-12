// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXAMPLES_COMMON_IMAGE_COMPARISON_INCLUDED_
#define _NBL_EXAMPLES_COMMON_IMAGE_COMPARISON_INCLUDED_

#include "nbl/asset/ICPUImage.h"
#include "nbl/asset/ICPUImageView.h"
#include "nbl/asset/format/EFormat.h"

#include <cstring>

namespace nbl::examples::image
{

struct SCodeUnitDiff
{
    static inline uint32_t resolveCodeUnitBytes(const asset::E_FORMAT format)
    {
        if (asset::isBlockCompressionFormat(format))
            return 1u;
        const uint32_t channels = asset::getFormatChannelCount(format);
        if (!channels)
            return 1u;
        const uint32_t texelBytes = asset::getTexelOrBlockBytesize(format);
        if (!texelBytes || (texelBytes % channels) != 0u)
            return 1u;
        const uint32_t bytesPerChannel = texelBytes / channels;
        if (bytesPerChannel == 1u || bytesPerChannel == 2u || bytesPerChannel == 4u)
            return bytesPerChannel;
        return 1u;
    }

    template<typename T>
    static inline uint32_t absoluteDiff(const T a, const T b)
    {
        return (a >= b) ? static_cast<uint32_t>(a - b) : static_cast<uint32_t>(b - a);
    }
};

inline bool compareCpuImageViewsByCodeUnit(
    const asset::ICPUImageView* a,
    const asset::ICPUImageView* b,
    uint64_t& diffCodeUnitCount,
    uint32_t& maxDiffCodeUnitValue
)
{
    diffCodeUnitCount = 0u;
    maxDiffCodeUnitValue = 0u;
    if (!a || !b)
        return false;

    const auto* imgA = a->getCreationParameters().image.get();
    const auto* imgB = b->getCreationParameters().image.get();
    if (!imgA || !imgB)
        return false;

    const auto paramsA = imgA->getCreationParameters();
    const auto paramsB = imgB->getCreationParameters();
    if (paramsA.format != paramsB.format)
        return false;
    if (paramsA.extent != paramsB.extent)
        return false;

    const auto* bufA = imgA->getBuffer();
    const auto* bufB = imgB->getBuffer();
    if (!bufA || !bufB)
        return false;

    const size_t sizeA = bufA->getSize();
    if (sizeA != bufB->getSize())
        return false;

    const auto* dataA = static_cast<const uint8_t*>(bufA->getPointer());
    const auto* dataB = static_cast<const uint8_t*>(bufB->getPointer());
    if (!dataA || !dataB)
        return false;

    const uint32_t codeUnitBytes = SCodeUnitDiff::resolveCodeUnitBytes(paramsA.format);
    const size_t comparableSize = sizeA - (sizeA % codeUnitBytes);
    for (size_t i = 0u; i < comparableSize; i += codeUnitBytes)
    {
        uint32_t absDiff = 0u;
        if (codeUnitBytes == 1u)
        {
            const uint8_t va = dataA[i];
            const uint8_t vb = dataB[i];
            absDiff = SCodeUnitDiff::absoluteDiff<uint8_t>(va, vb);
        }
        else if (codeUnitBytes == 2u)
        {
            uint16_t va = 0u;
            uint16_t vb = 0u;
            std::memcpy(&va, dataA + i, sizeof(va));
            std::memcpy(&vb, dataB + i, sizeof(vb));
            absDiff = SCodeUnitDiff::absoluteDiff<uint16_t>(va, vb);
        }
        else
        {
            uint32_t va = 0u;
            uint32_t vb = 0u;
            std::memcpy(&va, dataA + i, sizeof(va));
            std::memcpy(&vb, dataB + i, sizeof(vb));
            absDiff = SCodeUnitDiff::absoluteDiff<uint32_t>(va, vb);
        }
        if (!absDiff)
            continue;
        ++diffCodeUnitCount;
        if (absDiff > maxDiffCodeUnitValue)
            maxDiffCodeUnitValue = absDiff;
    }

    for (size_t i = comparableSize; i < sizeA; ++i)
    {
        const uint32_t absDiff = SCodeUnitDiff::absoluteDiff<uint8_t>(dataA[i], dataB[i]);
        if (!absDiff)
            continue;
        ++diffCodeUnitCount;
        if (absDiff > maxDiffCodeUnitValue)
            maxDiffCodeUnitValue = absDiff;
    }

    return true;
}

} // namespace nbl::examples::image

#endif
