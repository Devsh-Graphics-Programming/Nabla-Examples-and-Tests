// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXAMPLES_COMMON_IMAGE_COMPARISON_INCLUDED_
#define _NBL_EXAMPLES_COMMON_IMAGE_COMPARISON_INCLUDED_

#include "nbl/asset/ICPUImage.h"
#include "nbl/asset/ICPUImageView.h"

namespace nbl::examples::image
{

inline bool compareCpuImageViewsByCodeUnit(const asset::ICPUImageView* a, const asset::ICPUImageView* b, uint64_t& diffCount, uint16_t& maxDiff)
{
    diffCount = 0u;
    maxDiff = 0u;
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

    for (size_t i = 0u; i < sizeA; ++i)
    {
        const int diff = static_cast<int>(dataA[i]) - static_cast<int>(dataB[i]);
        const uint16_t absDiff = static_cast<uint16_t>(diff < 0 ? -diff : diff);
        if (!absDiff)
            continue;
        ++diffCount;
        if (absDiff > maxDiff)
            maxDiff = absDiff;
    }

    return true;
}

} // namespace nbl::examples::image

#endif

