// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _AUTOEXPOSURE_COMMON_INCLUDED_
#define _AUTOEXPOSURE_COMMON_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/luma_meter/common.hlsl"

namespace nbl
{
namespace hlsl
{

struct AutoexposurePushData
{
    luma_meter::MeteringWindow window;
    float32_t2 lumaMinMax;
    float32_t sampleCount;
    uint32_t2 viewportSize;
    uint64_t lumaMeterBDA;
};

#ifdef __HLSL_VERSION

#ifndef WorkgroupSize
#error "Define WorkgroupSize!"
#endif

#ifndef DeviceSubgroupSize
#error "Define DeviceSubgroupSize!"
#endif

#endif

}
}

#endif