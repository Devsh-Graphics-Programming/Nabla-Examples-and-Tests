// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _AUTOEXPOSURE_COMMON_INCLUDED_
#define _AUTOEXPOSURE_COMMON_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/luma_meter/common.hlsl"
#include "nbl/builtin/hlsl/workgroup2/arithmetic_config.hlsl"

namespace nbl
{
namespace hlsl
{

struct AutoexposurePushData
{
    luma_meter::MeteringWindow window;
    float32_t lumaMin;
    float32_t lumaMax;
    uint32_t2 viewportSize;
    float32_t2 exposureAdaptationFactors;
    uint64_t pLumaMeterBuf;
    uint64_t pLastFrameEVBuf;
    uint64_t pCurrFrameEVBuf;

    // mean only
    float32_t rcpFirstPassWGCount;

    // histogram only
    uint32_t lowerBoundPercentile;
    uint32_t upperBoundPercentile;
};

#ifdef __HLSL_VERSION

#ifndef WORKGROUP_SIZE
#error "Define WORKGROUP_SIZE!"
#endif

#ifndef SUBGROUP_SIZE
#error "Define SUBGROUP_SIZE!"
#endif

#ifndef WG_CONFIG_T
#error "Define WG_CONFIG_T!"
#endif

using wg_config_t = WG_CONFIG_T;

struct device_capabilities
{
#ifdef NATIVE_SUBGROUP_ARITHMETIC
    NBL_CONSTEXPR_STATIC_INLINE bool shaderSubgroupArithmetic = true;
#else
    NBL_CONSTEXPR_STATIC_INLINE bool shaderSubgroupArithmetic = false;
#endif
};

#endif

}
}

#endif