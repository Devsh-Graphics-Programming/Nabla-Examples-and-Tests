#ifndef _COOPERATIVE_BINARY_SEARCH_H_INCLUDED_
#define _COOPERATIVE_BINARY_SEARCH_H_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/basic.h>
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>

// TODO: NBL_CONSTEXPR_NSPC_VAR
static const uint32_t WorkgroupSize = 256;

struct PushConstants
{
	uint32_t EntityCount;
};

#endif // _COOPERATIVE_BINARY_SEARCH_H_INCLUDED_
