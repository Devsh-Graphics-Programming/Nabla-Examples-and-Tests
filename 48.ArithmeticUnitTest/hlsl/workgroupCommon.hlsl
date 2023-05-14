// REVIEW: IS the below comment relevant?
//
// ORDER OF INCLUDES MATTERS !!!!!
// first the feature that requires the most shared memory should be included
// anyway when one is using more than 2 features that rely on shared memory,
// they should declare the shared memory of appropriate size by themselves.
// But in this unit test we don't because we need to test if the default
// sizing macros actually work for all workgroup sizes.
#include "shaderCommon.hlsl"
#include <nbl/builtin/hlsl/workgroup/arithmetic.hlsl>
#include <nbl/builtin/hlsl/workgroup/ballot.hlsl>
#include <nbl/builtin/hlsl/shared_memory_accessor.hlsl>