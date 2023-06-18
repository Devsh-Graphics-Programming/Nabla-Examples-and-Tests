#include "shaderCommon.hlsl"

#include "nbl/builtin/hlsl/workgroup/arithmetic.hlsl"
#include "nbl/builtin/hlsl/shared_memory_accessor.hlsl"

#define inclusive_scan_t(Binop) nbl::hlsl::workgroup::inclusive_scan<uint, nbl::hlsl::binops::Binop<uint>, nbl::hlsl::SharedMemoryAdaptor<nbl::hlsl::MemProxy> >

[numthreads(_NBL_HLSL_WORKGROUP_SIZE_, 1, 1)]
void main(uint gl_GlobalInvocationID: SV_DispatchThreadID)
{
	const uint sourceVal = inputValue[gl_GlobalInvocationID.x];
	inclusive_scan_t(bitwise_and) iscan_and;
	outand[0].output[gl_GlobalInvocationID.x] = iscan_and(sourceVal);
	
	inclusive_scan_t(bitwise_xor) iscan_xor;
	outxor[0].output[gl_GlobalInvocationID.x] = iscan_xor(sourceVal);
	
	inclusive_scan_t(bitwise_or) iscan_or;
	outor[0].output[gl_GlobalInvocationID.x] = iscan_or(sourceVal);
	
	inclusive_scan_t(add) iscan_add;
	outadd[0].output[gl_GlobalInvocationID.x] = iscan_add(sourceVal);
	
	inclusive_scan_t(mul) iscan_mul;
	outmul[0].output[gl_GlobalInvocationID.x] = iscan_mul(sourceVal);
	
	inclusive_scan_t(min) iscan_min;
	outmin[0].output[gl_GlobalInvocationID.x] = iscan_min(sourceVal);
	
	inclusive_scan_t(max) iscan_max;
	outmax[0].output[gl_GlobalInvocationID.x] = iscan_max(sourceVal);
}
