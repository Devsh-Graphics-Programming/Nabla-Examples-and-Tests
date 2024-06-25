#include "common.hlsl"
#include "nbl/builtin/hlsl/workgroup/fft.hlsl"
//#include "nbl/builtin/hlsl/subgroup/fft.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"

[[vk::push_constant]] PushConstantData pushConstants;

groupshared output_t sharedmem[2 * WorkgroupSize];

struct SharedMemoryAccessor {
	void set(uint32_t idx, output_t x) {
		sharedmem[idx] = x;
	}
	
	output_t get(uint32_t idx) {
		return sharedmem[idx];
	}

	void workgroupExecutionAndMemoryBarrier() {
        AllMemoryBarrierWithGroupSync();
    }
};

struct Accessor {
	void set(uint32_t idx, output_t x) {
		vk::RawBufferStore<output_t>(pushConstants.outputAddress + sizeof(output_t) * idx, x);
	}
	
	input_t get(uint32_t idx) {
		return vk::RawBufferLoad<input_t>(pushConstants.inputAddress + sizeof(input_t) * idx);
	}

	void workgroupExecutionAndMemoryBarrier() {
        AllMemoryBarrierWithGroupSync();
    }
};

[numthreads(WorkgroupSize,1,1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	if (nbl::hlsl::workgroup::SubgroupContiguousIndex() >= pushConstants.dataElementCount)
		return;

	// Subgroup (works fine)
	/*
	if (nbl::hlsl::glsl::gl_SubgroupInvocationID() >= pushConstants.dataElementCount)
		return;
	*/

	Accessor accessor;
	SharedMemoryAccessor sharedmemAccessor;
	nbl::hlsl::workgroup::FFT<2, false, input_t>::template __call<Accessor, SharedMemoryAccessor>(accessor, sharedmemAccessor);	
	//nbl::hlsl::workgroup::FFT<2, true, input_t>::template __call<Accessor, SharedMemoryAccessor>(accessor, sharedmemAccessor);
	
	// Subgroup (works fine)
	/*

	nbl::hlsl::complex_t<input_t> lo, hi;
	lo.real(accessor.get(nbl::hlsl::glsl::gl_SubgroupInvocationID()));
	hi.real(accessor.get(nbl::hlsl::glsl::gl_SubgroupInvocationID() + _NBL_HLSL_WORKGROUP_SIZE_));
	lo.imag(accessor.get(nbl::hlsl::glsl::gl_SubgroupInvocationID() + 2 * _NBL_HLSL_WORKGROUP_SIZE_));
	hi.imag(accessor.get(nbl::hlsl::glsl::gl_SubgroupInvocationID() + 3 * _NBL_HLSL_WORKGROUP_SIZE_));
	
	nbl::hlsl::subgroup::FFT<false, input_t>::__call(lo, hi);
	//nbl::hlsl::subgroup::FFT<true, input_t>::__call(lo, hi);

	accessor.set(nbl::hlsl::glsl::gl_SubgroupInvocationID(), lo.real());
	accessor.set(nbl::hlsl::glsl::gl_SubgroupInvocationID() + _NBL_HLSL_WORKGROUP_SIZE_, hi.real());
	accessor.set(nbl::hlsl::glsl::gl_SubgroupInvocationID() + 2 * _NBL_HLSL_WORKGROUP_SIZE_, lo.imag());
	accessor.set(nbl::hlsl::glsl::gl_SubgroupInvocationID() + 3 * _NBL_HLSL_WORKGROUP_SIZE_, hi.imag());		

	*/
}