#include "common.hlsl"
#include "nbl/builtin/hlsl/workgroup/fft.hlsl"

[[vk::push_constant]] PushConstantData pushConstants;

groupshared output_t sharedmem[4 * WorkgroupSize];

struct Accessor {
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

[numthreads(WorkgroupSize,1,1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	if (ID.x>=pushConstants.dataElementCount)
		return;

	const input_t selfLo = vk::RawBufferLoad<input_t>(pushConstants.inputAddress + sizeof(input_t) * ID.x);
	const input_t selfHi = vk::RawBufferLoad<input_t>(pushConstants.inputAddress + sizeof(input_t) * (ID.x + WorkgroupSize));

	nbl::hlsl::complex_t<output_t> lo = {selfLo.x, selfLo.y};
	nbl::hlsl::complex_t<output_t> hi = {selfHi.x, selfHi.y};
	// Since this FFT is convolution-only, the only real way to check it works (without tracking down order) is checking
	// that IFFT(FFT(x)) = x and vice-versa
	Accessor foo;	
	nbl::hlsl::workgroup::FFT<Accessor, output_t, WorkgroupSize, SubgroupSize, true>(foo, lo, hi);
	nbl::hlsl::workgroup::FFT<Accessor, output_t, WorkgroupSize, SubgroupSize, false>(foo, lo, hi);
	//nbl::hlsl::subgroup::FFT<output_t, SubgroupSize, false>(lo, hi);
	//nbl::hlsl::subgroup::FFT<output_t, SubgroupSize, true>(lo, hi);

	vk::RawBufferStore<output_t>(pushConstants.outputAddress+sizeof(output_t) * ID.x * 2, lo.real());
	vk::RawBufferStore<output_t>(pushConstants.outputAddress+sizeof(output_t) * (ID.x * 2 + 1), lo.imag());
	vk::RawBufferStore<output_t>(pushConstants.outputAddress+sizeof(output_t) * ((ID.x + WorkgroupSize) * 2), hi.real());
	vk::RawBufferStore<output_t>(pushConstants.outputAddress+sizeof(output_t) * ((ID.x + WorkgroupSize) * 2 + 1), hi.imag());
}