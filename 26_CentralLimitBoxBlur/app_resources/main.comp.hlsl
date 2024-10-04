#pragma shader_stage(compute)

static const uint32_t MAX_ITEMS_PER_THREAD = WORKGROUP_SIZE;

#include "nbl/builtin/hlsl/central_limit_blur/common.hlsl"
#include "descriptors.hlsl"

#include "nbl/builtin/hlsl/central_limit_blur/box_blur.hlsl"

// https://github.com/microsoft/DirectXShaderCompiler/issues/6144
nbl::hlsl::uint32_t3 nbl::hlsl::glsl::gl_WorkGroupSize() { return uint32_t3( WORKGROUP_SIZE, 1, 1 ); }

static const uint32_t arithmeticSz = nbl::hlsl::workgroup::scratch_size_arithmetic<WORKGROUP_SIZE>::value;
static const uint32_t smemSize = WORKGROUP_SIZE + arithmeticSz;

groupshared uint32_t scratch[ smemSize ];

template<typename T>
struct ScratchProxy
{
	void get(const uint32_t ix, NBL_REF_ARG(float32_t) value) {
		value = asfloat(scratch[ix]);
	}

	void set(const uint32_t ix, const float32_t value) {
		scratch[ix] = asuint(value);
	}

	void workgroupExecutionAndMemoryBarrier() {
		nbl::hlsl::glsl::barrier();
	}
};

template<typename T>
struct SpillProxy
{
	void get(const uint32_t ix, NBL_REF_ARG(float32_t) value) {
		value = spill[ix];
	}

	void set(const uint32_t ix, const float32_t value) {
		spill[ix] = value;
	}

	void workgroupExecutionAndMemoryBarrier() {
		nbl::hlsl::glsl::barrier();
	}
	
	float32_t spill[MAX_ITEMS_PER_THREAD * 4]; // TODO: 4 is channel count. check if ness
};

[[vk::push_constant]]
nbl::hlsl::central_limit_blur::BoxBlurParams boxBlurParams;

[numthreads( WORKGROUP_SIZE, 1, 1 )]
void main(uint32_t GroupIndex: SV_GroupIndex)
{
	uint32_t2 texSize;
	output.GetDimensions( texSize.x, texSize.y );
	uint16_t axisIdx = uint16_t( boxBlurParams.direction );
	uint16_t items_per_thread = texSize[axisIdx] / (WORKGROUP_SIZE - 1);
	uint32_t wrapMode = boxBlurParams.wrapMode;
	nbl::hlsl::float32_t4 borderColor = boxBlurParams.getBorderColor();

	ScratchProxy<float32_t> scratchProxy;
	SpillProxy<float32_t> spillProxy;

	TextureProxy<MAX_ITEMS_PER_THREAD> textureProxy;
	for (uint16_t i = 0u; i < items_per_thread; ++i) {
		textureProxy.preload( axisIdx, i );
	}
	nbl::hlsl::glsl::barrier();

	for (uint16_t chIdx = 0u; chIdx < boxBlurParams.channelCount; ++chIdx) {
		nbl::hlsl::central_limit_blur::BoxBlur<__decltype(textureProxy), __decltype(scratchProxy),
											   __decltype(spillProxy), WORKGROUP_SIZE, arithmeticSz>
											   (GroupIndex, 1, items_per_thread, chIdx, boxBlurParams.radius, wrapMode, borderColor, textureProxy, scratchProxy, spillProxy);
	}

	nbl::hlsl::glsl::barrier();
	for (uint16_t i = 0; i < items_per_thread; ++i) {
		textureProxy.poststore( axisIdx, i );
	}
	
}