#pragma shader_stage(compute)

#include "common/uniform_sphere.hlsl"
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>

[[vk::binding(0, 0)]] RWStructuredBuffer<UniformSphereInputValues> inputTestValues;
[[vk::binding(1, 0)]] RWStructuredBuffer<UniformSphereTestResults> outputTestValues;

[numthreads(64, 1, 1)]
[shader("compute")]
void main()
{
	const uint32_t invID = nbl::hlsl::glsl::gl_GlobalInvocationID().x;
#ifdef BENCH_ITERS
	// Derive baseU from invID so u is a runtime value — prevents loop DCE after unrolling.
	const float32_t2 baseU = frac(float32_t(invID) * float32_t2(0.6180339887f, 0.7548776662f));
	uint32_t3 acc = (uint32_t3)0;
	[loop]
	for (uint32_t i = 0u; i < uint32_t(BENCH_ITERS); i++)
	{
		float32_t2 u = frac(baseU + float32_t(i) * float32_t2(0.6180339887f, 0.7548776662f));
		acc ^= asuint(sampling::UniformSphere<float32_t>::generate(u));
	}
	UniformSphereTestResults result = (UniformSphereTestResults)0;
	result.generated = asfloat(acc);
	outputTestValues[invID] = result;
#else
	UniformSphereTestExecutor executor;
	executor(inputTestValues[invID], outputTestValues[invID]);
#endif
}
