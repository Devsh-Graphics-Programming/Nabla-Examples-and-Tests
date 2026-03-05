#pragma shader_stage(compute)

#include "common/projected_hemisphere.hlsl"
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>

[[vk::binding(0, 0)]] RWStructuredBuffer<ProjectedHemisphereInputValues> inputTestValues;
[[vk::binding(1, 0)]] RWStructuredBuffer<ProjectedHemisphereTestResults> outputTestValues;

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
		acc ^= asuint(sampling::ProjectedHemisphere<float32_t>::generate(u));
	}
	ProjectedHemisphereTestResults result = (ProjectedHemisphereTestResults)0;
	result.generated = asfloat(acc);
	outputTestValues[invID] = result;
#else
	ProjectedHemisphereTestExecutor executor;
	executor(inputTestValues[invID], outputTestValues[invID]);
#endif
}
