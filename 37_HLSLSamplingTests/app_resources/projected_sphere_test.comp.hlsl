#pragma shader_stage(compute)

#include "common/projected_sphere.hlsl"
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>

[[vk::binding(0, 0)]] RWStructuredBuffer<ProjectedSphereInputValues> inputTestValues;
[[vk::binding(1, 0)]] RWStructuredBuffer<ProjectedSphereTestResults> outputTestValues;

[numthreads(64, 1, 1)]
[shader("compute")]
void main()
{
	const uint32_t invID = nbl::hlsl::glsl::gl_GlobalInvocationID().x;
#ifdef BENCH_ITERS
	// Derive baseU from invID so u is a runtime value — prevents loop DCE after unrolling.
	const float32_t3 baseU = frac(float32_t(invID) * float32_t3(0.6180339887f, 0.7548776662f, 0.5698402910f));
	uint32_t3 acc = (uint32_t3)0;
	[loop]
	for (uint32_t i = 0u; i < uint32_t(BENCH_ITERS); i++)
	{
		float32_t3 u = frac(baseU + float32_t(i) * float32_t3(0.6180339887f, 0.7548776662f, 0.5698402910f));
		acc ^= asuint(sampling::ProjectedSphere<float32_t>::generate(u));
	}
	ProjectedSphereTestResults result = (ProjectedSphereTestResults)0;
	result.generated = asfloat(acc);
	outputTestValues[invID] = result;
#else
	ProjectedSphereTestExecutor executor;
	executor(inputTestValues[invID], outputTestValues[invID]);
#endif
}
