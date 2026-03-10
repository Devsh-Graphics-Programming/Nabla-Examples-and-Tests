#include "common.hlsl"

static const float32_t3 SunlightDirection = float32_t3(0.7071f, -0.7071f, 0.0f);
static const float32_t3 TerrainColor = float32_t3(1.0f, 1.0f, 1.0f);

[shader("pixel")]
float4 fragMain(PSInput input) : SV_Target
{
	static const float AmbientLightIntensity = 0.1f;
	const float diffuseLightIntensity = max(dot(-SunlightDirection, input.normal), 0.0f);

	const float32_t3 fragColor = (AmbientLightIntensity + diffuseLightIntensity) * TerrainColor;

	return float32_t4(fragColor, 1.0f);
}
