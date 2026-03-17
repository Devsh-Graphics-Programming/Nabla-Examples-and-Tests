#pragma shader_stage(fragment)

#include "common.hlsl"

static const float32_t3 SunlightDirection = float32_t3(0.7071f, -0.7071f, 0.0f);

[shader("pixel")]
float4 fragMain(PSInput input) : SV_Target
{
	static const float AmbientLightIntensity = 0.1f;
	const float diffuseLightIntensity = max(dot(-SunlightDirection, input.normal), 0.0f);

	MainObject mainObj = loadMainObject(pc.triangleMeshMainObjectIndex);
	DTMSettings dtmSettings = loadDTMSettings(mainObj.dtmSettingsIdx);
		
	const float32_t3 HeightColor = input.height < 50.0f ? float32_t3(0.0, 1.0, 0.0) : (input.height < 75.0f ? float32_t3(1.0, 1.0, 0.0) : float32_t3(1.0, 0.0, 0.0));
		
	const float32_t3 fragColor = (AmbientLightIntensity + diffuseLightIntensity) * HeightColor;

	return float32_t4(fragColor, 1.0f);
}
