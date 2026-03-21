#define FRAGMENT_SHADER_INPUT
#pragma shader_stage(fragment)

#include "dtm.hlsl"
#include "common.hlsl"

static const float32_t3 SunlightDirection = float32_t3(0.7071f, -0.7071f, 0.0f);

[shader("pixel")]
float4 fragMain(PSInput input) : SV_Target
{
	static const float AmbientLightIntensity = 0.1f;
	const float diffuseLightIntensity = max(dot(-SunlightDirection, input.getNormal()), 0.0f);

	const MainObject mainObj = loadMainObject(pc.triangleMeshMainObjectIndex);
	const DTMSettings dtmSettings = loadDTMSettings(mainObj.dtmSettingsIdx);
	
	float32_t3 triangleVertices[3];
    triangleVertices[0] = input.getScreenSpaceVertexAttribs(0);
    triangleVertices[1] = input.getScreenSpaceVertexAttribs(1);
    triangleVertices[2] = input.getScreenSpaceVertexAttribs(2);

	const float height = input.getHeight();
	const float heightDeriv = fwidth(height);

	const float32_t4 HeightColor = dtm::calculateDTMHeightColor(dtmSettings.heightShadingSettings, heightDeriv, triangleVertices, input.position.xy, height);

	const float32_t4 fragColor = (AmbientLightIntensity + diffuseLightIntensity) * HeightColor;

	return fragColor;
}