#include "common.hlsl"

[shader("miss")]
void main(inout ShadowPayload payload)
{
	payload.attenuation = payload.attenuation * -1;
}
