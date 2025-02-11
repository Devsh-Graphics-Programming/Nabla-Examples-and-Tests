#include "common.hlsl"

[shader("miss")]
void main(inout ShadowPayload payload)
{
	payload.isShadowed = false;
}
