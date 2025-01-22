#include "common.hlsl"

[shader("miss")]
void main(inout ShadowPayload p)
{
	p.isShadowed = false;
}
