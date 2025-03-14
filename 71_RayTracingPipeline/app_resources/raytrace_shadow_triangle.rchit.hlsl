#include "common.hlsl"

[shader("closesthit")]
void main(inout OcclusionPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    payload.attenuation = 0;
}
