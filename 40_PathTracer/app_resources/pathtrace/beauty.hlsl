#include "common.hlsl"

#include "nbl/examples/common/KeyedQuantizedSequence.hlsl"


using namespace nbl;
using namespace nbl::hlsl;
using namespace nbl::this_example;
using namespace nbl::hlsl::path_tracing;

[[vk::push_constant]] SBeautyPushConstants pc;


struct[raypayload] BeautyPayload
{
    float32_t3 color : read(caller) : write(closesthit,miss);
    float16_t3 throughput : read(caller) : write(closesthit,miss);
    float16_t  otherTechniqueWeight : read(caller) : write(closesthit,miss);
    float16_t3 albedo : read(caller) : write(closesthit,miss);
    float16_t3 worldNormal : read(caller) : write(closesthit,miss);
};

[shader("raygeneration")]
void raygen()
{
    const uint32_t3 launchID = spirv::LaunchIdKHR;
    const uint32_t3 launchSize = spirv::LaunchSizeKHR;

    gAlbedo[launchID] = float32_t4(float32_t3(launchID)/float32_t3(launchSize),1.f);
}

[shader("closesthit")]
void closesthit(inout BeautyPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
}

[shader("miss")]
void miss(inout BeautyPayload payload)
{
}