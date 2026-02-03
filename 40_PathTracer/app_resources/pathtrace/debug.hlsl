#include "renderer/shaders/pathtrace/common.hlsl"
using namespace nbl::hlsl;
using namespace nbl::this_example;

[[vk::push_constant]] SDebugPushConstants pc;


struct[raypayload] DebugPayload
{
    uint32_t instanceID : read(caller):write(closesthit);
    uint32_t primitiveID : read(caller):write(closesthit);
};

[shader("raygeneration")]
void raygen()
{
    const uint32_t3 launchID = spirv::LaunchIdKHR;
    const uint32_t3 launchSize = spirv::LaunchSizeKHR;

    gAlbedo[launchID] = float32_t4(float32_t3(launchID)/float32_t3(launchSize),1.f);
}

[shader("miss")]
void miss(inout DebugPayload payload)
{
}