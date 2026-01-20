#include "renderer/shaders/pathtrace/common.hlsl"
using namespace nbl::hlsl;
using namespace nbl::this_example;

[[vk::push_constant]] SPrevisPushConstants pc;


struct[raypayload] PrevisPayload
{
    uint16_t materialID : read(caller):write(closesthit);
};

[shader("raygeneration")]
void raygen()
{
    const uint32_t3 launchID = spirv::LaunchIdKHR;
    const uint32_t3 launchSize = spirv::LaunchSizeKHR;

    gAlbedo[launchID] = float32_t4(float32_t3(launchID)/float32_t3(launchSize),1.f);
}

[shader("miss")]
void miss(inout PrevisPayload payload)
{
}
