#include "renderer/shaders/pathtrace/common.hlsl"
using namespace nbl::hlsl;
using namespace nbl::this_example;

[[vk::push_constant]] SBeautyPushConstants pc;

[shader("raygeneration")]
void pathtrace_beauty()
{
    const uint32_t3 launchID = spirv::LaunchIdKHR;
    const uint32_t3 launchSize = spirv::LaunchSizeKHR;

    gAlbedo[launchID] = float32_t4(float32_t3(launchID)/float32_t3(launchSize),1.f);
}
