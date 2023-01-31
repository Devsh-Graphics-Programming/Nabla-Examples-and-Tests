

enum class ObjectType : uint32_t
{
    LINE = 0u,
    ELLIPSE = 1u,
};

struct DrawObject
{
    ObjectType type;
    uint32_t styleIdx;
    uint64_t address;
};

struct LinePoints
{
    // prev, start, end, next
    double2 p[4u];
};

struct EllipseInfo
{
    double2 majorAxis;
    double2 center;
    uint2 rangeAnglesPacked; // [0, 2Pi)
    uint32_t eccentricityPacked; // (0, 1]
    uint32_t _pad; // TODO we may need to add prev/next tangent if curve joins are not Bi-Arc
};

struct Globals
{
    double4x4 viewProjection;
    // Next two vars will be part of styles that the objects will reference
    float4 color;
    float lineWidth;
    float antiAliasingFactor;
    uint2 resolution;
};

#ifndef __cplusplus

// TODO: Remove these two when we include our builtin shaders
#define nbl_hlsl_PI 3.14159265359
#define UINT32_MAX      0xffffffffu

struct PSInput
{
	float4 position : SV_Position;
    [[vk::location(0)]] float4 color : COLOR; 
    [[vk::location(1)]] nointerpolation float4 start_end : COLOR1; 
    [[vk::location(2)]] nointerpolation uint3 lineWidth_eccentricity_objType : COLOR2; 
};

[[vk::binding(0,0)]] ConstantBuffer<Globals> globals : register(b0);
[[vk::binding(1,0)]] StructuredBuffer<DrawObject> drawObjects : register(t0);
#endif