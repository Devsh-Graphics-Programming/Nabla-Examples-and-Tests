
#pragma shader_stage(fragment)

enum class ObjectType : uint32_t
{
    LINE = 0u,
    ELLIPSE = 1u,
};

struct DrawObject
{
    ObjectType type;
    uint styleIdx;
    uint64_t address;
};

struct LinePoints
{
    // prev, start, end, next
    double2 p[4u];
};

struct Ellipse
{
    double2 majorAxis;
    double2 center;
    uint2 rangeAnglesPacked; // [0, 2Pi)
    uint eccentricityPacked; // (0, 1]
    uint _pad; // TODO we may need to add prev/next tangent if curve joins are not Bi-Arc
};

struct Globals
{
    double4x4 viewProjection;
    // Next two vars will be part of styles that the objects will reference
    float4 color;
    uint lineWidth;
    uint pad;
    uint2 resolution;
};

struct PSInput
{
    float4 position : SV_Position;
    [[vk::location(0)]] float4 color : COLOR; 
    [[vk::location(1)]] nointerpolation float4 start_end : COLOR1; 
    [[vk::location(2)]] nointerpolation uint3 lineWidth_eccentricity_objType : COLOR2; 
};

[[vk::binding(0,0)]] ConstantBuffer<Globals> globals : register(b0);
[[vk::binding(1,0)]] StructuredBuffer<DrawObject> drawObjects : register(t0);

float4 main(PSInput input) : SV_TARGET
{
    ObjectType objType = (ObjectType)input.lineWidth_eccentricity_objType.z;
    if (objType == ObjectType::ELLIPSE)
    {
    }
    else if (objType == ObjectType::LINE)
    {
        const float2 start = input.start_end.xy;
        const float2 end = input.start_end.zw;
        const uint lineWidthHalf = ((float)input.lineWidth_eccentricity_objType.x) / 2.0f;
        float2 lineVec = end - start;
        float2 pointVec = input.position.xy - start;
        float pointVecLen = length(pointVec);
        float lineLen = length(end - start);
        float projectionLength = dot(normalize(pointVec),normalize(lineVec)) * pointVecLen;
        float t = projectionLength / lineLen;
        if (t < 0 && pointVecLen > lineWidthHalf)
            discard;
        else if (t > 1 && length(input.position.xy - end) > lineWidthHalf)
            discard;
        return input.color;
    }
    return input.color;
}
