
#pragma shader_stage(vertex)

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

PSInput main(uint vertexID : SV_VertexID)
{
    const uint vertexIdx = vertexID & 0x3u;
    const uint objectID = vertexID >> 2;
    
    DrawObject drawObj = drawObjects[objectID];
    ObjectType objType = drawObj.type;
    
    PSInput outV;
    outV.color = globals.color;
    outV.lineWidth_eccentricity_objType.x = globals.lineWidth;
    outV.lineWidth_eccentricity_objType.z = (uint)objType;

    if (objType == ObjectType::ELLIPSE)
    {
    }
    else
    {
        double3x3 transformation = (double3x3)globals.viewProjection;
        LinePoints points = vk::RawBufferLoad<LinePoints>(drawObj.address, 8u);
        float2 transformedPoints[4u];
        for(uint i = 0u; i < 4u; ++i)
        {
            double2 ndc = mul(transformation, double3(points.p[i], 1)).xy; // Transform to NDC
            transformedPoints[i] = (float2)((ndc + 1.0) * 0.5 * globals.resolution); // Transform to Screen Space
        }

        const float2 lineVector = normalize(transformedPoints[2u] - transformedPoints[1u]);
        const float2 normalToLine = float2(-lineVector.y, lineVector.x);

        if (vertexIdx == 0u || vertexIdx == 1u)
        {
            const float2 vectorPrev = normalize(transformedPoints[1u] - transformedPoints[0u]);
            const float2 normalPrevLine = float2(-vectorPrev.y, vectorPrev.x);
            const float2 miter = normalize(normalPrevLine + normalToLine);

            outV.position.xy = transformedPoints[1u] + (miter * ((float)vertexIdx - 0.5f) * globals.lineWidth) / dot(normalToLine, normalPrevLine);
        }
        else // if (vertexIdx == 2u || vertexIdx == 3u)
        {
            const float2 vectorNext = normalize(transformedPoints[3u] - transformedPoints[2u]);
            const float2 normalNextLine = float2(-vectorNext.y, vectorNext.x);
            const float2 miter = normalize(normalNextLine + normalToLine);

            outV.position.xy = transformedPoints[2u] + (miter * ((float)vertexIdx - 2.5f) * globals.lineWidth) / dot(normalToLine, normalNextLine);
        }

        outV.start_end.xy = transformedPoints[1u];
        outV.start_end.wz = transformedPoints[2u];
        outV.position.xy = outV.position.xy / globals.resolution * 2.0 - 1.0; // back to NDC for SV_Position
        outV.position.w = 1u;
    }
	return outV;
}
