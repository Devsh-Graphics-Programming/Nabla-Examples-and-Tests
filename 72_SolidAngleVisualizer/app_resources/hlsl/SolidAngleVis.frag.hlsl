#pragma wave shader_stage(fragment)

#include "common.hlsl"

#include <nbl/builtin/hlsl/ext/FullScreenTriangle/SVertexAttributes.hlsl>

using namespace nbl::hlsl;
using namespace ext::FullScreenTriangle;

[[vk::push_constant]] struct PushConstants pc;

static const float CIRCLE_RADIUS = 0.45f;

// --- Geometry Utils ---

// Adjacency of edges to faces
static const int2 edgeToFaces[12] = { 
    {4,2}, {3,4}, {2,5}, {5,3}, 
    {2,0}, {0,3}, {1,2}, {3,1}, 
    {0,4}, {5,0}, {4,1}, {1,5} 
};

static const float3 localNormals[6] = {
    float3(0, 0, -1), // Face 0 (Z-)
    float3(0, 0, 1),  // Face 1 (Z+)
    float3(-1, 0, 0), // Face 2 (X-)
    float3(1, 0, 0),  // Face 3 (X+)
    float3(0, -1, 0), // Face 4 (Y-)
    float3(0, 1, 0)   // Face 5 (Y+)
};

static float3 corners[8];
static float3 faceCenters[6] = { float3(0,0,0), float3(0,0,0), float3(0,0,0), 
                            float3(0,0,0), float3(0,0,0), float3(0,0,0) };
static float2 projCorners[8];


// Converts UV into centered, aspect-corrected NDC circle space
float2 toCircleSpace(float2 uv)
{
    float aspect = pc.viewport.z / pc.viewport.w;
    float2 centered = uv - 0.5f;
    centered.x *= aspect;
    return centered;
}

// Distance to a 2D line segment
float sdSegment(float2 p, float2 a, float2 b)
{
    float2 pa = p - a;
    float2 ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0f, 1.0f);
    return length(pa - ba * h);
}

// TODO: Hemispherical Projection (Solid Angle / Orthographic/Lambertian Projection)
float2 project(float3 p) 
{
    return normalize(p).xy;
}

void computeCubeGeo()
{
    for (int i = 0; i < 8; i++)
    {
        float3 localPos = float3(i % 2, (i / 2) % 2, (i / 4) % 2) * 2.0f - 1.0f;
        float3 worldPos = mul(pc.modelMatrix, float4(localPos, 1.0f)).xyz;
        
        corners[i] = worldPos;
        
        faceCenters[i/4]      += worldPos / 4.0f; 
        faceCenters[2+i%2]    += worldPos / 4.0f; 
        faceCenters[4+(i/2)%2] += worldPos / 4.0f; 

        float3 viewPos = worldPos; 
        projCorners[i] = project(viewPos);
    }
}

int getVisibilityCount(int2 faces, float3 cameraPos)
{
    float3x3 rotMatrix = (float3x3)pc.modelMatrix;
    float3 n_world_f1 = mul(rotMatrix, localNormals[faces.x]);
    float3 n_world_f2 = mul(rotMatrix, localNormals[faces.y]);
    
    float3 viewVec_f1 = faceCenters[faces.x] - cameraPos; 
    float3 viewVec_f2 = faceCenters[faces.y] - cameraPos;

    // Face is visible if its outward normal points towards the origin (camera).
    bool visible1 = dot(n_world_f1, viewVec_f1) < 0.0f;
    bool visible2 = dot(n_world_f2, viewVec_f2) < 0.0f;

    // Determine Line Style:
    bool isSilhouette = visible1 != visible2; // One face visible, the other hidden
    bool isInner = visible1 && visible2;      // Both faces visible
    
    int visibilityCount = 0;
    if (isSilhouette) 
    {
        visibilityCount = 1;
    }
    else if (isInner)
    {
        visibilityCount = 2;
    }

    return visibilityCount;
}

void drawLine(float2 p, int a, int b, int visibilityCount, inout float4 color, float aaWidth)
{
    if (visibilityCount > 0)
    {
        float3 A = corners[a];
        float3 B = corners[b];

        float avgDepth = (length(A) + length(B)) * 0.5f;
        float referenceDepth = 3.0f;
        float depthScale = referenceDepth / avgDepth;

        float baseWidth = (visibilityCount == 1) ? 0.005f : 0.002f;
        float intensity = (visibilityCount == 1) ? 1.0f : 0.5f;
        float4 edgeColor = (visibilityCount == 1) ? float4(0.0f, 0.5f, 1.0f, 1.0f) : float4(1.0f, 0.0f, 0.0f, 1.0f); // Blue vs Red
        
        float width = min(baseWidth * depthScale, 0.03f); 
        
        float dist = sdSegment(p, projCorners[a], projCorners[b]);
        
        float alpha = 1.0f - smoothstep(width - aaWidth, width + aaWidth, dist);
        
        color += edgeColor * alpha * intensity;
    }
}

void drawRing(float2 p, inout float4 color, float aaWidth)
{
    float positionLength = length(p);

    // Mask to cut off drawing outside the circle
    // float circleMask = 1.0f - smoothstep(CIRCLE_RADIUS, CIRCLE_RADIUS + aaWidth, positionLength);
    // color *= circleMask;
    
    // Add a white background circle ring
    float ringWidth = 0.005f;
    float ringDistance = abs(positionLength - CIRCLE_RADIUS);
    float ringAlpha = 1.0f - smoothstep(ringWidth - aaWidth, ringWidth + aaWidth, ringDistance);
    
    // Ring color is now white
    color = max(color, float4(1.0, 1.0, 1.0, 1.0) * ringAlpha); 
}

[[vk::location(0)]] float32_t4 main(SVertexAttributes vx) : SV_Target0
{
    float3 cameraPos = float3(0, 0, 0); // Camera at origin
    float2 p = toCircleSpace(vx.uv);
    float4 color = float4(0, 0, 0, 0);

    computeCubeGeo();
    
    float aaWidth = max(fwidth(p.x), fwidth(p.y)); 

    for (int j = 0; j < 12; j++)
    {
        int a = j % 4 * (j < 4 ? 1 : 2) - (j / 4 == 1 ? j % 2 : 0);
        int b = a + (4 >> (j / 4));

        int2 faces = edgeToFaces[j];
        int visibilityCount = getVisibilityCount(faces, cameraPos);
        drawLine(p, a, b, visibilityCount, color, aaWidth);
    }

    drawRing(p, color, aaWidth);

    return color;
}