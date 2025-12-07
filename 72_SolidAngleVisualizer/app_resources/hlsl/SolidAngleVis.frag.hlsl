#pragma wave shader_stage(fragment)

#include "common.hlsl"

#include <nbl/builtin/hlsl/ext/FullScreenTriangle/SVertexAttributes.hlsl>

using namespace nbl::hlsl;
using namespace ext::FullScreenTriangle;

[[vk::push_constant]] struct PushConstants pc;

static const float CIRCLE_RADIUS = 0.75f;

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



// Converts UV into centered, aspect-corrected NDC circle space
float2 toCircleSpace(float2 uv)
{
    // Map [0,1] UV to [-1,1]
    float2 p = uv * 2.0f - 1.0f;

    // Correct aspect ratio
    float aspect = pc.viewport.z / pc.viewport.w; // width / height
    p.x *= aspect;

    return p * CIRCLE_RADIUS;
}

void computeCubeGeo()
{
    for (int i = 0; i < 8; i++)
    {
        float3 localPos = float3(i % 2, (i / 2) % 2, (i / 4) % 2) * 2.0f - 1.0f;
        float3 worldPos = mul(pc.modelMatrix, float4(localPos, 1.0f)).xyz;
        
        corners[i] = worldPos.xyz;
        
        faceCenters[i/4]      += worldPos / 4.0f; 
        faceCenters[2+i%2]    += worldPos / 4.0f; 
        faceCenters[4+(i/2)%2] += worldPos / 4.0f; 
    }
}

float4 drawRing(float2 p, float aaWidth)
{
    float positionLength = length(p);
    
    // Add a white background circle ring
    float ringWidth = 0.01f;
    float ringDistance = abs(positionLength - CIRCLE_RADIUS);
    float ringAlpha = 1.0f - smoothstep(ringWidth - aaWidth, ringWidth + aaWidth, ringDistance);
    
    return ringAlpha.xxxx; 
}

// Check if a face on the hemisphere is visible from camera at origin
bool isFaceVisible(float3 faceCenter, float3 faceNormal)
{
    // Face is visible if normal points toward camera (at origin)
    float3 viewVec = -normalize(faceCenter); // Vector from face to camera
    return dot(faceNormal, viewVec) > 0.0f;
}

int getEdgeVisibility(int edgeIdx, float3 cameraPos)
{
    int2 faces = edgeToFaces[edgeIdx];
    
    // Transform normals to world space
    float3x3 rotMatrix = (float3x3)pc.modelMatrix;
    float3 n_world_f1 = mul(rotMatrix, localNormals[faces.x]);
    float3 n_world_f2 = mul(rotMatrix, localNormals[faces.y]);
    
    bool visible1 = isFaceVisible(faceCenters[faces.x], n_world_f1);
    bool visible2 = isFaceVisible(faceCenters[faces.y], n_world_f2);
    
    // Silhouette: exactly one face visible
    if (visible1 != visible2) return 1;
    
    // Inner edge: both faces visible
    if (visible1 && visible2) return 2;
    
    // Hidden edge: both faces hidden
    return 0;
}

// Draw great circle arc in fragment shader
float4 drawGreatCircleArc(float3 fragPos, int2 edgeVerts, int visibility, float aaWidth)
{
    if (visibility == 0) return float4(0,0,0,0); // Hidden edge
    
    float3 v0 = normalize(corners[edgeVerts.x]);
    float3 v1 = normalize(corners[edgeVerts.y]);
    float3 p = normalize(fragPos); // Current point on hemisphere
    
    // Skip fragment if not in front of hemisphere or edge if both endpoints are behind horizon
    if (p.z < 0.0f || (v0.z < 0.0f && v1.z < 0.0f)) 
        return float4(0,0,0,0);
    
    // Great circle plane normal
    float3 arcNormal = normalize(cross(v0, v1));
    
    // Distance to great circle
    float dist = abs(dot(p, arcNormal));
    
    // Check if point is within arc bounds
    float dotMid = dot(v0, v1);
    bool onArc = (dot(p, v0) >= dotMid) && (dot(p, v1) >= dotMid);
    
    if (!onArc) return float4(0,0,0,0);
    
    // Depth-based width scaling
    float avgDepth = (length(corners[edgeVerts.x]) + length(corners[edgeVerts.y])) * 0.5f;
    float depthScale = 3.0f / avgDepth;
    
    float baseWidth = (visibility == 1) ? 0.01f : 0.005f;
    float width = min(baseWidth * depthScale, 0.02f);
    
    float alpha = 1.0f - smoothstep(width - aaWidth, width + aaWidth, dist);
    
    float4 edgeColor = (visibility == 1) ? 
        float4(0.0f, 0.5f, 1.0f, 1.0f) :  // Silhouette: blue
        float4(1.0f, 0.0f, 0.0f, 1.0f);   // Inner: red
    
    float intensity = (visibility == 1) ? 1.0f : 0.5f;
    return edgeColor * alpha * intensity;
}

[[vk::location(0)]] float32_t4 main(SVertexAttributes vx) : SV_Target0
{
    float3 cameraPos = float3(0, 0, 0);
    float4 color = float4(0, 0, 0, 0);
    float2 p = toCircleSpace(vx.uv);
    
    // Convert 2D disk position to 3D hemisphere position
    // p is in range [-CIRCLE_RADIUS, CIRCLE_RADIUS]
    float2 normalized = p / CIRCLE_RADIUS; // Now in range [-1, 1]
    float r2 = dot(normalized, normalized);
    
    if (r2 > 1.0f)
        discard;
    
    // Convert UV to 3D position on hemisphere
    float3 spherePos = normalize(float3(normalized.x, normalized.y, sqrt(1 - r2)));
    
    computeCubeGeo(); // Your existing function
    
    float aaWidth = length(float2(ddx(p.x), ddy(p.y))); 
    
    // Draw edges as great circle arcs
    for (int j = 0; j < 12; j++) 
    {
        int a = j % 4 * (j < 4 ? 1 : 2) - (j / 4 == 1 ? j % 2 : 0);
        int b = a + (4 >> (j / 4));
        
        int visibility = getEdgeVisibility(j, cameraPos);
        color += drawGreatCircleArc(spherePos, int2(a, b), visibility, aaWidth);
    }
    
    color += drawRing(p, aaWidth);
    
    return color;
}