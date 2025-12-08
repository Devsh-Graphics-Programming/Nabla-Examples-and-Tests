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

//float3(i % 2, (i / 2) % 2, (i / 4) % 2) * 2.0f - 1.0f
static const float3 constCorners[8] = {
    float3(-1, -1, -1), // 0
    float3( 1, -1, -1), // 1
    float3(-1,  1, -1), // 2
    float3( 1,  1, -1), // 3
    float3(-1, -1,  1), // 4
    float3( 1, -1,  1), // 5
    float3(-1,  1,  1), // 6
    float3( 1,  1,  1)  // 7
};

// All 12 edges of the cube (vertex index pairs)
static const int2 allEdges[12] = {
    {0, 1}, {2, 3}, {4, 5}, {6, 7}, // Edges along X axis
    {0, 2}, {1, 3}, {4, 6}, {5, 7}, // Edges along Y axis
    {0, 4}, {1, 5}, {2, 6}, {3, 7}  // Edges along Z axis
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


static const float3 colorLUT[27] = {
    // Row 1: Pure and bright colors
    float3(0, 0, 0),        // 0: Black
    float3(1, 1, 1),        // 1: White
    float3(0.5, 0.5, 0.5),  // 2: Gray
    
    // Row 2: Primary colors
    float3(1, 0, 0),        // 3: Red
    float3(0, 1, 0),        // 4: Green
    float3(0, 0, 1),        // 5: Blue
    
    // Row 3: Secondary colors
    float3(1, 1, 0),        // 6: Yellow
    float3(1, 0, 1),        // 7: Magenta
    float3(0, 1, 1),        // 8: Cyan
    
    // Row 4: Orange family
    float3(1, 0.5, 0),      // 9: Orange
    float3(1, 0.65, 0),     // 10: Light Orange
    float3(0.8, 0.4, 0),    // 11: Dark Orange
    
    // Row 5: Pink/Rose family
    float3(1, 0.4, 0.7),    // 12: Pink
    float3(1, 0.75, 0.8),   // 13: Light Pink
    float3(0.7, 0.1, 0.3),  // 14: Deep Rose
    
    // Row 6: Purple/Violet family
    float3(0.5, 0, 0.5),    // 15: Purple
    float3(0.6, 0.4, 0.8),  // 16: Light Purple
    float3(0.3, 0, 0.5),    // 17: Indigo
    
    // Row 7: Green variations
    float3(0, 0.5, 0),      // 18: Dark Green
    float3(0.5, 1, 0),      // 19: Lime
    float3(0, 0.5, 0.25),   // 20: Forest Green
    
    // Row 8: Blue variations
    float3(0, 0, 0.5),      // 21: Navy
    float3(0.3, 0.7, 1),    // 22: Sky Blue
    float3(0, 0.4, 0.6),    // 23: Teal
    
    // Row 9: Earth tones
    float3(0.6, 0.4, 0.2),  // 24: Brown
    float3(0.8, 0.7, 0.3),  // 25: Tan/Beige
    float3(0.4, 0.3, 0.1)   // 26: Dark Brown
};


    
// Vertices are ordered CCW relative to the camera view.
static const int silhouettes[27][7] = {
    {6, 1, 3, 2, 6, 4, 5}, // 0: Black
    {6, 2, 6, 4, 5, 7, 3}, // 1: White 
    {6, 0, 4, 5, 7, 3, 2}, // 2: Gray 
    {6, 1, 3, 7, 6, 4, 5,}, // 3: Red 
    {4, 4, 5, 7, 6, -1, -1}, // 4: Green 
    {6, 0, 4, 5, 7, 6, 2}, // 5: Blue 
    {6, 0, 1, 3, 7, 6, 4}, // 6: Yellow 
    {6, 0, 1, 5, 7, 6, 4}, // 7: Magenta              
    {6, 0, 1, 5, 7, 6, 2}, // 8: Cyan 
    {6, 1, 3, 2, 6, 7, 5}, // 9: Orange
    {4, 2, 6, 7, 3, -1, -1}, // 10: Light Orange
    {6, 0, 4, 6, 7, 3, 2}, // 11: Dark Orange
    {4, 1, 3, 7, 5, -1, -1}, // 12: Pink
    {6, 0, 4, 6, 7, 3, 2}, // 13: Light Pink
    {4, 0, 4, 6, 2, -1, -1}, // 14: Deep Rose
    {6, 0, 1, 3, 7, 5, 4}, // 15: Purple
    {4, 0, 1, 5, 4, -1, -1}, // 16: Light Purple
    {6, 0, 1, 5, 4, 6, 2}, // 17: Indigo
    {6, 0, 2, 6, 7, 5, 1}, // 18: Dark Green
    {6, 0, 2, 6, 7, 3, 1}, // 19: Lime
    {6, 0, 4, 6, 7, 3, 1}, // 20: Forest Green
    {6, 0, 2, 3, 7, 5, 1}, // 21: Navy
    {4, 0, 2, 3, 1, -1, -1}, // 22: Sky Blue
    {6, 0, 4, 6, 2, 3, 1}, // 23: Teal
    {6, 0, 2, 3, 7, 5, 4},  // 24: Brown
    {6, 0, 2, 3, 1, 5, 4}, // 25: Tan/Beige
    {6, 1, 5, 4, 6, 2, 3}  // 26: Dark Brown
};

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
        float3 localPos = constCorners[i]; //float3(i % 2, (i / 2) % 2, (i / 4) % 2) * 2.0f - 1.0f;
        float3 worldPos = mul(pc.modelMatrix, float4(localPos, 1.0f)).xyz;
        
        corners[i] = worldPos.xyz;
        
        faceCenters[i/4]      += worldPos / 4.0f; 
        faceCenters[2+i%2]    += worldPos / 4.0f; 
        faceCenters[4+(i/2)%2] += worldPos / 4.0f; 
    }
}

float4 drawCorners(float3 spherePos, float aaWidth)
{
    float4 color = float4(0,0,0,0);
    // Draw corner labels for debugging
    for (int i = 0; i < 8; i++)
    {
        float3 corner = normalize(corners[i]);
        float2 cornerPos = corner.xy;
        // Project corner onto 2D circle space
        
        // Distance from current fragment to corner
        float dist = length(spherePos.xy - cornerPos);
        
        // Draw a small colored dot at the corner
        float dotSize = 0.03f;
        float dotAlpha = 1.0f - smoothstep(dotSize - aaWidth, dotSize + aaWidth, dist);
        
        if (dotAlpha > 0.0f)
        {
            float brightness = float(i) / 7.0f;
            float3 dotColor = colorLUT[i];
            color += float4(dotColor * dotAlpha, dotAlpha);
        }
    }
    return color;
}

float4 drawRing(float2 p, float aaWidth)
{
    float positionLength = length(p);
    
    // Add a white background circle ring
    float ringWidth = 0.01f;
    float ringDistance = abs(positionLength - CIRCLE_RADIUS);
    float ringAlpha = 1.0f - smoothstep(ringWidth - aaWidth, ringWidth + aaWidth, ringDistance);
    
    return ringAlpha * float4(1, 1, 1, 1); 
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

// Draw great circle arc in fragment shader with horizon clipping
float4 drawGreatCircleArc(float3 fragPos, int2 edgeVerts, int visibility, float aaWidth)
{
    if (visibility == 0) return float4(0,0,0,0); // Hidden edge
    
    float3 v0 = normalize(corners[edgeVerts.x]);
    float3 v1 = normalize(corners[edgeVerts.y]);
    float3 p = normalize(fragPos); // Current point on hemisphere
    
    // HORIZON CLIPPING: Current fragment must be on front hemisphere
    if (p.z < 0.0f) 
        return float4(0,0,0,0);
    
    // HORIZON CLIPPING: Skip edge if both endpoints are behind horizon
    if (v0.z < 0.0f && v1.z < 0.0f) 
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

float4 drawHiddenEdges(float3 spherePos, int configIndex, float aaWidth)
{
    float4 color = float4(0,0,0,0);
    // Draw the remaining edges (non-silhouette) in a different color
    float3 hiddenEdgeColor = float3(0.3, 0.3, 0); // dark yellow color for hidden edges
    
    for (int i = 0; i < 12; i++)
    {
        int2 edge = allEdges[i];
        
        // Check if this edge is already drawn as a silhouette edge
        bool isSilhouette = false;
        int vertexCount = silhouettes[configIndex][0];
        // Draw the 6 silhouette edges
        for (int i = 0; i < vertexCount; i++) 
        {
            int v0Idx = silhouettes[configIndex][i + 1];
            int v1Idx = silhouettes[configIndex][((i + 1) % vertexCount) + 1];
            
            if ((edge.x == v0Idx && edge.y == v1Idx) || (edge.x == v1Idx && edge.y == v0Idx))
            {
                isSilhouette = true;
                break;
            }
        }
        
        // Only draw if it's not a silhouette edge
        if (!isSilhouette)
        {
            float4 edgeContribution = drawGreatCircleArc(spherePos, edge, 1, aaWidth);
            color += float4(hiddenEdgeColor * edgeContribution.a, edgeContribution.a);
        }
    }
    return color;
}

[[vk::location(0)]] float32_t4 main(SVertexAttributes vx) : SV_Target0
{
    float4 color = float4(0, 0, 0, 0);
    float2 p = toCircleSpace(vx.uv);
    
    // Convert 2D disk position to 3D hemisphere position
    float2 normalized = p / CIRCLE_RADIUS;
    float r2 = dot(normalized, normalized);
    float aaWidth = length(float2(ddx(vx.uv.x), ddy(vx.uv.y))); 

    if (all(vx.uv >= float2(0.49f, 0.49f) ) && all(vx.uv <= float2(0.51f, 0.51f)))
    {
        return float4(colorLUT[configIndex], 1.0f);
    }
    
    // Convert UV to 3D position on hemisphere
    float3 spherePos = normalize(float3(normalized.x, normalized.y, sqrt(1 - r2)));
    
    computeCubeGeo();
    
    // Get OBB center in world space
    float3 obbCenter = mul(pc.modelMatrix, float4(0, 0, 0, 1)).xyz;

    float3x3 rotMatrix = (float3x3)pc.modelMatrix;
    float3 proj = mul(obbCenter, rotMatrix); // Get all 3 projections at once

    // Get squared column lengths
    float lenSqX = dot(rotMatrix[0], rotMatrix[0]);
    float lenSqY = dot(rotMatrix[1], rotMatrix[1]);
    float lenSqZ = dot(rotMatrix[2], rotMatrix[2]);

    int3 region = int3(
        proj.x < -lenSqX ? 0 : (proj.x > lenSqX ? 2 : 1),
        proj.y < -lenSqY ? 0 : (proj.y > lenSqY ? 2 : 1),
        proj.z < -lenSqZ ? 0 : (proj.z > lenSqZ ? 2 : 1)
    );

    int configIndex = region.x + region.y * 3 + region.z * 9; // 0-26
    
    int vertexCount = silhouettes[configIndex][0];
    for (int i = 0; i < vertexCount; i++) 
    {
        int v0Idx = silhouettes[configIndex][i + 1];
        int v1Idx = silhouettes[configIndex][((i + 1) % vertexCount) + 1];
        
        float4 edgeContribution = drawGreatCircleArc(spherePos, int2(v0Idx, v1Idx), 1, aaWidth);
        color += float4(colorLUT[i] * edgeContribution.a, edgeContribution.a);
    }
    
    color += drawHiddenEdges(spherePos, configIndex, aaWidth);

    color += drawCorners(spherePos, aaWidth);
    
    color += drawRing(p, aaWidth);

    if (r2 > 1.1f)
        color.a = 0.0f; // Outside circle, make transparent
    
    return color;
}