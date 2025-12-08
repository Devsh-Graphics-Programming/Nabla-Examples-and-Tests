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


static const float3 colorLUT[8] = {
    float3(0, 0, 0),        // 0: Black
    float3(1, 0, 0),       // 1: Red
    float3(0, 1, 0),       // 2: Green
    float3(1, 1, 0),       // 3: Yellow
    float3(0, 0, 1),       // 4: Blue
    float3(1, 0, 1),       // 5: Magenta
    float3(0, 1, 1),       // 6: Cyan
    float3(1, 1, 1)        // 7: White
};


    
// Vertices are ordered CCW relative to the camera view.
static const int silhouettes[8][6] = {
    {2, 3, 1, 5, 4, 6}, // 0: Black
    {6, 7, 5, 1, 0, 2}, // 1: Red
    {7, 6, 4, 0, 1, 3}, // 2: Green
    {3, 7, 5, 4, 0, 2}, // 3: Yellow
    {3, 2, 0, 4, 5, 7}, // 4: Cyan
    {1, 3, 7, 6, 4, 0}, // 5: Magenta
    {0, 1, 5, 7, 6, 2}, // 6: White
    {4, 6, 2, 3, 1, 5}  // 7: Gray
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

[[vk::location(0)]] float32_t4 main(SVertexAttributes vx) : SV_Target0
{
    float4 color = float4(0, 0, 0, 0);
    float2 p = toCircleSpace(vx.uv);
    
    // Convert 2D disk position to 3D hemisphere position
    float2 normalized = p / CIRCLE_RADIUS;
    float r2 = dot(normalized, normalized);
    
    // Convert UV to 3D position on hemisphere
    float3 spherePos = normalize(float3(normalized.x, normalized.y, sqrt(1 - r2)));
    
    computeCubeGeo();
    
    float3 obbCenter = mul(pc.modelMatrix, float4(0, 0, 0, 1)).xyz;
    
    float3 viewDir = obbCenter; 
    
    // Is this correct?
    float dotX = dot(viewDir, float3(pc.modelMatrix[0][0], pc.modelMatrix[1][0], pc.modelMatrix[2][0]));
    float dotY = dot(viewDir, float3(pc.modelMatrix[0][1], pc.modelMatrix[1][1], pc.modelMatrix[2][1]));
    float dotZ = dot(viewDir, float3(pc.modelMatrix[0][2], pc.modelMatrix[1][2], pc.modelMatrix[2][2]));

    // Determine octant from ray direction signs
    int octant = (dotX >= 0 ? 4 : 0) + 
                 (dotY >= 0 ? 2 : 0) + 
                 (dotZ >= 0 ? 1 : 0);

    if (all(vx.uv >= float2(0.49f, 0.49f) ) && all(vx.uv <= float2(0.51f, 0.51f)))
    {
        return float4(colorLUT[octant], 1.0f);
    }
    
    float aaWidth = length(float2(ddx(vx.uv.x), ddy(vx.uv.y))); 
    

    // Draw the 6 silhouette edges
    for (int i = 0; i < 6; i++) 
    {
        int v0Idx = silhouettes[octant][i];
        int v1Idx = silhouettes[octant][(i + 1) % 6];
        
        float4 edgeContribution = drawGreatCircleArc(spherePos, int2(v0Idx, v1Idx), 1, aaWidth);
        color += float4(colorLUT[i] * edgeContribution.a, edgeContribution.a);
    }
    
    // Draw the remaining edges (non-silhouette) in a different color
    float3 hiddenEdgeColor = float3(0.3, 0.3, 0.3); // Gray color for hidden edges
    
    for (int i = 0; i < 12; i++)
    {
        int2 edge = allEdges[i];
        
        // Check if this edge is already drawn as a silhouette edge
        bool isSilhouette = false;
        for (int j = 0; j < 6; j++)
        {
            int v0 = silhouettes[octant][j];
            int v1 = silhouettes[octant][(j + 1) % 6];
            
            if ((edge.x == v0 && edge.y == v1) || (edge.x == v1 && edge.y == v0))
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
    
    color += drawRing(p, aaWidth);

    // if (r2 > 1.1f)
    //     color.a = 0.0f; // Outside circle, make transparent
    
    return color;
}