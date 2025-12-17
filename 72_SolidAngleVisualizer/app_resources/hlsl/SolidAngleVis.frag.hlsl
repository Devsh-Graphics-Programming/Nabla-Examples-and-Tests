#pragma wave shader_stage(fragment)

#include "common.hlsl"
#include <nbl/builtin/hlsl/ext/FullScreenTriangle/SVertexAttributes.hlsl>
#include "utils.hlsl"

using namespace nbl::hlsl;
using namespace ext::FullScreenTriangle;

[[vk::push_constant]] struct PushConstants pc;
[[vk::binding(0, 0)]] RWStructuredBuffer<ResultData> DebugDataBuffer;

static const float CIRCLE_RADIUS = 0.5f;

// --- Geometry Utils ---

static const float3 constCorners[8] = {
	float3(-1, -1, -1), float3(1, -1, -1), float3(-1,  1, -1), float3(1,  1, -1),
	float3(-1, -1,  1), float3(1, -1,  1), float3(-1,  1,  1), float3(1,  1,  1)
};

static const int2 allEdges[12] = {
	{0, 1}, {2, 3}, {4, 5}, {6, 7}, // X axis
	{0, 2}, {1, 3}, {4, 6}, {5, 7}, // Y axis
	{0, 4}, {1, 5}, {2, 6}, {3, 7}  // Z axis
};

// Adjacency of edges to faces
// Corrected Adjacency of edges to faces
static const int2 edgeToFaces[12] = {
	// Edge Index:  | allEdges[i]  | Shared Faces: 

	/* 0 (0-1) */   {4, 0},         // Y- (4) and Z- (0)
	/* 1 (2-3) */   {5, 0},         // Y+ (5) and Z- (0)
	/* 2 (4-5) */   {4, 1},         // Y- (4) and Z+ (1)
	/* 3 (6-7) */   {5, 1},         // Y+ (5) and Z+ (1)

	/* 4 (0-2) */   {2, 0},         // X- (2) and Z- (0)
	/* 5 (1-3) */   {3, 0},         // X+ (3) and Z- (0)
	/* 6 (4-6) */   {2, 1},         // X- (2) and Z+ (1)
	/* 7 (5-7) */   {3, 1},         // X+ (3) and Z+ (1)

	/* 8 (0-4) */   {2, 4},         // X- (2) and Y- (4)
	/* 9 (1-5) */   {3, 4},         // X+ (3) and Y- (4)
	/* 10 (2-6) */  {2, 5},         // X- (2) and Y+ (5)
	/* 11 (3-7) */  {3, 5}          // X+ (3) and Y+ (5)
};
static float3 corners[8];
static float3 faceCenters[6] = {
	float3(0,0,0), float3(0,0,0), float3(0,0,0),
	float3(0,0,0), float3(0,0,0), float3(0,0,0)
};

static const float3 localNormals[6] = {
	float3(0, 0, -1), // Face 0 (Z-)
	float3(0, 0, 1),  // Face 1 (Z+)
	float3(-1, 0, 0), // Face 2 (X-)
	float3(1, 0, 0),  // Face 3 (X+)
	float3(0, -1, 0), // Face 4 (Y-)
	float3(0, 1, 0)   // Face 5 (Y+)
};


// TODO: unused, remove later
// Vertices are ordered CCW relative to the camera view.
static const int silhouettes[27][7] = {
	{6, 1, 3, 2, 6, 4, 5},      // 0: Black
	{6, 2, 6, 4, 5, 7, 3},      // 1: White 
	{6, 0, 4, 5, 7, 3, 2},      // 2: Gray 
	{6, 1, 3, 7, 6, 4, 5,},     // 3: Red 
	{4, 4, 5, 7, 6, -1, -1},    // 4: Green 
	{6, 0, 4, 5, 7, 6, 2},      // 5: Blue 
	{6, 0, 1, 3, 7, 6, 4},      // 6: Yellow 
	{6, 0, 1, 5, 7, 6, 4},      // 7: Magenta              
	{6, 0, 1, 5, 7, 6, 2},      // 8: Cyan 
	{6, 1, 3, 2, 6, 7, 5},      // 9: Orange
	{4, 2, 6, 7, 3, -1, -1},    // 10: Light Orange
	{6, 0, 4, 6, 7, 3, 2},      // 11: Dark Orange
	{4, 1, 3, 7, 5, -1, -1},    // 12: Pink
	{6, 0, 4, 6, 7, 3, 2},      // 13: Light Pink
	{4, 0, 4, 6, 2, -1, -1},    // 14: Deep Rose
	{6, 0, 1, 3, 7, 5, 4},      // 15: Purple
	{4, 0, 1, 5, 4, -1, -1},    // 16: Light Purple
	{6, 0, 1, 5, 4, 6, 2},      // 17: Indigo
	{6, 0, 2, 6, 7, 5, 1},      // 18: Dark Green
	{6, 0, 2, 6, 7, 3, 1},      // 19: Lime
	{6, 0, 4, 6, 7, 3, 1},      // 20: Forest Green
	{6, 0, 2, 3, 7, 5, 1},      // 21: Navy
	{4, 0, 2, 3, 1, -1, -1},    // 22: Sky Blue
	{6, 0, 4, 6, 2, 3, 1},      // 23: Teal
	{6, 0, 2, 3, 7, 5, 4},      // 24: Brown
	{6, 0, 2, 3, 1, 5, 4},      // 25: Tan/Beige
	{6, 1, 5, 4, 6, 2, 3}       // 26: Dark Brown
};

// Binary packed silhouettes
static const uint32_t binSilhouettes[27] = {
	0b11000000000000101100110010011001,
	0b11000000000000011111101100110010,
	0b11000000000000010011111101100000,
	0b11000000000000101100110111011001,
	0b10000000000000000000110111101100,
	0b11000000000000010110111101100000,
	0b11000000000000100110111011001000,
	0b11000000000000100110111101001000,
	0b11000000000000010110111101001000,
	0b11000000000000101111110010011001,
	0b10000000000000000000011111110010,
	0b11000000000000010011111110100000,
	0b10000000000000000000101111011001,
	0b11000000000000010011111110100000,
	0b10000000000000000000010110100000,
	0b11000000000000100101111011001000,
	0b10000000000000000000100101001000,
	0b11000000000000010110100101001000,
	0b11000000000000001101111110010000,
	0b11000000000000001011111110010000,
	0b11000000000000001011111110100000,
	0b11000000000000001101111011010000,
	0b10000000000000000000001011010000,
	0b11000000000000001011010110100000,
	0b11000000000000100101111011010000,
	0b11000000000000100101001011010000,
	0b11000000000000011010110100101001,
};

int getSilhouetteVertex(uint32_t packedSil, int index)
{
	return (packedSil >> (3 * index)) & 0x7;
}

// Get silhouette size
int getSilhouetteSize(uint32_t sil)
{
	return (sil >> 29) & 0x7;

}

// Check if vertex has negative z
bool getVertexZNeg(int vertexIdx)
{
	return normalize(corners[vertexIdx]).z < 0.0f;
}

#include "Drawing.hlsl"


void setDebugData(uint32_t sil, int3 region, int configIndex, uint32_t clippedVertexCount)
{
#if DEBUG_DATA
	DebugDataBuffer[0].silhouetteVertexCount = uint32_t(getSilhouetteSize(sil));
	DebugDataBuffer[0].region = uint3(region);
	DebugDataBuffer[0].silhouetteIndex = uint32_t(configIndex);
	DebugDataBuffer[0].clippedVertexCount = clippedVertexCount;
	for (int i = 0; i < 6; i++)
	{
		DebugDataBuffer[0].vertices[i] = uint32_t(getSilhouetteVertex(sil, i));
	}
	DebugDataBuffer[0].silhouette = sil;
#endif
}

float2 toCircleSpace(float2 uv)
{
	float2 p = uv * 2.0f - 1.0f;
	float aspect = pc.viewport.z / pc.viewport.w;
	p.x *= aspect;
	return p;
}

uint32_t packSilhouette(const int s[7]) 
{
	uint32_t packed = 0;
	int size = s[0] & 0x7; // 3 bits for size

	// Pack vertices LSB-first (vertex1 in lowest 3 bits above size)
	for (int i = 1; i <= 6; ++i) {
		int v = s[i];
		if (v < 0) v = 0; // replace unused vertices with 0
		packed |= (v & 0x7) << (3 * (i - 1)); // vertex i-1 shifted by 3*(i-1)
	}

	// Put size in the MSB (bits 29-31 for a 32-bit uint, leaving 29 bits for vertices)
	packed |= (size & 0x7) << 29;

	return packed;
}

void computeCubeGeo()
{
	for (int i = 0; i < 8; i++)
	for (int i = 0; i < 8; i++)
	{
		float3 localPos = constCorners[i];
		float3 worldPos = mul(pc.modelMatrix, float4(localPos, 1.0f)).xyz;
		corners[i] = worldPos.xyz;
		faceCenters[i / 4] += worldPos / 4.0f;
		faceCenters[2 + i % 2] += worldPos / 4.0f;
		faceCenters[4 + (i / 2) % 2] += worldPos / 4.0f;
	}
}

[[vk::location(0)]] float32_t4 main(SVertexAttributes vx) : SV_Target0
{
	float4 color = float4(0, 0, 0, 0);
	float aaWidth = length(float2(ddx(vx.uv.x), ddy(vx.uv.y)));
	float2 p = toCircleSpace(vx.uv);

	float2 normalized = p / CIRCLE_RADIUS;
	float r2 = dot(normalized, normalized);

	float3 spherePos;
	if (r2 <= 1.0f)
	{
		spherePos = float3(normalized.x, normalized.y, sqrt(1.0f - r2));
	}
	else
	{
		float uv2Plus1 = r2 + 1.0f;
		spherePos = float3(normalized.x * 2.0f, normalized.y * 2.0f, 1.0f - r2) / uv2Plus1;
	}
	spherePos = normalize(spherePos);

	computeCubeGeo();

	float3 obbCenter = mul(pc.modelMatrix, float4(0, 0, 0, 1)).xyz;

	float3x3 upper3x3 = (float3x3)pc.modelMatrix;

#if 1
	// Compute reciprocal scales
	float3 rcpScales = rsqrt(float3(
		dot(upper3x3[0], upper3x3[0]),
		dot(upper3x3[1], upper3x3[1]),
		dot(upper3x3[2], upper3x3[2])
	));

	// Build inverse-rotation-only matrix
	float3x3 invRot;
	invRot[0] = upper3x3[0] * rcpScales.x;
	invRot[1] = upper3x3[1] * rcpScales.y;
	invRot[2] = upper3x3[2] * rcpScales.z;

	// Project center into OBB local space
	float3 normalizedProj = mul(invRot, obbCenter);
#else
	float3 normalizedProj = mul(inverse(upper3x3), obbCenter);
#endif
	int3 region = int3(
		normalizedProj.x < -1.0f ? 0 : (normalizedProj.x > 1.0f ? 2 : 1),
		normalizedProj.y < -1.0f ? 0 : (normalizedProj.y > 1.0f ? 2 : 1),
		normalizedProj.z < -1.0f ? 0 : (normalizedProj.z > 1.0f ? 2 : 1)
	);
	int configIndex = region.x + region.y * 3 + region.z * 9;

	// uint32_t sil = packSilhouette(silhouettes[configIndex]);
	uint32_t sil = binSilhouettes[configIndex];

	int vertexCount = getSilhouetteSize(sil);
	bool longSilhouette = (vertexCount == 6);
	uint32_t silEdgeMask = 0;

#if DEBUG_DATA
	{
		for (int i = 0; i < vertexCount; i++)
		{
			int vIdx = i % vertexCount;
			int v1Idx = (i + 1) % vertexCount;

			int v0Corner = getSilhouetteVertex(sil, vIdx);
			int v1Corner = getSilhouetteVertex(sil, v1Idx);
			// Mark edge as part of silhouette 
			for (int e = 0; e < 12; e++)
			{
				int2 edge = allEdges[e];
				if ((edge.x == v0Corner && edge.y == v1Corner) ||
				(edge.x == v1Corner && edge.y == v0Corner))
				{
					silEdgeMask |= (1u << e);
				}
			}
		}
		validateEdgeVisibility(sil, vertexCount, silEdgeMask);
	}
#endif
	// Build clip mask for vertices below horizon (z < 0)
	uint32_t clipMask = 0u;
	NBL_UNROLL
	for (int i = 0; i < 6; i++)
	{
		if (i >= vertexCount) break;
		clipMask |= (getVertexZNeg(getSilhouetteVertex(sil, i)) ? 1u : 0u) << i;
	}

	int clipCount = countbits(clipMask);

	// Total clipped vertices
	int clippedVertCount = vertexCount + (clipMask != 0u ? (2 - clipCount) : 0);

	// Find rotation amount to place positive vertices first
	int rotateAmount = 0;
	if (clipMask != 0u)
	{
		uint32_t invertedMask = ~clipMask & ((1u << vertexCount) - 1u);
		bool wrapAround = ((clipMask & 1u) != 0u) && ((clipMask >> (vertexCount - 1)) & 1u);

		rotateAmount = wrapAround ?
			((firstbithigh(invertedMask) + 1) % vertexCount) :
			firstbitlow(clipMask);
	}

	// Rotate silhouette bits
	uint32_t vertexBits = sil & 0x1FFFFFFF;
	uint32_t rotatedVertexBits = rotr(vertexBits, rotateAmount * 3, vertexCount * 3);
	uint32_t rotatedSil = (sil & 0xE0000000) | rotatedVertexBits;

	// Rotate the clip mask to match
	uint32_t rotatedClipMask = rotr(clipMask, rotateAmount, vertexCount);

	// Draw clipped silhouette edges
	for (int i = 0; i < clippedVertCount; i++)
	{
		int nextI = (i + 1) % clippedVertCount;

		int vIdx = i % vertexCount;
		int v1Idx = nextI % vertexCount;

		// Extract clip bits directly
		bool v0Clipped = (rotatedClipMask >> vIdx) & 1u;
		bool v1Clipped = (rotatedClipMask >> v1Idx) & 1u;

		// Skip if both clipped
		if (v0Clipped && v1Clipped) continue;

		int v0Corner = getSilhouetteVertex(rotatedSil, vIdx);
		int v1Corner = getSilhouetteVertex(rotatedSil, v1Idx);

		float3 v0 = normalize(corners[v0Corner]);
		float3 v1 = normalize(corners[v1Corner]);

		float3 points[2] = { corners[v0Corner], corners[v1Corner] };

		// Clip using bit state
		if (v0Clipped)
		{
			float t = v0.z / (v0.z - v1.z);
			points[0] = normalize(lerp(corners[v0Corner], corners[v1Corner], t));
		}
		else if (v1Clipped)
		{
			float t = v0.z / (v0.z - v1.z);
			points[1] = normalize(lerp(corners[v0Corner], corners[v1Corner], t));
		}

		// Draw edge
		float4 edgeContribution = drawGreatCircleArc(spherePos, points, 1, aaWidth);
		color += float4(colorLUT[i] * edgeContribution.a, edgeContribution.a);

	}


	setDebugData(sil, region, configIndex, clippedVertCount);

	color += drawHiddenEdges(spherePos, silEdgeMask, aaWidth);
	color += drawCorners(spherePos, p, aaWidth);
	color += drawRing(p, aaWidth);

	if (all(vx.uv >= float2(0.49f, 0.49f)) && all(vx.uv <= float2(0.51f, 0.51f)))
	{
		return float4(colorLUT[configIndex], 1.0f);
	}

	return color;
}