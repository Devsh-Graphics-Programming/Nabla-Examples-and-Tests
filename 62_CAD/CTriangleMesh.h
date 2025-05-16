#pragma once

#include <nabla.h>
#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include "shaders/globals.hlsl"

using namespace nbl;

struct DTMHeightShadingSettingsInfo
{
	// Height Shading Mode
	E_HEIGHT_SHADING_MODE heightShadingMode;

	// Used as fixed interval length for "DISCRETE_FIXED_LENGTH_INTERVALS" shading mode
	float intervalLength;

	// Converts an interval index to its corresponding height value
	// For example, if this value is 10.0, then an interval index of 2 corresponds to a height of 20.0.
	// This computed height is later used to determine the interpolated color for shading.
	// It makes sense for this variable to be always equal to `intervalLength` but sometimes it's a different scaling so that last index corresponds to largestHeight
	float intervalIndexToHeightMultiplier;

	// Used for "DISCRETE_FIXED_LENGTH_INTERVALS" shading mode
	// If `isCenteredShading` is true, the intervals are centered around `minHeight`, meaning the
	// first interval spans [minHeight - intervalLength / 2.0, minHeight + intervalLength / 2.0].
	// Otherwise, intervals are aligned from `minHeight` upward, so the first interval spans
	// [minHeight, minHeight + intervalLength].
	bool isCenteredShading;

	void addHeightColorMapEntry(float height, float32_t4 color)
	{
		heightColorSet.emplace(height, color);
	}

	bool fillShaderDTMSettingsHeightColorMap(DTMSettings& dtmSettings) const
	{
		const uint32_t mapSize = heightColorSet.size();
		if (mapSize > DTMHeightShadingSettings::HeightColorMapMaxEntries)
			return false;
		dtmSettings.heightShadingSettings.heightColorEntryCount = mapSize;

		int index = 0;
		for (auto it = heightColorSet.begin(); it != heightColorSet.end(); ++it)
		{
			dtmSettings.heightShadingSettings.heightColorMapHeights[index] = it->height;
			dtmSettings.heightShadingSettings.heightColorMapColors[index] = it->color;
			++index;
		}

		return true;
	}
	
private:
	struct HeightColor
	{
		float height;
		float32_t4 color;

		bool operator<(const HeightColor& other) const
		{
			return height < other.height;
		}
	};

	std::set<HeightColor> heightColorSet;
};

struct DTMContourSettingsInfo
{
	LineStyleInfo lineStyleInfo;

	float startHeight;
	float endHeight;
	float heightInterval;
};

struct DTMSettingsInfo
{
	static constexpr uint32_t MaxContourSettings = DTMSettings::MaxContourSettings;

	uint32_t mode = 0u; // related to E_DTM_MODE

	// outline
	LineStyleInfo outlineStyleInfo;
	// contours
	uint32_t contourSettingsCount = 0u;
	DTMContourSettingsInfo contourSettings[MaxContourSettings];
	// height shading
	DTMHeightShadingSettingsInfo heightShadingInfo;
};

class CTriangleMesh final
{
public:
	using index_t = uint32_t;
	using vertex_t = TriangleMeshVertex;

	inline void setVertices(core::vector<vertex_t>&& vertices)
	{
		m_vertices = std::move(vertices);
	}
	inline void setIndices(core::vector<uint32_t>&& indices)
	{
		m_indices = std::move(indices);
	}

	inline const core::vector<vertex_t>& getVertices() const
	{
		return m_vertices;
	}
	inline const core::vector<uint32_t>& getIndices() const
	{
		return m_indices;
	}

	inline size_t getVertexBuffByteSize() const
	{
		return sizeof(vertex_t) * m_vertices.size();
	}
	inline size_t getIndexBuffByteSize() const
	{
		return sizeof(index_t) * m_indices.size();
	}
	inline size_t getIndexCount() const
	{
		return m_indices.size();
	}
	
	inline void clear()
	{
		m_vertices.clear();
		m_indices.clear();
	}

	core::vector<vertex_t> m_vertices;
	core::vector<index_t> m_indices;
};