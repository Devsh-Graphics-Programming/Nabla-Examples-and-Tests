#pragma once

#include <nabla.h>
#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include "shaders/globals.hlsl"

using namespace nbl;

struct DTMSettingsInfo
{
	enum E_HEIGHT_SHADING_MODE
	{
		DISCRETE_VARIABLE_LENGTH_INTERVALS,
		DISCRETE_FIXED_LENGTH_INTERVALS,
		CONTINOUS_INTERVALS
	};

	LineStyleInfo outlineLineStyleInfo;
	LineStyleInfo contourLineStyleInfo;
	
	float contourLinesStartHeight;
	float contourLinesEndHeight;
	float contourLinesHeightInterval;

	float intervalWidth;
	E_HEIGHT_SHADING_MODE heightShadingMode;

	void addHeightColorMapEntry(float height, float32_t4 color)
	{
		heightColorSet.emplace(height, color);
	}

	bool fillShaderDTMSettingsHeightColorMap(DTMSettings& dtmSettings) const
	{
		const uint32_t mapSize = heightColorSet.size();
		if (mapSize > DTMSettings::HeightColorMapMaxEntries)
			return false;
		dtmSettings.heightColorEntryCount = mapSize;

		int index = 0;
		for (auto it = heightColorSet.begin(); it != heightColorSet.end(); ++it)
		{
			dtmSettings.heightColorMapHeights[index] = it->height;
			dtmSettings.heightColorMapColors[index] = it->color;
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

class CTriangleMesh final
{
public:
	using index_t = uint32_t;
	using vertex_t = TriangleMeshVertex;

	struct DrawData
	{
		PushConstants pushConstants;
		uint64_t indexBufferOffset;
		uint64_t indexCount;
	};

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


private:
	core::vector<vertex_t> m_vertices;
	core::vector<index_t> m_indices;
};