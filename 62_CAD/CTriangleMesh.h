#pragma once

#include <nabla.h>
#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include "shaders/globals.hlsl"

using namespace nbl;

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

	inline size_t getVtxBuffByteSize() const
	{
		return sizeof(vertex_t) * m_vertices.size();
	}
	inline size_t getIdxBuffByteSize() const
	{
		return sizeof(index_t) * m_indices.size();
	}
	inline size_t getIdxCnt() const
	{
		return m_indices.size();
	}


private:
	core::vector<vertex_t> m_vertices;
	core::vector<index_t> m_indices;
};