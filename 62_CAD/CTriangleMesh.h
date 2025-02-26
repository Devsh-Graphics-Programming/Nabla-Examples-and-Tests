#pragma once

#include <nabla.h>
#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include "shaders/globals.hlsl"

using namespace nbl;

class CTriangleMesh final
{
public:
	inline void setVertices(core::vector<TriangleMeshVertex>&& vertices)
	{
		m_vertices = std::move(vertices);
	}
	inline void setIndices(core::vector<uint32_t>&& indices)
	{
		m_indices = std::move(indices);
	}

	inline const core::vector<TriangleMeshVertex>& getVertices() const
	{
		return m_vertices;
	}
	inline const core::vector<uint32_t>& getIndices() const
	{
		return m_indices;
	}

	inline size_t getVtxBuffByteSize() const
	{
		return sizeof(decltype(m_vertices)::value_type);
	}
	inline size_t getIdxBuffByteSize() const
	{
		return sizeof(decltype(m_indices)::value_type);
	}


private:
	core::vector<TriangleMeshVertex> m_vertices;
	core::vector<uint32_t> m_indices;
};