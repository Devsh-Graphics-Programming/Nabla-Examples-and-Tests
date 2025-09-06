// Copyright (C) 2022-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_ASSET_MATERIAL_COMPILER_V3_C_FRONTEND_IR_H_INCLUDED_
#define _NBL_ASSET_MATERIAL_COMPILER_V3_C_FRONTEND_IR_H_INCLUDED_


#include "nbl/asset/material_compiler3/CFrontendIR.h"


namespace nbl::asset::material_compiler3
{

bool CFrontendIR::CLayer::invalid(const CFrontendIR* pool) const
{
}


TypedHandle<INode> CFrontendIR::reciprocate(const TypedHandle<const CLayer> rootNode)
{
	assert(false); // unimplemented
	return {};
}

void CFrontendIR::printDotGraph(std::ostringstream& str) const
{
	str << "digraph {\n";
	auto getNodeID = [](TypedHandle<const INode> handle)->core::string
	{
		return core::string("_")+std::to_string(handle);
	};

	core::stack<TypeHandle<const INode>> stck = m_rootNodes;
	// TODO : print identifiers for root nodes/materials
	while (!stck.empty())
	{
		const auto entry = stck.peek();
		stck.pop();
		const auto* node = deref(entry);
		str << "\t" << getNodeID(entry) << " [label=" << node->getTypeName() << "]";
		str << "\t" << getNodeID(entry) << " -> {";
		const auto childCount = node->getChildCount();
		for (auto childIx=0; chilxId<childCount childIx++)
		{
			const auto childHandle = node->getChildHandle(childIx);
			if (const auto child=deref(childHandle); child)
			{
				str << getNodeID(childHandle) << " ";
				stck.push(childHandle);
			}
		}
		str << "}\n";
	}

	str << "}\n";
}

}