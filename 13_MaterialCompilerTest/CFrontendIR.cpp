// Copyright (C) 2022-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/asset/material_compiler3/CFrontendIR.h"


namespace nbl::asset::material_compiler3
{

constexpr auto ELL_ERROR = nbl::system::ILogger::E_LOG_LEVEL::ELL_ERROR;
using namespace nbl::system;

bool CFrontendIR::CEmitter::invalid(const SInvalidCheckArgs& args) const
{
	if (const auto* radianceNode=args.pool->deref(radiance); !radianceNode)
	{
		args.logger.log("Radiance node of correct type must be attached, but is %u of type %s",ELL_ERROR,radiance,args.pool->getTypeName(radiance).data());
		return false;
	}
	// not checking validty of profile because invalid means no emission profile
	// check for NaN and non invertible matrix
	if (profile && !(hlsl::determinant(profileTransform)>hlsl::numeric_limits<hlsl::float32_t>::min))
	{
		args.logger.log("Emission Profile's Transform is not an invertible matrix!");
		return true;
	}
	return false;
}

bool CFrontendIR::CBeer::invalid(const SInvalidCheckArgs& args) const
{
	if (!args.pool->deref(perpTransparency))
	{
		args.logger.log("Perpendicular Transparency node of correct type must be attached, but is %u of type %s",ELL_ERROR,perpTransparency,args.pool->getTypeName(perpTransparency).data());
		return true;
	}
	return false;
}

bool CFrontendIR::CFresnel::invalid(const SInvalidCheckArgs& args) const
{
	if (!args.pool->deref(orientedRealEta))
	{
		args.logger.log("Oriented Real Eta node of correct type must be attached, but is %u of type %s",ELL_ERROR,orientedRealEta,args.pool->getTypeName(orientedRealEta).data());
		return true;
	}
	if (!args.pool->deref(orientedImagEta))
	{
		args.logger.log("Oriented Imaginary Eta node of correct type must be attached, but is %u of type %s",ELL_ERROR,orientedImagEta,args.pool->getTypeName(orientedImagEta).data());
		return true;
	}
	return false;
}

bool CFrontendIR::COrenNayar::invalid(const SInvalidCheckArgs& args) const
{
	if (!ndParams)
	{	
		args.logger.log("Normal Distribution Parameters are invalid",ELL_ERROR);
		return true;
	}
	return false;
}

bool CFrontendIR::CCookTorrance::invalid(const SInvalidCheckArgs& args) const
{
	if (!ndParams)
	{	
		args.logger.log("Normal Distribution Parameters are invalid",ELL_ERROR);
		return true;
	}
	if (args.isBTDF && !args.pool->deref(orientedRealEta))
	{
		args.logger.log("Cook Torrance BTDF requires the Index of Refraction to compute the refraction direction, but is %u of type %s",ELL_ERROR,orientedRealEta,args.pool->getTypeName(orientedRealEta).data());
		return true;
	}
	return false;
}


auto CFrontendIR::reciprocate(const TypedHandle<const IExprNode> other) -> TypedHandle<IExprNode>
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

	core::vector<TypedHandle<const CLayer>> layerStack = m_rootNodes;
	core::stack<TypedHandle<const IExprNode>> exprStack;
	// TODO : print identifiers for root nodes/materials
	while (!layerStack.empty())
	{
		const auto* layerNode = deref(layerStack.back());
		layerStack.pop_back();
		if (layerNode->coated)
		{
			// TODO: print coating
			layerStack.push_back(layerNode->coated);
		}
		// TODO: print labelled edges
		exprStack.push(layerNode->brdfTop);
		exprStack.push(layerNode->btdf);
		exprStack.push(layerNode->brdfBottom);
		while (!exprStack.empty())
		{
			const auto entry = exprStack.top();
			exprStack.pop();
			const auto* node = deref(entry);
			str << "\t" << getNodeID(entry) << " [label=" << node->getTypeName() << "\\n";
			if (const auto* debug = deref(node->debugInfo); debug && !debug->data().empty())
				str << std::string_view(reinterpret_cast<const char*>(debug->data().data()),debug->data().size());
			str << "]\n\t" << getNodeID(entry) << " -> {";
			const auto childCount = node->getChildCount();
			for (auto childIx=0; childIx<childCount; childIx++)
			{
				const auto childHandle = node->getChildHandle(childIx);
				if (const auto child=deref(childHandle); child)
				{
					str << getNodeID(childHandle) << " ";
					exprStack.push(childHandle);
				}
			}
			str << "}\n";
		}
	}

	str << "}\n";
}

}