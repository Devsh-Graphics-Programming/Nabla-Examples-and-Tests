// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_THIS_EXAMPLE_C_SCENE_LOADER_H_INCLUDED_
#define _NBL_THIS_EXAMPLE_C_SCENE_LOADER_H_INCLUDED_


#include "nabla.h"

#include "nbl/ext/MitsubaLoader/CMitsubaMetadata.h"


namespace nbl::this_example
{

class CSceneLoader : public core::IReferenceCounted, public core::InterfaceUnmovable
{
    public:
		struct SCachedCreationParams
		{
			core::smart_refctd_ptr<asset::IAssetManager> assMan = nullptr;
			system::logger_opt_smart_ptr logger = nullptr;
		};
		struct SCreationParams : SCachedCreationParams
		{
			inline operator bool() const
			{
				if (!assMan)
					return false;
				return true;
			}
		};
		static core::smart_refctd_ptr<CSceneLoader> create(SCreationParams&& params);

		struct SLoadResult
		{
			struct SSensor
			{
			};

			inline operator bool() const
			{
				if (!scene || !sensors.empty())
					return false;
				return true;
			}

			//
			core::smart_refctd_ptr<const asset::ICPUScene> scene = {};
			//
			core::vector<SSensor> sensors;
			// TODO: for Material Compiler
			//std::future<bool> compileShadersFuture = {};
		};
		struct SLoadParams
		{
			system::path relPath = "";
			system::path workingDirectory = "";
		};
		SLoadResult load(SLoadParams&& _params);

    protected:
		struct SConstructorParams : SCachedCreationParams
		{
		};
		inline CSceneLoader(SConstructorParams&& _params) : m_params(std::move(_params)) {}
		virtual inline ~CSceneLoader() {}

		SConstructorParams m_params;
};

}
#endif
