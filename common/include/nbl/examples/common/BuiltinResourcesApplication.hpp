// Copyright (C) 2023-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXAMPLES_BUILTIN_RESOURCE_APPLICATION_HPP_INCLUDED_
#define _NBL_EXAMPLES_BUILTIN_RESOURCE_APPLICATION_HPP_INCLUDED_

// we need a system, logger and an asset manager
#include "nbl/application_templates/MonoAssetManagerApplication.hpp"

#ifdef NBL_EMBED_BUILTIN_RESOURCES
	#include "nbl/builtin/examples/include/CArchive.h"
	#include "nbl/builtin/examples/src/CArchive.h"
	#include "nbl/builtin/examples/build/CArchive.h"
	#if __has_include("nbl/this_example/builtin/CArchive.h")
		#include "nbl/this_example/builtin/CArchive.h"
	#endif
	// TODO: (**) there should be also 5th arch "nbl/this_example/builtin/build/CArchive.h"
	/*
		#if __has_include("nbl/this_example/builtin/build/CArchive.h")
		#include "nbl/this_example/builtin/build/CArchive.h"
		#endif
	*/
	//! this ain't meant to be the same as this_example ordinary archive
#endif

namespace nbl::examples
{

// Virtual Inheritance because apps might end up doing diamond inheritance
class BuiltinResourcesApplication : public virtual application_templates::MonoAssetManagerApplication
{
		using base_t = MonoAssetManagerApplication;

	public:
		using base_t::base_t;

	protected:
		// need this one for skipping passing all args into ApplicationFramework
		BuiltinResourcesApplication() = default;

		virtual bool onAppInitialized(core::smart_refctd_ptr<system::ISystem>&& system) override
		{
			if (!base_t::onAppInitialized(std::move(system)))
				return false;

			using namespace core;

			smart_refctd_ptr<system::IFileArchive> examplesHeaderArch,examplesSourceArch,examplesBuildSpirvArch,thisExampleArch;
			#ifdef NBL_EMBED_BUILTIN_RESOURCES
			examplesHeaderArch = core::make_smart_refctd_ptr<nbl::builtin::examples::include::CArchive>(smart_refctd_ptr(m_logger));
			examplesSourceArch = core::make_smart_refctd_ptr<nbl::builtin::examples::src::CArchive>(smart_refctd_ptr(m_logger));
			examplesBuildSpirvArch = core::make_smart_refctd_ptr<nbl::builtin::examples::build::CArchive>(smart_refctd_ptr(m_logger));

			#ifdef _NBL_THIS_EXAMPLE_BUILTIN_C_ARCHIVE_H_
				thisExampleArch = make_smart_refctd_ptr<nbl::this_example::builtin::CArchive>(smart_refctd_ptr(m_logger));
			#endif
			// TODO: (**)
			#else
			examplesHeaderArch = make_smart_refctd_ptr<system::CMountDirectoryArchive>(localInputCWD/"../common/include/nbl/examples",smart_refctd_ptr(m_logger),m_system.get());
			examplesSourceArch = make_smart_refctd_ptr<system::CMountDirectoryArchive>(localInputCWD/"../common/src/nbl/examples",smart_refctd_ptr(m_logger),m_system.get());
			examplesBuildSpirvArch = make_smart_refctd_ptr<system::CMountDirectoryArchive>(NBL_EXAMPLES_BUILD_MOUNT_POINT, smart_refctd_ptr(m_logger), m_system.get());
			thisExampleArch = make_smart_refctd_ptr<system::CMountDirectoryArchive>(localInputCWD/"app_resources",smart_refctd_ptr(m_logger),m_system.get());
			// TODO: (**)
			#endif
			// yes all 3 aliases are meant to be the same
			m_system->mount(std::move(examplesHeaderArch),"nbl/examples");
			m_system->mount(std::move(examplesSourceArch),"nbl/examples");
			m_system->mount(std::move(examplesBuildSpirvArch),"nbl/examples");
			if (thisExampleArch)
				m_system->mount(std::move(thisExampleArch),"app_resources");
			// TODO: (**)

			return true;
		}
};

}

#endif // _NBL_EXAMPLES_BUILTIN_RESOURCE_APPLICATION_HPP_INCLUDED_