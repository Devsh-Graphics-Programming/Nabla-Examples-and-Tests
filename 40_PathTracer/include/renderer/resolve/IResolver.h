// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_THIS_EXAMPLE_I_RESOLVER_H_INCLUDED_
#define _NBL_THIS_EXAMPLE_I_RESOLVER_H_INCLUDED_


#include "renderer/CSession.h"


namespace nbl::this_example
{

class IResolver : public core::IReferenceCounted, public core::InterfaceUnmovable
{
    public:
		//
		inline CSession* getActiveSession() {return m_activeSession.get();}
		inline const CSession* getActiveSession() const {return m_activeSession.get();}

		//
		virtual uint64_t computeScratchSize(const CSession* session) const = 0;
		inline uint64_t computeScratchSize() const {return computeScratchSize(m_activeSession.get());}

		//
		inline bool changeSession(core::smart_refctd_ptr<CSession>&& session)
		{
			m_activeSession = std::move(session);
			if (!m_activeSession || !m_activeSession->isInitialized() || !changeSession_impl())
			{
				m_activeSession = {};
				return false;
			}
			return true;
		}

		//
		virtual bool resolve(video::IGPUCommandBuffer* cv, video::IGPUBuffer* scratch) = 0;

    protected:
		virtual bool changeSession_impl() = 0;

		core::smart_refctd_ptr<CSession> m_activeSession;
};

}
#endif
