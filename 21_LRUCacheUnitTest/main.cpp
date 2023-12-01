// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


// I've moved out a tiny part of this example into a shared header for reuse, please open and read it.
#include "../common/MonoSystemMonoLoggerApplication.hpp"

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;


class LRUCacheTestApp final : public nbl::examples::MonoSystemMonoLoggerApplication
{
		using base_t = examples::MonoSystemMonoLoggerApplication;
	public:
		using base_t::base_t;

		// we stuff all our work here because its a "single shot" app
		bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			// Remember to call the base class initialization!
			if (!base_t::onAppInitialized(std::move(system)))
				return false;

			m_logger->log("LRU cache unit test");
			m_logger->log("Testing large cache...");
			LRUCache<int, char> hugeCache(50000000u);
			hugeCache.insert(0, '0');
			hugeCache.print(m_logger);


			LRUCache<int, char> cache(5u);

			m_logger->log("Testing insert with const key, const val...");
			//const, const
			cache.insert(10, 'c');
			cache.insert(11, 'd');
			cache.insert(12, 'e');
			cache.insert(13, 'f');

			cache.print(m_logger);

			char returned = *(cache.get(11));
			assert(returned == 'd');
			returned = *(cache.get(10));
			assert(returned == 'c');
			returned = *(cache.get(13));
			assert(returned == 'f');

			cache.print(m_logger);

			//non const, const
			int i = 0;
			cache.insert(++i, '1');
			cache.insert(++i, '2');
			cache.insert(++i, '3');
			returned = *(cache.get(1));
			assert(returned == '1');

			//const, non const
			char ch = 'N';
			cache.insert(4, ch);

			returned = *(cache.peek(4));
			assert(returned == 'N');

			//non const, non const
			i = 6;
			ch = 'Y';
			cache.insert(i, ch);

			returned = *(cache.get(6));
			assert(returned == 'Y');

			returned = *(cache.get(i));
			assert(returned == ch);

			cache.erase(520);
			cache.erase(5);

			auto returnedNullptr = cache.get(5);
			assert(returnedNullptr == nullptr);
			auto peekedNullptr = cache.peek(5);
			assert(peekedNullptr == nullptr);

			core::LRUCache<int, std::string> cache2(5u);

			cache2.insert(500, "five hundred");			//inserts at addr = 0
			cache2.insert(510, "five hundred and ten");	//inserts at addr = 472
			cache2.insert(52, "fifty two");
			i = 20;
			cache2.insert(++i, "key is 21");
			cache2.insert(++i, "key is 22");
			cache2.insert(++i, "key is 23");
			i = 111;
			cache2.print(m_logger);
			cache2.insert(++i, "key is 112");

		#ifdef _NBL_DEBUG
			cache2.print(m_logger);
		#endif
			m_logger->log("all good");

			return true;
		}

		void workLoopBody() override {}

		bool keepRunning() override { return false; }
};

NBL_MAIN_FUNC(LRUCacheTestApp)