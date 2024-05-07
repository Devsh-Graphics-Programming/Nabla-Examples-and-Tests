// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


// I've moved out a tiny part of this example into a shared header for reuse, please open and read it.
#include "nbl/application_templates/MonoSystemMonoLoggerApplication.hpp"

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;


class LRUCacheTestApp final : public nbl::application_templates::MonoSystemMonoLoggerApplication
{
		using base_t = application_templates::MonoSystemMonoLoggerApplication;
	public:
		using base_t::base_t;

		constexpr static uint32_t InvalidTextureIdx = 41234;
		struct TextureReference
		{
			uint32_t alloc_idx;
			uint64_t lastUsedSemaphoreValue;

			TextureReference(uint32_t alloc_idx, uint64_t semaphoreVal) : alloc_idx(alloc_idx), lastUsedSemaphoreValue(semaphoreVal) {}
			TextureReference(uint64_t semaphoreVal) : TextureReference(InvalidTextureIdx, semaphoreVal) {}
			TextureReference() : TextureReference(InvalidTextureIdx, ~0ull) {}

			// In LRU Cache `insert` function, in case of cache hit, we need to assign semaphore value to TextureReference without changing `alloc_idx`
			inline TextureReference& operator=(uint64_t semamphoreVal) { lastUsedSemaphoreValue = semamphoreVal; return *this;  }
		};

		using TextureLRUCache = core::LRUCache<uint32_t, TextureReference>;

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
			m_logger->log("We're Referencing `10:c`");
			char returned = *(cache.get(10));
			cache.print(m_logger);
			m_logger->log("We're erasing `11:d`");
			cache.erase(11);
			assert(cache.get(11) == nullptr);
			cache.print(m_logger);
			m_logger->log("We're adding `11:d` again");
			cache.insert(11, 'd');
			cache.print(m_logger);

			assert(returned == 'c');
			returned = *(cache.get(11));
			assert(returned == 'd');
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

			m_textureLRUCache = TextureLRUCache(1024u);
			{
				SIntendedSubmitInfo intendedNextSubmit = {};
				auto evictionCallback = [&](const TextureReference& evicted)
					{
					};
				const auto nextSemaSignal = intendedNextSubmit.getFutureScratchSemaphore();
				TextureReference* inserted = m_textureLRUCache.insert(69420, nextSemaSignal.value, evictionCallback);
			}
			{
				SIntendedSubmitInfo intendedNextSubmit = {};
				auto evictionCallback = [&](const TextureReference& evicted)
					{
					};
				const auto nextSemaSignal = intendedNextSubmit.getFutureScratchSemaphore();
				TextureReference* inserted = m_textureLRUCache.insert(69420, nextSemaSignal.value, evictionCallback);
			}

		#ifdef _NBL_DEBUG
			cache2.print(m_logger);
		#endif
			m_logger->log("all good");

			constexpr uint32_t InvalidIdx = ~0u;
			struct TextureReference
			{
				uint32_t alloc_idx;
				uint64_t lastUsedSemaphoreValue;

				// copy ctor
				TextureReference(const TextureReference& tref)
				{
					assert(false); // based on the code in this test, copy constuctor shouldn't be called
				}
				TextureReference(TextureReference&& tref) = default;
				inline TextureReference& operator=(TextureReference&& tref) = default;

				TextureReference(uint32_t alloc_idx, uint64_t semaphoreVal) : alloc_idx(alloc_idx), lastUsedSemaphoreValue(semaphoreVal) {}
				TextureReference(uint64_t semaphoreVal) : TextureReference(InvalidIdx, semaphoreVal) {}
				TextureReference() : TextureReference(InvalidIdx, ~0ull) {}

				// In LRU Cache `insert` function, in case of cache hit, we need to assign semaphore value to TextureReference without changing `alloc_idx`
				inline TextureReference& operator=(uint64_t semamphoreVal) { lastUsedSemaphoreValue = semamphoreVal; return *this;  }
			};
			using TextureLRUCache = LRUCache<uint32_t, TextureReference>;

			TextureLRUCache textureCache = TextureLRUCache(3u);

			static_assert(std::is_assignable_v<TextureReference, uint64_t>);
			static_assert(std::is_constructible_v<TextureReference, uint64_t>);

			textureCache.insert(91u, TextureReference{ ~0u, 69u });
			textureCache.insert(92u, TextureReference{ 20u, 70u });
			textureCache.insert(93u, TextureReference{ 10u, 71u });
			auto t = textureCache.get(91u);
			assert(t->lastUsedSemaphoreValue == 69u); // make 91 jump to front, now 92 is the LRU
			// next insertion will evict because capacity is 3
			auto insertion = textureCache.insert(99u, 6999ull, [](const TextureReference& evictedTextureRef) -> void { assert(evictedTextureRef.alloc_idx == 20u); });
			assert(insertion->alloc_idx == InvalidIdx);
			assert(insertion->lastUsedSemaphoreValue == 6999ull);

			return true;
		}
		TextureLRUCache m_textureLRUCache;

		void workLoopBody() override {}

		bool keepRunning() override { return false; }
};

NBL_MAIN_FUNC(LRUCacheTestApp)