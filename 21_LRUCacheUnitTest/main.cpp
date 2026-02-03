// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


// I've moved out a tiny part of this example into a shared header for reuse, please open and read it.
#include "nbl/application_templates/MonoSystemMonoLoggerApplication.hpp"
#include <ranges>

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

		using TextureLRUCache = core::ResizableLRUCache<uint32_t, TextureReference>;

		// we stuff all our work here because its a "single shot" app
		bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			// Remember to call the base class initialization!
			if (!base_t::onAppInitialized(std::move(system)))
				return false;

			m_logger->log("LRU cache unit test");
			m_logger->log("Testing large cache...");
			ResizableLRUCache<int, char> hugeCache(50000000u);
			hugeCache.insert(0, '0');
			hugeCache.print(m_logger);

			// Use this to ensure the disposal function is properly called on every element of the first cache
			ResizableLRUCache<int, char>::disposal_func_t df([&](std::pair<int, char> a) 
				{ 
					std::ostringstream tmp;
					tmp << "Disposal function called on element (" << a.first << ", " << a.second << ")";
					m_logger->log(tmp.str()); 
				});
			ResizableLRUCache<int, char> cache(5u, std::move(df));

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
			assert(cache.getState() == "{12, e}, {10, c}, {11, d}, {13, f}, {1, 1}");
			cache.insert(++i, '2');
			assert(cache.getState() == "{10, c}, {11, d}, {13, f}, {1, 1}, {2, 2}");
			cache.insert(++i, '3');
			assert(cache.getState() == "{11, d}, {13, f}, {1, 1}, {2, 2}, {3, 3}");
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

			// Try clearing the cache
			m_logger->log("Clearing test");
			m_logger->log("Print contents before clearing");
			cache.print(m_logger);
			assert(cache.getState() == "{2, 2}, {3, 3}, {1, 1}, {4, N}, {6, Y}");
			cache.clear();
			m_logger->log("Print contents after clearing");
			cache.print(m_logger);
			assert(cache.getState() == "");

			ResizableLRUCache<int, std::string> cache2(5u);

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

			// Grow test - try growing the cache
			m_logger->log("Growing test");
			auto previousState = cache2.getState();
			// Grow cache
			assert(cache2.grow(10));
			// Cache state should be the same
			assert(cache2.getState() == previousState);
			cache2.print(m_logger);
			cache2.insert(++i, "key is 113");
			cache2.insert(++i, "key is 114");
			cache2.insert(++i, "key is 115");
			cache2.insert(++i, "key is 116");
			cache2.insert(++i, "key is 117");
			cache2.print(m_logger);
			// Should evict key 52
			cache2.insert(++i, "key is 118");
			cache2.print(m_logger);
			const auto latestState = cache2.getState();
			assert(latestState == "{21, key is 21}, {22, key is 22}, {23, key is 23}, {112, key is 112}, {113, key is 113}, {114, key is 114}, {115, key is 115}, {116, key is 116}, {117, key is 117}, {118, key is 118}");
			// Invalid grow should fail
			assert(!cache2.grow(5));
			assert(!cache2.grow(10));
			// Call a bunch of grows that shouldn't fail and some others that should
			for (auto i = 1u; i < 50; i++)
			{
				assert(cache2.grow(50 * i));
				assert(!cache2.grow(25 * i));
				assert(!cache2.grow(50 * i));
				assert(cache2.getState() == latestState);
			}
			
			// Single element cache test - checking for edge cases
			ResizableLRUCache<int, std::string> cache3(1u);
			cache3.insert(0, "foo");
			cache3.insert(1, "bar");
			cache3.clear();

			// Cache iterator test
			constexpr uint32_t cache4Size = 10;
			ResizableLRUCache<uint32_t, uint32_t> cache4(cache4Size);
			for (auto i = 0u; i < cache4Size; i++)
			{
				cache4.insert(i, i);
			}
			// Default iterator is MRU -> LRU
			uint32_t counter = cache4Size - 1;
			for (auto& pair : cache4)
			{
				assert(pair.first == counter && pair.second == counter);
				counter--;
			}
			// Reverse LRU -> MRU traversal
			counter = 0u;
			for (auto it = cache4.crbegin(); it != cache4.crend(); it++)
			{
				assert(it->first == counter && it->second == counter);
				counter++;
			}

			// Cache copy test
			ResizableLRUCache<uint32_t, uint32_t> cache4Copy(cache4);
			for (auto it = cache4.cbegin(), itCopy = cache4Copy.cbegin(); it != cache4.cend(); it++, itCopy++)
			{
				assert(*it == *itCopy);
				// Assert deep copy
				assert(it.operator->() != itCopy.operator->());

			}

			// Besides the disposal function that gets called when evicting, we need to check that the Cache properly destroys all resident `Key,Value` pairs when destroyed
			struct Foo
			{
				int* destroyCounter;

				Foo(int* _destroyCounter) : destroyCounter(_destroyCounter){}

				void operator=(Foo&& other)
				{
					destroyCounter = other.destroyCounter;
					other.destroyCounter = nullptr;
				}

				Foo(Foo&& other)
				{
					operator=(std::move(other));
				}

				~Foo()
				{
					// Only count destructions of objects resident in Cache and not ones that happen right after moving out of
					if (destroyCounter)
						(*destroyCounter)++;
				}
			};

			int destroyCounter = 0;
			{
				ResizableLRUCache<int, Foo> cache5(10u);
				for (int i = 0; i < 10; i++)
					cache5.insert(i, Foo(&destroyCounter));
				int x = 0;
			}
			assert(destroyCounter == 10);

			m_logger->log("all good");

			m_textureLRUCache = std::unique_ptr<TextureLRUCache>(new TextureLRUCache(1024u));
			{
				SIntendedSubmitInfo intendedNextSubmit = {};
				auto evictionCallback = [&](const TextureReference& evicted)
					{
					};
				const auto nextSemaSignal = intendedNextSubmit.getFutureScratchSemaphore();
				TextureReference* inserted = m_textureLRUCache->insert(69420, nextSemaSignal.value, evictionCallback);
			}
			{
				SIntendedSubmitInfo intendedNextSubmit = {};
				auto evictionCallback = [&](const TextureReference& evicted)
					{
					};
				const auto nextSemaSignal = intendedNextSubmit.getFutureScratchSemaphore();
				TextureReference* inserted = m_textureLRUCache->insert(69420, nextSemaSignal.value, evictionCallback);
			}

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
			using TextureLRUCache = ResizableLRUCache<uint32_t, TextureReference>;

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
		std::unique_ptr<TextureLRUCache> m_textureLRUCache;

		void workLoopBody() override {}

		bool keepRunning() override { return false; }
};

NBL_MAIN_FUNC(LRUCacheTestApp)