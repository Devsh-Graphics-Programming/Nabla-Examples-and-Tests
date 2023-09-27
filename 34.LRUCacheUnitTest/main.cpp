#define _IRR_STATIC_LIB_
#include <nabla.h>
#include "nbl/core/containers/LRUcache.h"

using namespace nbl;
using namespace nbl::core;

int main()
{
	auto logger = make_smart_refctd_ptr<CLogger>(nullptr);
	logger->log("LRU cache unit test");
	logger->log("Testing large cache...");
	LRUCache<int, char> hugeCache(50000000u);
	hugeCache.insert(0, '0');
#ifdef _NBL_DEBUG
	hugeCache.print(logger);
#endif


	LRUCache<int, char> cache(5u);

	logger->log("Testing insert with const key, const val...");
	//const, const
	cache.insert(10, 'c');
	cache.insert(11, 'd');
	cache.insert(12, 'e');
	cache.insert(13, 'f');

#ifdef _NBL_DEBUG
	cache.print(logger);
#endif

	char returned = *(cache.get(11));
	assert(returned == 'd');
	returned = *(cache.get(10));
	assert(returned == 'c');
	returned = *(cache.get(13));
	assert(returned == 'f');

#ifdef _NBL_DEBUG
	cache.print(logger);
#endif

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
#ifdef _NBL_DEBUG
	cache2.print(logger);
#endif
	cache2.insert(++i, "key is 112");

#ifdef _NBL_DEBUG
	cache2.print(logger);
#endif
	logger->log("all good");
	return 0;
}
