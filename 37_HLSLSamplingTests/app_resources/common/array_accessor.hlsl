#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_ARRAY_ACCESSOR_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_ARRAY_ACCESSOR_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

using namespace nbl::hlsl;

// Generic read-only accessor over a fixed-size array, for HLSL/C++ dual compilation.
template<typename T, uint32_t N>
struct ArrayAccessor
{
	using value_type = T;
	template<typename V, typename I>
	void get(I i, NBL_REF_ARG(V) val) NBL_CONST_MEMBER_FUNC { val = V(data[i]); }
	T operator[](uint32_t i) NBL_CONST_MEMBER_FUNC { return data[i]; }
	T data[N];
};

#endif
