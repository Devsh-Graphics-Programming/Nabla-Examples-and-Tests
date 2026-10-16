#ifndef _NBL_THIS_EXAMPLE_BDA_ACCESSORS_HLSL_INCLUDED_
#define _NBL_THIS_EXAMPLE_BDA_ACCESSORS_HLSL_INCLUDED_


#include "nbl/builtin/hlsl/cpp_compat.hlsl"


namespace nbl
{
namespace this_example
{

#ifdef __HLSL_VERSION

// Generic read-only accessor backed by a buffer-device-address pointer.
// Stride is sizeof(V) (one V per index).
template<typename V>
struct BDAReadAccessor
{
   uint64_t base;

   static BDAReadAccessor<V> create(uint64_t _base)
   {
      BDAReadAccessor<V> r;
      r.base = _base;
      return r;
   }

   template<typename U, typename Idx>
   void get(Idx index, NBL_REF_ARG(U) val) NBL_CONST_MEMBER_FUNC
   {
      val = vk::RawBufferLoad<U>(base + uint64_t(index) * uint64_t(sizeof(U)));
   }
};

#endif

} // namespace this_example
} // namespace nbl
#endif
