#include "common.hlsl"

struct TestExecutor
{
	void operator()(NBL_CONST_REF_ARG(InputTestValues) input, NBL_REF_ARG(TestValues) output)
	{
		uint64_t2 Vec2A = { input.coordX, input.coordY };
		uint64_t2 Vec2B = { input.coordZ, input.coordW };
	
		uint64_t3 Vec3A = { input.coordX, input.coordY, input.coordZ };
		uint64_t3 Vec3B = { input.coordY, input.coordZ, input.coordW };
	
		uint64_t4 Vec4A = { input.coordX, input.coordY, input.coordZ, input.coordW };
		uint64_t4 Vec4B = { input.coordY, input.coordZ, input.coordW, input.coordX };
	
		uint16_t4 Vec4BFull = createAnyBitIntegerVecFromU64Vec<uint16_t, false, fullBits_4, 4>(Vec4B);
		int32_t2 Vec2BSignedFull = createAnyBitIntegerVecFromU64Vec<int32_t, true, fullBits_2, 2>(Vec2B);
		int32_t3 Vec3BSignedFull = createAnyBitIntegerVecFromU64Vec<int32_t, true, fullBits_3, 3>(Vec3B);
		int16_t4 Vec4BSignedFull = createAnyBitIntegerVecFromU64Vec<int16_t, true, fullBits_4, 4>(Vec4B);
	
		morton::code<false, fullBits_4, 4, emulated_uint64_t> morton_emulated_4A = createMortonFromU64Vec<false, fullBits_4, 4, emulated_uint64_t>(Vec4A);
		morton::code<true, fullBits_2, 2, emulated_uint64_t> morton_emulated_2_signed = createMortonFromU64Vec<true, fullBits_2, 2, emulated_uint64_t>(Vec2A);
		morton::code<true, fullBits_3, 3, emulated_uint64_t> morton_emulated_3_signed = createMortonFromU64Vec<true, fullBits_3, 3, emulated_uint64_t>(Vec3A);
		morton::code<true, fullBits_4, 4, emulated_uint64_t> morton_emulated_4_signed = createMortonFromU64Vec<true, fullBits_4, 4, emulated_uint64_t>(Vec4A);
	
		
		output.mortonUnsignedLess_emulated_4 = uint32_t4(morton_emulated_4A.lessThan<false>(Vec4BFull));
		
		output.mortonSignedLess_emulated_2 = uint32_t2(morton_emulated_2_signed.lessThan<false>(Vec2BSignedFull)); 
		output.mortonSignedLess_emulated_3 = uint32_t3(morton_emulated_3_signed.lessThan<false>(Vec3BSignedFull)); 
		output.mortonSignedLess_emulated_4 = uint32_t4(morton_emulated_4_signed.lessThan<false>(Vec4BSignedFull)); 
	
		uint16_t castedShift = uint16_t(input.shift);
	
		arithmetic_right_shift_operator<morton::code<true, fullBits_2, 2, emulated_uint64_t> > rightShiftSignedEmulated2;
		output.mortonSignedRightShift_emulated_2 = rightShiftSignedEmulated2(morton_emulated_2_signed, castedShift % fullBits_2); 
		arithmetic_right_shift_operator<morton::code<true, fullBits_3, 3, emulated_uint64_t> > rightShiftSignedEmulated3;
		output.mortonSignedRightShift_emulated_3 = rightShiftSignedEmulated3(morton_emulated_3_signed, castedShift % fullBits_3); 
		arithmetic_right_shift_operator<morton::code<true, fullBits_4, 4, emulated_uint64_t> > rightShiftSignedEmulated4;
		output.mortonSignedRightShift_emulated_4 = rightShiftSignedEmulated4(morton_emulated_4_signed, castedShift % fullBits_4);
	}
}
