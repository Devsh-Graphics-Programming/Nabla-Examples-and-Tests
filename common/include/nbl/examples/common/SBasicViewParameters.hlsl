#ifndef _NBL_EXAMPLES_S_BASIC_VIEW_PARAMETERS_HLSL_
#define _NBL_EXAMPLES_S_BASIC_VIEW_PARAMETERS_HLSL_


#include "nbl/builtin/hlsl/cpp_compat/matrix.hlsl"


namespace nbl
{
namespace hlsl
{
namespace examples
{

struct SBasicViewParameters
{
	float32_t4x4 MVP;
	float32_t3x4 MV;
	float32_t3x3 normalMat;
};

}
}
}
#endif

/*
	do not remove this text, WAVE is so bad that you can get errors if no proper ending xD
*/