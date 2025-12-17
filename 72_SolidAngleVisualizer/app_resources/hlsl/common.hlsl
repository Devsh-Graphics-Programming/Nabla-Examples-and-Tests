#ifndef _SOLID_ANGLE_VIS_COMMON_HLSL_
#define _SOLID_ANGLE_VIS_COMMON_HLSL_
#include "nbl/builtin/hlsl/cpp_compat.hlsl"

#define DEBUG_DATA 1

namespace nbl
{
    namespace hlsl
    {

        struct ResultData
        {
            uint32_t3 region;
            uint32_t silhouetteIndex;
            
            uint32_t silhouetteVertexCount;
            uint32_t silhouette;
            uint32_t clippedVertexCount;
            uint32_t edgeVisibilityMismatch;

            uint32_t vertices[6];
        };

        struct PushConstants
        {
            float32_t3x4 modelMatrix;
            float32_t4 viewport;
        };

        static const float32_t3 colorLUT[27] = {
            float32_t3(0, 0, 0), 		float32_t3(1, 1, 1), 		float32_t3(0.5, 0.5, 0.5),
            float32_t3(1, 0, 0), 		float32_t3(0, 1, 0), 		float32_t3(0, 0, 1),
            float32_t3(1, 1, 0), 		float32_t3(1, 0, 1), 		float32_t3(0, 1, 1),
            float32_t3(1, 0.5, 0), 		float32_t3(1, 0.65, 0), 	float32_t3(0.8, 0.4, 0),
            float32_t3(1, 0.4, 0.7), 	float32_t3(1, 0.75, 0.8), 	float32_t3(0.7, 0.1, 0.3),
            float32_t3(0.5, 0, 0.5), 	float32_t3(0.6, 0.4, 0.8), 	float32_t3(0.3, 0, 0.5),
            float32_t3(0, 0.5, 0), 		float32_t3(0.5, 1, 0), 		float32_t3(0, 0.5, 0.25),
            float32_t3(0, 0, 0.5), 		float32_t3(0.3, 0.7, 1), 	float32_t3(0, 0.4, 0.6),
            float32_t3(0.6, 0.4, 0.2), 	float32_t3(0.8, 0.7, 0.3), 	float32_t3(0.4, 0.3, 0.1)
        };

#ifndef __HLSL_VERSION
		static const char* colorNames[27] = {"Black",
			"White", "Gray", "Red", "Green", "Blue", "Yellow", "Magenta", "Cyan",
			"Orange", "Light Orange", "Dark Orange", "Pink", "Light Pink", "Deep Rose", "Purple", "Light Purple",
			"Indigo", "Dark Green", "Lime", "Forest Green", "Navy", "Sky Blue", "Teal", "Brown",
			"Tan/Beige", "Dark Brown"
		};
#endif // __HLSL_VERSION
    }
}
#endif // _SOLID_ANGLE_VIS_COMMON_HLSL_
