#pragma shader_stage(fragment)

#include "common.hlsl"
#include <nbl/builtin/hlsl/shapes/rounded_line.hlsl>
#include <nbl/builtin/hlsl/shapes/beziers.hlsl>

#if defined(NBL_FEATURE_FRAGMENT_SHADER_PIXEL_INTERLOCK)
[[vk::ext_instruction(/* OpBeginInvocationInterlockEXT */ 5364)]]
void beginInvocationInterlockEXT();
[[vk::ext_instruction(/* OpEndInvocationInterlockEXT */ 5365)]]
void endInvocationInterlockEXT();
#endif

// We're using the precomputed values to solve a quadratic equation of intersecting a bezier with a line perpendicular to major axis.
// (More details in the common.hlsl)
float tForMajorCoordinate(float detRcp2, float bRcp) 
{ 
    float detSqrt = sqrt(detRcp2);
    float2 roots = float2(-detSqrt,detSqrt)-bRcp;
    // assert(roots.x == roots.y);
    // assert(!isnan(roots.x));
    return roots.x;
}

float evaluateBezier(float A, float B, float C, float t) 
{ 
    return A * A * t + B * t + C;
}

float4 main(PSInput input) : SV_TARGET
{
#if defined(NBL_FEATURE_FRAGMENT_SHADER_PIXEL_INTERLOCK)
    [[vk::ext_capability(/*FragmentShaderPixelInterlockEXT*/ 5378)]]
    [[vk::ext_extension("SPV_EXT_fragment_shader_interlock")]]
    vk::ext_execution_mode(/*PixelInterlockOrderedEXT*/ 5366);
#endif
	
    ObjectType objType = input.getObjType();
    float localAlpha = 0.0f;
    uint currentMainObjectIdx = input.getMainObjectIdx();
    
    if (objType == ObjectType::LINE)
    {
        const float2 start = input.getLineStart();
        const float2 end = input.getLineEnd();
        const float lineThickness = input.getLineThickness();

        float distance = nbl::hlsl::shapes::RoundedLine_t::construct(start, end, lineThickness).signedDistance(input.position.xy);

        const float antiAliasingFactor = globals.antiAliasingFactor;
        localAlpha = 1.0f - smoothstep(-antiAliasingFactor, +antiAliasingFactor, distance);
    }
    else if (objType == ObjectType::QUAD_BEZIER)
    {
        const float2 a = input.getBezierP0();
        const float2 b = input.getBezierP1();
        const float2 c = input.getBezierP2();
        const float lineThickness = input.getLineThickness();
        
        // TODO[Przemek]: This is where we draw the bezier using the sdf, basically the udBezier funcion in that shaderToy we gave you
        // You'll be also working in the builtin shaders that provide thesee
        float distance = nbl::hlsl::shapes::QuadraticBezierOutline::construct(a, b, c, lineThickness).signedDistance(input.position.xy);

        const float antiAliasingFactor = globals.antiAliasingFactor;
        localAlpha = 1.0f - smoothstep(-antiAliasingFactor, +antiAliasingFactor, distance);
    }
    else if (objType == ObjectType::CURVE_BOX) 
    {
        float uv = input.getUVMinor();

        float minT = tForMajorCoordinate(input.getMinCurveDetRcp2(), input.getMinCurveBrcp());
        float minEv = evaluateBezier(input.getCurveMinA().y, input.getCurveMinB().y, input.getCurveMinC().y, minT);
        
        float maxT = tForMajorCoordinate(input.getMaxCurveDetRcp2(), input.getMaxCurveBrcp());
        float maxEv = evaluateBezier(input.getCurveMaxA().y, input.getCurveMaxB().y, input.getCurveMaxC().y, maxT);
        
        float4 col = input.getColor();
        const float antiAliasingFactor = globals.antiAliasingFactor;
        float distance = min(uv - minEv, maxEv - uv);

        // TODO: AA using ddx ddy
        if (distance >= 0)
        {
            localAlpha = 1.0;
        }

        col.w *= localAlpha;
        return float4(col);
    }

    uint2 fragCoord = uint2(input.position.xy);
    float4 col;
    
    if (localAlpha <= 0)
        discard;
    
#if defined(NBL_FEATURE_FRAGMENT_SHADER_PIXEL_INTERLOCK)
        beginInvocationInterlockEXT();

    const uint packedData = pseudoStencil[fragCoord];

    const uint localQuantizedAlpha = (uint)(localAlpha*255.f);
    const uint quantizedAlpha = bitfieldExtract(packedData,0,AlphaBits);
    // if geomID has changed, we resolve the SDF alpha (draw using blend), else accumulate
    const uint mainObjectIdx = bitfieldExtract(packedData,AlphaBits,MainObjectIdxBits);
    const bool resolve = currentMainObjectIdx!=mainObjectIdx;
    if (resolve || localQuantizedAlpha>quantizedAlpha)
        pseudoStencil[fragCoord] = bitfieldInsert(localQuantizedAlpha,currentMainObjectIdx,AlphaBits,MainObjectIdxBits);

    endInvocationInterlockEXT();
    
    if (!resolve)
        discard;
    
    // draw with previous geometry's style's color :kek:
    col = lineStyles[mainObjects[mainObjectIdx].styleIdx].color;
    col.w *= float(quantizedAlpha)/255.f;
#else
    col = input.getColor();
    col.w *= localAlpha;
#endif
    
    return float4(col);
}
