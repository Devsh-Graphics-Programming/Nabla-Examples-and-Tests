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

#define NBL_DRAW_ARC_LENGTH

// TODO[Lucas]: have a function for quadratic equation solving
// Write a general one, and maybe another one that uses precomputed values, and move these to somewhere nice in our builtin hlsl shaders if we don't have one
// See: https://github.com/erich666/GraphicsGems/blob/master/gems/Roots3And4.c

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
        float distance = nbl::hlsl::shapes::QuadraticBezierOutline<float>::construct(a, b, c, lineThickness).signedDistance(input.position.xy);

        const float antiAliasingFactor = globals.antiAliasingFactor;
        localAlpha = 1.0f - smoothstep(-antiAliasingFactor, +antiAliasingFactor, distance);
    }
    /*
    TODO[Lucas]:
        Another else case for CurveBox where you simply do what I said in the notes of common.hlsl PSInput
        and solve two quadratic equations, you could check for it being a "line" for the mid point being nan
        you will use input.getXXX() to get values needed for this computation
    */

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
#elif defined(NBL_DRAW_ARC_LENGTH)
    //idk if it is right place to do that.. shout i put that code in another file?
    const float2 P0 = input.getBezierP0();
    const float2 P1 = input.getBezierP1();
    const float2 P2 = input.getBezierP2();
    const float lineThickness = input.getLineThickness();
    nbl::hlsl::shapes::QuadraticBezierOutline<float> curveOutline = nbl::hlsl::shapes::QuadraticBezierOutline<float>::construct(P0, P1, P2, lineThickness);

    float tA = curveOutline.ud(input.position.xy).y;

    float bezierCurveArcLen = curveOutline.bezier.calcArcLen(1.0);
    float arcLen = curveOutline.bezier.calcArcLen(tA);

    //float resultColorIntensity = arcLen / bezierCurveArcLen;
    float resultColorIntensity = curveOutline.bezier.calcArcLenInverse(arcLen, 0.000001, arcLen / bezierCurveArcLen);

    col = float4(0.0, resultColorIntensity, 0.0, 1.0);
#else
    col = input.getColor();
    col.w *= localAlpha;
#endif

    return float4(col);
}