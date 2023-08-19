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

float4 main(PSInput input) : SV_TARGET
{
#if defined(NBL_FEATURE_FRAGMENT_SHADER_PIXEL_INTERLOCK)
    [[vk::ext_capability(/*FragmentShaderPixelInterlockEXT*/ 5378)]]
    [[vk::ext_extension("SPV_EXT_fragment_shader_interlock")]]
    vk::ext_execution_mode(/*PixelInterlockOrderedEXT*/ 5366);
#endif
	
    // TODO is this going to interact badly with the pixel interlock?	
    if (globals.clipEnabled != 0 && (any(input.position.xy < globals.clip.xy) || any(input.position.xy > globals.clip.zw))) discard;
    
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
        float2 positionFullscreen = input.uv;

        nbl::hlsl::shapes::QuadraticBezier curveMin = nbl::hlsl::shapes::QuadraticBezier::construct(
            input.getCurveMinP0(),
            input.getCurveMinP1(),
            input.getCurveMinP2()
        );
        nbl::hlsl::shapes::QuadraticBezier curveMax = nbl::hlsl::shapes::QuadraticBezier::construct(
            input.getCurveMaxP0(),
            input.getCurveMaxP1(),
            input.getCurveMaxP2()
        );
        const uint major = 1;

        uint minor = 1-major;
        float minT = curveMin.tForMajorCoordinate(major, positionFullscreen[major]);
        float minEv = curveMin.evaluate(minT)[minor];
        float maxT = curveMax.tForMajorCoordinate(major, positionFullscreen[major]);
        float maxEv = curveMax.evaluate(maxT)[minor];
        
        float4 col = input.getColor();
        const float antiAliasingFactor = globals.antiAliasingFactor;
        float distance = min(positionFullscreen[minor] - minEv, maxEv - positionFullscreen[minor]);
        if (distance >= 0)
        {
            localAlpha = 1.0f - smoothstep(-antiAliasingFactor, +antiAliasingFactor, distance); //1.0;
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
