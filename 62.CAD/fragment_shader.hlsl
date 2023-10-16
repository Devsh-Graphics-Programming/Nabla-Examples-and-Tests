#pragma shader_stage(fragment)

#include "common.hlsl"
#include <nbl/builtin/hlsl/shapes/rounded_line.hlsl>
#include <nbl/builtin/hlsl/shapes/beziers.hlsl>
#include <nbl/builtin/hlsl/algorithm.hlsl>

#if defined(NBL_FEATURE_FRAGMENT_SHADER_PIXEL_INTERLOCK)
[[vk::ext_instruction(/* OpBeginInvocationInterlockEXT */ 5364)]]
void beginInvocationInterlockEXT();
[[vk::ext_instruction(/* OpEndInvocationInterlockEXT */ 5365)]]
void endInvocationInterlockEXT();
#endif

// TODO[Lucas]: have a function for quadratic equation solving
// Write a general one, and maybe another one that uses precomputed values, and move these to somewhere nice in our builtin hlsl shaders if we don't have one
// See: https://github.com/erich666/GraphicsGems/blob/master/gems/Roots3And4.c

struct ArrayAccessor
{
    uint32_t styleIdx;
    using value_type = float;

    float operator[](const uint32_t ix)
    {
        return lineStyles[styleIdx].stipplePattern[ix];
    }
};

template<typename float_t, typename CurveType, typename StyleAccessor>
struct LineStyleClipper
{
    using float2_t = vector<float_t, 2>;

    static LineStyleClipper<float_t, CurveType, StyleAccessor> construct(StyleAccessor styleAccessor, 
                                                                         CurveType curve,
                                                                         typename CurveType::ArcLenCalculator arcLenCalc)
    {
        LineStyleClipper<float_t, CurveType, StyleAccessor> ret = { styleAccessor, curve, arcLenCalc };
        return ret;
    }
    
    float2_t operator()(float t)
    {
        const LineStyle style = lineStyles[styleAccessor.styleIdx];
        const float arcLen = arcLenCalc.calcArcLen(t);
        t = clamp(t, 0.0f, 1.0f);
        const float worldSpaceArcLen = arcLen * float(globals.worldToScreenRatio);
        float_t normalizedPlaceInPattern = frac(worldSpaceArcLen * style.reciprocalStipplePatternLen + style.phaseShift);
        uint32_t patternIdx = nbl::hlsl::upper_bound(styleAccessor, 0, style.stipplePatternSize, normalizedPlaceInPattern);
        
        // odd patternIdx means a "no draw section" and current candidate should split into two nearest draw sections
        if(patternIdx & 0x1)
        {   
            float diffToLeftDrawableSection = (patternIdx == 0) ? 0.0f : style.stipplePattern[patternIdx-1];
            float diffToRightDrawableSection = (patternIdx == style.stipplePatternSize) ? 1.0f : style.stipplePattern[patternIdx];
            diffToLeftDrawableSection -= normalizedPlaceInPattern;
            diffToRightDrawableSection -= normalizedPlaceInPattern;
            
            const float patternLen = float(globals.screenToWorldRatio) / style.reciprocalStipplePatternLen;
            float scrSpcOffsetToArcLen0 = diffToLeftDrawableSection * patternLen;
            float scrSpcOffsetToArcLen1 = diffToRightDrawableSection * patternLen;
            
            const float arcLenForT0 = arcLen + scrSpcOffsetToArcLen0;
            const float arcLenForT1 = arcLen + scrSpcOffsetToArcLen1;
            const float totalArcLen = arcLenCalc.calcArcLen(1.0f);

            // TODO: implement, for now code below creates artifacts for curvest that start or end with a "no draw section"
            //const float t0 = (0.0f <= arcLenForT0 && totalArcLen >= arcLenForT0) ? arcLenCalc.calcArcLenInverse(curve, arcLen + scrSpcOffsetToArcLen0, 0.000001f, t) : t;
            //const float t1 = (0.0f <= arcLenForT1 && totalArcLen >= arcLenForT1) ? arcLenCalc.calcArcLenInverse(curve, arcLen + scrSpcOffsetToArcLen1, 0.000001f, t) : t;
            
            const float t0 = arcLenCalc.calcArcLenInverse(curve, arcLenForT0, 0.000001f, t);
            const float t1 = arcLenCalc.calcArcLenInverse(curve, arcLenForT1, 0.000001f, t);

            return float2(t0, t1);
        }
        else
        {
            return t.xx;
        }
    }
    
    StyleAccessor styleAccessor;
    CurveType curve;
    typename CurveType::ArcLenCalculator arcLenCalc;
};

typedef LineStyleClipper<float, nbl::hlsl::shapes::Quadratic<float>, ArrayAccessor > BezierLineStyleClipper_float;

float4 main(PSInput input) : SV_TARGET
{
#if defined(NBL_FEATURE_FRAGMENT_SHADER_PIXEL_INTERLOCK)
    [[vk::ext_capability(/*FragmentShaderPixelInterlockEXT*/ 5378)]]
    [[vk::ext_extension("SPV_EXT_fragment_shader_interlock")]]
    vk::ext_execution_mode(/*PixelInterlockOrderedEXT*/ 5366);
#endif

    ObjectType objType = input.getObjType();
    float localAlpha = 0.0f;
    uint32_t currentMainObjectIdx = input.getMainObjectIdx();
    
    if (objType == ObjectType::LINE)
    {
        const float2 start = input.getLineStart();
        const float2 end = input.getLineEnd();
        const float lineThickness = input.getLineThickness();

        float distance = nbl::hlsl::shapes::RoundedLine_t<float>::construct(start, end).signedDistance(input.position.xy, lineThickness);

        const float antiAliasingFactor = globals.antiAliasingFactor;
        localAlpha = 1.0f - smoothstep(-antiAliasingFactor, +antiAliasingFactor, distance);
    }
    else if (objType == ObjectType::QUAD_BEZIER)
    {
        nbl::hlsl::shapes::Quadratic<float> quadratic = input.getQuadratic();
        nbl::hlsl::shapes::Quadratic<float>::ArcLenCalculator arcLenCalc = input.getQuadraticArcLenCalculator();
        
        const uint32_t styleIdx = mainObjects[currentMainObjectIdx].styleIdx;
        const float lineThickness = input.getLineThickness();
        float distance;
        
        if (!lineStyles[styleIdx].hasStipples())
        {
            distance = quadratic.signedDistance(input.position.xy, lineThickness);
        }
        else
        {
            const float lineThickness = input.getLineThickness();
            ArrayAccessor arrayAccessor;
            BezierLineStyleClipper_float clipper = BezierLineStyleClipper_float::construct(arrayAccessor, quadratic, arcLenCalc);
            
            distance = quadratic.signedDistance(input.position.xy, lineThickness, clipper);
        }
        
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

    const uint32_t packedData = pseudoStencil[fragCoord];

    const uint32_t localQuantizedAlpha = (uint32_t)(localAlpha*255.f);
    const uint32_t quantizedAlpha = bitfieldExtract(packedData,0,AlphaBits);
    // if geomID has changed, we resolve the SDF alpha (draw using blend), else accumulate
    const uint32_t mainObjectIdx = bitfieldExtract(packedData,AlphaBits,MainObjectIdxBits);
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