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

#define NBL_DRAW_ARC_LENGTH

// TODO[Lucas]: have a function for quadratic equation solving
// Write a general one, and maybe another one that uses precomputed values, and move these to somewhere nice in our builtin hlsl shaders if we don't have one
// See: https://github.com/erich666/GraphicsGems/blob/master/gems/Roots3And4.c

template<typename float_t>
struct BezierLineStyleClipper
{
    using float2_t = vector<float_t, 2>;

    static BezierLineStyleClipper<float_t> construct(uint32_t styleIdx)
    {
        BezierLineStyleClipper<float_t> ret = { styleIdx };
        return ret;
    }
    
    struct ArrayAccessor
    {
        uint32_t styleIdx;
        using value_type = float;

        float operator[](const uint32_t ix)
        {
            return lineStyles[styleIdx].stipplePattern[ix];
        }
    };
                                              // kinda dumb to pass quadratic here, will think of another solution
    inline float2_t operator()(const float t, typename nbl::hlsl::shapes::Quadratic<float_t> quadratic, typename nbl::hlsl::shapes::Quadratic<float_t>::ArcLengthPrecomputedValues preCompValues)
    {
        const float arcLen = quadratic.calcArcLen(t, preCompValues);
        float_t tMappedToPattern = frac(arcLen / float(globals.screenToWorldRatio) * lineStyles[styleIdx].recpiprocalStipplePatternLen + lineStyles[styleIdx].phaseShift);
        ArrayAccessor stippleAccessor = { styleIdx };
        uint32_t patternIdx = nbl::hlsl::upper_bound(stippleAccessor, 0, lineStyles[styleIdx].stipplePatternSize, tMappedToPattern);
        
        if(patternIdx & 0x1)
        {   
            float t0NormalizedLen = (patternIdx == 0) ? 1.0 : lineStyles[styleIdx].stipplePattern[patternIdx-1];
            float t1NormalizedLen = (patternIdx == lineStyles[styleIdx].stipplePatternSize) ? 1.0 : lineStyles[styleIdx].stipplePattern[patternIdx];
            t0NormalizedLen -= tMappedToPattern;
            t1NormalizedLen -= tMappedToPattern;
            
            // TODO: move to globals
            const float worldToScreenRatio = 1.0f/globals.screenToWorldRatio;
            
            float t0 = t0NormalizedLen / worldToScreenRatio / lineStyles[styleIdx].recpiprocalStipplePatternLen;
            float t1 = t1NormalizedLen / worldToScreenRatio / lineStyles[styleIdx].recpiprocalStipplePatternLen;
            
            t0 = quadratic.calcArcLenInverse(arcLen + t0, 0.000001f, 0.5f, preCompValues);
            t1 = quadratic.calcArcLenInverse(arcLen + t1, 0.000001f, 0.5f, preCompValues);
            
            t0 = clamp(t0, 0.0, 1.0);
            t1 = clamp(t1, 0.0, 1.0);
            
            return float2(t0, t1);
        }
        else
            return clamp(t, 0.0, 1.0).xx;
    }
  
    uint32_t styleIdx;
};

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

        float distance = nbl::hlsl::shapes::RoundedLine_t::construct(start, end, lineThickness).signedDistance(input.position.xy);

        const float antiAliasingFactor = globals.antiAliasingFactor;
        localAlpha = 1.0f - smoothstep(-antiAliasingFactor, +antiAliasingFactor, distance);
    }
    else if (objType == ObjectType::QUAD_BEZIER)
    {
#define NBL_DRAW_STIPPLE_PATTERN_LINES
#ifdef NBL_DRAW_STIPPLE_PATTERN_LINES
        QuadBezierAnalyticArcLengthCalculator<float> preCompValues_calculator = input.getPrecomputedArcLenData();
        nbl::hlsl::shapes::Quadratic<float>::ArcLengthPrecomputedValues preCompValues;
        preCompValues.lenA2 = preCompValues_calculator.lenA2;
        preCompValues.AdotB = preCompValues_calculator.AdotB;
        preCompValues.a = preCompValues_calculator.a;
        preCompValues.b = preCompValues_calculator.b;
        preCompValues.c = preCompValues_calculator.c;
        preCompValues.b_over_4a = preCompValues_calculator.b_over_4a;
        
        const float lineThickness = input.getLineThickness();
        BezierLineStyleClipper<float> clipper = BezierLineStyleClipper<float>::construct(mainObjects[currentMainObjectIdx].styleIdx);
        float distance = input.getQuadratic().signedDistance2(input.position.xy, lineThickness, preCompValues, clipper);

        const float antiAliasingFactor = globals.antiAliasingFactor;
        localAlpha = 1.0f - smoothstep(-antiAliasingFactor, +antiAliasingFactor, distance);
#else
        const float lineThickness = input.getLineThickness();
        float distance = input.getQuadratic().signedDistance(input.position.xy, lineThickness);

        const float antiAliasingFactor = globals.antiAliasingFactor;
        localAlpha = 1.0f - smoothstep(-antiAliasingFactor, +antiAliasingFactor, distance);
#endif

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
    
    nbl::hlsl::shapes::Quadratic<float> quadratic = input.getQuadratic();
    QuadBezierAnalyticArcLengthCalculator<float> preCompValues_calculator = input.getPrecomputedArcLenData();
    nbl::hlsl::shapes::Quadratic<float>::ArcLengthPrecomputedValues preCompValues;
    preCompValues.lenA2 = preCompValues_calculator.lenA2;
    preCompValues.AdotB = preCompValues_calculator.AdotB;
    preCompValues.a = preCompValues_calculator.a;
    preCompValues.b = preCompValues_calculator.b;
    preCompValues.c = preCompValues_calculator.c;
    preCompValues.b_over_4a = preCompValues_calculator.b_over_4a;
    
    float tA = quadratic.ud(input.position.xy).y;

    float bezierCurveArcLen = quadratic.calcArcLen(1.0, preCompValues);
    float arcLen = quadratic.calcArcLen(tA, preCompValues);
    
    float resultColorIntensity = quadratic.calcArcLenInverse(arcLen, 0.000001f, arcLen / bezierCurveArcLen, preCompValues);

    col = float4(0.0f, resultColorIntensity, 0.0f, 1.0f);
    
#else
    col = input.getColor();
    col.w *= localAlpha;
#endif

    return float4(col);
}