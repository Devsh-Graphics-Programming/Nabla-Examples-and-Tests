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

template<typename float_t, typename CurveType>
struct LineStyleClipper
{
    using float2_t = vector<float_t, 2>;

    static LineStyleClipper<float_t, CurveType> construct(uint32_t styleIdx, 
                                                          CurveType curve,
                                                          typename CurveType::ArcLenCalculator arcLenCalc)
    {
        LineStyleClipper<float_t, CurveType> ret = { styleIdx, curve, arcLenCalc };
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
    
    float2_t operator()(const float t)
    {
        const float arcLen = arcLenCalc.calcArcLen(t);
        float_t tMappedToPattern = frac(arcLen / float(globals.screenToWorldRatio) * lineStyles[styleIdx].recpiprocalStipplePatternLen + lineStyles[styleIdx].phaseShift);
        ArrayAccessor stippleAccessor = { styleIdx };
        uint32_t patternIdx = nbl::hlsl::upper_bound(stippleAccessor, 0, lineStyles[styleIdx].stipplePatternSize, tMappedToPattern);
        
        if(patternIdx & 0x1)
        {   
            float t0NormalizedLen = (patternIdx == 0) ? 1.0 : lineStyles[styleIdx].stipplePattern[patternIdx-1];
            float t1NormalizedLen = (patternIdx == lineStyles[styleIdx].stipplePatternSize) ? 1.0 : lineStyles[styleIdx].stipplePattern[patternIdx];
            t0NormalizedLen -= tMappedToPattern;
            t1NormalizedLen -= tMappedToPattern;
            
            float t0 = t0NormalizedLen / globals.worldToScreenRatio / lineStyles[styleIdx].recpiprocalStipplePatternLen;
            float t1 = t1NormalizedLen / globals.worldToScreenRatio / lineStyles[styleIdx].recpiprocalStipplePatternLen;
            
            t0 = arcLenCalc.calcArcLenInverse(curve, arcLen + t0, 0.000001f, 0.5f);
            t1 = arcLenCalc.calcArcLenInverse(curve, arcLen + t1, 0.000001f, 0.5f);
            
            t0 = clamp(t0, 0.0, 1.0);
            t1 = clamp(t1, 0.0, 1.0);
            
            return float2(t0, t1);
        }
        else
            return clamp(t, 0.0, 1.0).xx;
    }
    
    uint32_t styleIdx;
    CurveType curve;
    typename CurveType::ArcLenCalculator arcLenCalc;
};

typedef LineStyleClipper<float, nbl::hlsl::shapes::Quadratic<float> > BezierLineStyleClipper_float;

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
            BezierLineStyleClipper_float clipper = BezierLineStyleClipper_float::construct(styleIdx, quadratic, arcLenCalc);
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
#elif defined(NBL_DRAW_ARC_LENGTH)
    nbl::hlsl::shapes::Quadratic<float> quadratic = input.getQuadratic();
    nbl::hlsl::shapes::Quadratic<float>::ArcLenCalculator arcLenCalc = input.getQuadraticArcLenCalculator();
    
    const uint32_t styleIdx = mainObjects[currentMainObjectIdx].styleIdx;
    //BezierLineStyleClipper_float clipper = BezierLineStyleClipper_float::construct(styleIdx, quadratic, arcLenCalc);
    nbl::hlsl::shapes::Quadratic<float>::DefaultClipper clipper = nbl::hlsl::shapes::Quadratic<float>::DefaultClipper::construct();
    const float lineThickness = input.getLineThickness();
    
    float tA = quadratic.ud(input.position.xy, lineThickness, clipper).y;

    float bezierCurveArcLen = arcLenCalc.calcArcLen(1.0);
    float arcLen = arcLenCalc.calcArcLen(tA);
    
    float resultColorIntensity = abs(arcLenCalc.calcArcLenInverse(quadratic, arcLen, 0.000001f, arcLen / bezierCurveArcLen) - tA);
    //float resultColorIntensity = tA;
    //float resultColorIntensity = abs(arcLen / bezierCurveArcLen - tA);
    
    if(isnan(resultColorIntensity))
        resultColorIntensity = 1.0f;
        
    if(resultColorIntensity > exp2(-23))
        resultColorIntensity = 1.0f;
        
    col = float4(resultColorIntensity, 0.0f, 0.0f, 1.0f);
    //col.w *= localAlpha;
    
    //col = float4(arcLen, bezierCurveArcLen, 0.0f, 1.0f);
    
    float lenTan = sqrt(tA*(arcLenCalc.a*tA + arcLenCalc.b) + arcLenCalc.c);
    float logTermA = arcLenCalc.b + 2.0f * sqrt(arcLenCalc.a) * sqrt(arcLenCalc.c);
    float logTermB = arcLenCalc.b + 2.0f * arcLenCalc.a * tA + 2.0f * sqrt(arcLenCalc.a) * lenTan;
    
    //col = float4(tA * (arcLenCalc.a * tA + arcLenCalc.b) + arcLenCalc.c, arcLen, sqrt(arcLenCalc.c), 1.0);
    //col = float4(logTermA, logTermB, arcLen, 1.0);
    //col = float4(tA, 0.0f, 0.0f, 1.0f);
    //col = float4(arcLenCalc.a, arcLenCalc.b, arcLenCalc.c, arcLen);
    //col = float4(tA, 0.0f, 0.0f, 1.0f);
    
#else
    const float2 a = input.getBezierP0();
    const float2 b = input.getBezierP1();
    const float2 c = input.getBezierP2();
    const float lineThickness = input.getLineThickness();

    col = float4(0.0f, nbl::hlsl::shapes::QuadraticBezierOutline::construct(a, b, c, lineThickness).ud(input.position.xy).y, 0.0f, 1.0f);
    col.w *= localAlpha;
#endif

    return float4(col);
}