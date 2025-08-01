#ifndef _CAD_EXAMPLE_MAIN_PIPELINE_COMMON_HLSL_INCLUDED_
#define _CAD_EXAMPLE_MAIN_PIPELINE_COMMON_HLSL_INCLUDED_

#include "../globals.hlsl"

// This function soley exists to match n4ce's behaviour, colors and color operations for DTMs, Curves, Lines, Hatches are done in linear space and then outputted to linear surface (as if surface had UNORM format, but ours is SRGB)
// We should do gamma "uncorrection" to account for the fact that our surface format is SRGB and will do gamma correction
void gammaUncorrect(inout float3 col)
{
    bool outputToSRGB = true; // TODO
    float gamma = (outputToSRGB) ? 2.2f : 1.0f;
    col.rgb = pow(col.rgb, gamma);
}

// TODO: Use these in C++ as well once numeric_limits<uint32_t> compiles on C++
float32_t2 unpackCurveBoxUnorm(uint32_t2 value)
{
    return float32_t2(value) / float32_t(numeric_limits<uint32_t>::max);
}

float32_t2 unpackCurveBoxSnorm(int32_t2 value)
{
    return float32_t2(value) / float32_t(numeric_limits<int32_t>::max);
}


uint32_t2 packCurveBoxUnorm(float32_t2 value)
{
    return value * float32_t(numeric_limits<uint32_t>::max);
}

int32_t2 packCurveBoxSnorm(float32_t2 value)
{
    return value * float32_t(numeric_limits<int32_t>::max);
}

// The root we're always looking for:
// 2 * C / (-B - detSqrt)
// We send to the FS: -B, 2C, det
template<typename float_t>
struct PrecomputedRootFinder 
{
    using float_t2 = vector<float_t, 2>;
    using float_t3 = vector<float_t, 3>;
    
    float_t C2;
    float_t negB;
    float_t det;

    float_t computeRoots() 
    {
        return C2 / (negB - sqrt(det));
    }

    static PrecomputedRootFinder construct(float_t negB, float_t C2, float_t det)
    {
        PrecomputedRootFinder result;
        result.C2 = C2;
        result.det = det;
        result.negB = negB;
        return result;
    }

    static PrecomputedRootFinder construct(math::equations::Quadratic<float_t> quadratic)
    {
        PrecomputedRootFinder result;
        result.C2 = quadratic.c * 2.0;
        result.negB = -quadratic.b;
        result.det = quadratic.b * quadratic.b - 4.0 * quadratic.a * quadratic.c;
        return result;
    }
};

 // TODO[Przemek]: your triangle mesh passed parameters from vtx to fragment will need to be here (e.g. height).
 //     As always try to reuse parameters and try not to introduce new ones
struct PSInput
{
    [[vk::location(0)]] float4 position : SV_Position;
    [[vk::location(1)]] float4 clip : SV_ClipDistance;
    
    [[vk::location(2)]] nointerpolation uint4 data1 : COLOR1;
    [[vk::location(3)]] nointerpolation float4 data2 : COLOR2;
    [[vk::location(4)]] nointerpolation float4 data3 : COLOR3;
    [[vk::location(5)]] nointerpolation float4 data4 : COLOR4;
    // Data segments that need interpolation, mostly for hatches
    [[vk::location(6)]] float4 interp_data5 : COLOR5;
    [[vk::location(7)]] nointerpolation float data6 : COLOR6;
    
#ifdef FRAGMENT_SHADER_INPUT
    [[vk::location(8)]] [[vk::ext_decorate(/*spv::DecoratePerVertexKHR*/5285)]] float3 vertexScreenSpacePos[3] : COLOR7;
#else
    [[vk::location(8)]] float3 vertexScreenSpacePos : COLOR7;
#endif
    // ArcLenCalculator<float>

    // Set functions used in vshader, get functions used in fshader
    // We have to do this because we don't have union in hlsl and this is the best way to alias
    
    /* SHARED: ALL ObjectTypes */
    ObjectType getObjType() { return (ObjectType) data1.x; }
    uint getMainObjectIdx() { return data1.y; }
    
    void setObjType(ObjectType objType) { data1.x = (uint) objType; }
    void setMainObjectIdx(uint mainObjIdx) { data1.y = mainObjIdx; }
    
    /* SHARED: LINE + QUAD_BEZIER (Curve Outlines) */
    float getLineThickness() { return asfloat(data1.z); }
    float getPatternStretch() { return asfloat(data1.w); }

    void setLineThickness(float lineThickness) { data1.z = asuint(lineThickness); }
    void setPatternStretch(float stretch) { data1.w = asuint(stretch); }

    void setCurrentPhaseShift(float phaseShift)  { interp_data5.x = phaseShift; }
    float getCurrentPhaseShift() { return interp_data5.x; }

    /* LINE */
    float2 getLineStart() { return data2.xy; }
    float2 getLineEnd() { return data2.zw; }
    void setLineStart(float2 lineStart) { data2.xy = lineStart; }
    void setLineEnd(float2 lineEnd) { data2.zw = lineEnd; }
    
    /* QUAD_BEZIER */
    shapes::Quadratic<float> getQuadratic()
    {
        return shapes::Quadratic<float>::construct(data2.xy, data2.zw, data3.xy);
    }
    void setQuadratic(shapes::Quadratic<float> quadratic)
    {
        data2.xy = quadratic.A;
        data2.zw = quadratic.B;
        data3.xy = quadratic.C;
    }
    
    void setQuadraticPrecomputedArcLenData(shapes::Quadratic<float>::ArcLengthCalculator preCompData) 
    {
        data3.zw = float2(preCompData.lenA2, preCompData.AdotB);
        data4 = float4(preCompData.a, preCompData.b, preCompData.c, preCompData.b_over_4a);
    }
    shapes::Quadratic<float>::ArcLengthCalculator getQuadraticArcLengthCalculator()
    {
        return shapes::Quadratic<float>::ArcLengthCalculator::construct(data3.z, data3.w, data4.x, data4.y, data4.z, data4.w);
    }
    
    /* CURVE_BOX */
    // Curves are split in the vertex shader based on their tmin and tmax
    // Min curve is smaller in the minor coordinate (e.g. in the default of y top to bottom sweep,
    // curveMin = smaller x / left, curveMax = bigger x / right)
    // TODO: possible optimization: passing precomputed values for solving the quadratic equation instead

    // data2, data3, data4
    math::equations::Quadratic<float> getCurveMinMinor() {
        return math::equations::Quadratic<float>::construct(data2.x, data2.y, data2.z);
    }
    math::equations::Quadratic<float> getCurveMaxMinor() {
        return math::equations::Quadratic<float>::construct(data2.w, data3.x, data3.y);
    }

    void setCurveMinMinor(math::equations::Quadratic<float> bezier) {
        data2.x = bezier.a;
        data2.y = bezier.b;
        data2.z = bezier.c;
    }
    void setCurveMaxMinor(math::equations::Quadratic<float> bezier) {
        data2.w = bezier.a;
        data3.x = bezier.b;
        data3.y = bezier.c;
    }

    // data4
    math::equations::Quadratic<float> getCurveMinMajor() {
        return math::equations::Quadratic<float>::construct(data4.x, data4.y, data3.z);
    }
    math::equations::Quadratic<float> getCurveMaxMajor() {
        return math::equations::Quadratic<float>::construct(data4.z, data4.w, data3.w);
    }

    void setCurveMinMajor(math::equations::Quadratic<float> bezier) {
        data4.x = bezier.a;
        data4.y = bezier.b;
        data3.z = bezier.c;
    }
    void setCurveMaxMajor(math::equations::Quadratic<float> bezier) {
        data4.z = bezier.a;
        data4.w = bezier.b;
        data3.w = bezier.c;
    }

    // Curve box value along minor & major axis
    float getMinorBBoxUV() { return interp_data5.x; };
    void setMinorBBoxUV(float minorBBoxUV) { interp_data5.x = minorBBoxUV; }
    float getMajorBBoxUV() { return interp_data5.y; };
    void setMajorBBoxUV(float majorBBoxUV) { interp_data5.y = majorBBoxUV; }

    float2 getCurveBoxScreenSpaceSize() { return asfloat(data1.zw); }
    void setCurveBoxScreenSpaceSize(float2 aabbSize) { data1.zw = asuint(aabbSize); }
    
    /* POLYLINE_CONNECTOR */
    void setPolylineConnectorTrapezoidStart(float2 trapezoidStart) { data2.xy = trapezoidStart; }
    void setPolylineConnectorTrapezoidEnd(float2 trapezoidEnd) { data2.zw = trapezoidEnd; }
    void setPolylineConnectorTrapezoidShortBase(float shortBase) { data3.x = shortBase; }
    void setPolylineConnectorTrapezoidLongBase(float longBase) { data3.y = longBase; }
    void setPolylineConnectorCircleCenter(float2 C) { data3.zw = C; }

    float2 getPolylineConnectorTrapezoidStart() { return data2.xy; }
    float2 getPolylineConnectorTrapezoidEnd() { return data2.zw; }
    float getPolylineConnectorTrapezoidShortBase() { return data3.x; }
    float getPolylineConnectorTrapezoidLongBase() { return data3.y; }
    float2 getPolylineConnectorCircleCenter() { return data3.zw; }
    
    /* FONT_GLYPH */
    float2 getFontGlyphUV() { return interp_data5.xy; }
    uint32_t getFontGlyphTextureId() { return asuint(data2.x); }
    float getFontGlyphPxRange() { return data2.y; }

    void setFontGlyphUV(float2 uv) { interp_data5.xy = uv; }
    void setFontGlyphTextureId(uint32_t textureId) { data2.x = asfloat(textureId); }
    void setFontGlyphPxRange(float glyphPxRange) { data2.y = glyphPxRange; }

    /* IMAGE */
    float2 getImageUV() { return interp_data5.xy; }
    uint32_t getImageTextureId() { return asuint(data2.x); }
    
    void setImageUV(float2 uv) { interp_data5.xy = uv; }
    void setImageTextureId(uint32_t textureId) { data2.x = asfloat(textureId); }

    /* TRIANGLE MESH */
    
#ifndef FRAGMENT_SHADER_INPUT // vertex shader
    void setScreenSpaceVertexAttribs(float3 pos) { vertexScreenSpacePos = pos; }
#else // fragment shader
    float3 getScreenSpaceVertexAttribs(uint32_t vertexIndex) { return vertexScreenSpacePos[vertexIndex]; }
#endif

    /* GRID DTM */
    uint getGridDTMHeightTextureID() { return data1.z; }
    float2 getGridDTMScreenSpaceGridExtents() { return data2.xy; }
    float getGridDTMScreenSpaceCellWidth() { return data2.z; }

    void setGridDTMHeightTextureID(uint textureID) { data1.z = textureID; }
    void setGridDTMScreenSpaceGridExtents(float2 screenSpaceGridExtends) { data2.xy = screenSpaceGridExtends; }
    void setGridDTMScreenSpaceCellWidth(float screenSpaceGridWidth) { data2.z = screenSpaceGridWidth; }

    void setCurrentWorldToScreenRatio(float worldToScreen) { data6.x = worldToScreen; }
    float getCurrentWorldToScreenRatio() { return data6.x; }

};

// Set 0 - Scene Data and Globals, buffer bindings don't change the buffers only get updated

// [[vk::binding(0, 0)]] ConstantBuffer<Globals> globals; ---> moved to globals.hlsl

[[vk::push_constant]] PushConstants pc;

[[vk::combinedImageSampler]][[vk::binding(1, 0)]] Texture2DArray<float3> msdfTextures : register(t4);
[[vk::combinedImageSampler]][[vk::binding(1, 0)]] SamplerState msdfSampler : register(s4);

[[vk::binding(2, 0)]] SamplerState textureSampler : register(s5);
[[vk::binding(3, 0)]] Texture2D textures[ImagesBindingArraySize] : register(t5);
[[vk::binding(3, 0)]] Texture2D<uint32_t> texturesU32[ImagesBindingArraySize] : register(t5);

// Set 1 - Window dependant data which has higher update frequency due to multiple windows and resize need image recreation and descriptor writes
[[vk::binding(0, 1)]] globallycoherent RWTexture2D<uint> pseudoStencil : register(u0);
[[vk::binding(1, 1)]] globallycoherent RWTexture2D<uint> colorStorage : register(u1);

#endif
