#define FRAGMENT_SHADER_INPUT
#include "common.hlsl"

[shader("pixel")]
float4 fragGeoref(PSInput input) : SV_TARGET
{
    ObjectType objType = input.getObjType();

    float4 color = float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    if (objType == ObjectType::STREAMED_IMAGE) 
    {
        const float2 uv = input.getImageUV();
        const uint32_t textureId = input.getImageTextureId();

        if (textureId != InvalidTextureIndex)
            color = textures[NonUniformResourceIndex(textureId)].Sample(textureSampler, float2(uv.x, uv.y));
    }

    color.rgb *= color.a; // premul alpha
    
    return color;
}
