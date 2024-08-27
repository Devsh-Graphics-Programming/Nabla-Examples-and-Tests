#ifndef _FLIP_EXAMPLE_BINDINGS_HLSL
#define _FLIP_EXAMPLE_BINDINGS_HLSL

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

NBL_CONSTEXPR uint32_t s_pcu = 1;
NBL_CONSTEXPR uint32_t b_pcuGridData = 0;
NBL_CONSTEXPR uint32_t b_pcuPInBuffer = 1;
NBL_CONSTEXPR uint32_t b_pcuPOutBuffer = 2;
NBL_CONSTEXPR uint32_t b_pcuPairBuffer = 3;
NBL_CONSTEXPR uint32_t b_pcuGridIDBuffer = 4;

#ifndef __HLSL_VERSION

NBL_CONSTEXPR IGPUDescriptorSetLayout::SBinding pcuMakePairs_bs1[] = {
    {
        .binding = b_pcuGridData,
        .type = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_pcuPInBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_pcuPairBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    }
};
NBL_CONSTEXPR IGPUDescriptorSetLayout::SBinding pcuSetGridID_bs1[] = {
    {
        .binding = b_pcuPairBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_pcuGridIDBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    }
};
NBL_CONSTEXPR IGPUDescriptorSetLayout::SBinding pcuShuffle_bs1[] = {
    {
        .binding = b_pcuPInBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_pcuPOutBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_pcuPairBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    }
};

#endif

#endif