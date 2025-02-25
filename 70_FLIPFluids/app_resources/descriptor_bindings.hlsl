#ifndef _FLIP_EXAMPLE_BINDINGS_HLSL
#define _FLIP_EXAMPLE_BINDINGS_HLSL

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

#ifndef __HLSL_VERSION
using namespace nbl;
using namespace video;
#endif

// particlesInit
NBL_CONSTEXPR uint32_t s_pi = 1;
NBL_CONSTEXPR uint32_t b_piGridData = 0;

#ifndef __HLSL_VERSION
NBL_CONSTEXPR IGPUDescriptorSetLayout::SBinding piParticlesInit_bs1[] = {
    {
        .binding = b_piGridData,
        .type = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    }
};
#endif

// genParticleVertices
NBL_CONSTEXPR uint32_t s_gpv = 1;
NBL_CONSTEXPR uint32_t b_gpvCamData = 0;
NBL_CONSTEXPR uint32_t b_gpvPParams = 1;

#ifndef __HLSL_VERSION
NBL_CONSTEXPR IGPUDescriptorSetLayout::SBinding gpvGenVertices_bs1[] = {
    {
        .binding = b_gpvCamData,
        .type = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_gpvPParams,
        .type = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    }
};
#endif

// updateFluidCells
NBL_CONSTEXPR uint32_t s_ufc = 1;
NBL_CONSTEXPR uint32_t b_ufcGridData = 0;
NBL_CONSTEXPR uint32_t b_ufcGridPCount = 1;
NBL_CONSTEXPR uint32_t b_ufcCMIn = 2;
NBL_CONSTEXPR uint32_t b_ufcCMOut = 3;
NBL_CONSTEXPR uint32_t b_ufcVel = 4;
NBL_CONSTEXPR uint32_t b_ufcPrevVel = 5;

#ifndef __HLSL_VERSION
NBL_CONSTEXPR IGPUDescriptorSetLayout::SBinding ufcAccWeights_bs1[] = {
    {
        .binding = b_ufcGridData,
        .type = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_ufcGridPCount,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_ufcVel,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 3,
    },
    {
        .binding = b_ufcPrevVel,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 3,
    }
};
NBL_CONSTEXPR IGPUDescriptorSetLayout::SBinding ufcFluidCell_bs1[] = {
    {
        .binding = b_ufcGridData,
        .type = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_ufcGridPCount,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_ufcCMOut,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    }
};
NBL_CONSTEXPR IGPUDescriptorSetLayout::SBinding ufcNeighborCell_bs1[] = {
    {
        .binding = b_ufcGridData,
        .type = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_ufcCMIn,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_ufcCMOut,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_ufcVel,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 3,
    },
    {
        .binding = b_ufcPrevVel,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 3,
    }
};
#endif

// applyBodyForces
NBL_CONSTEXPR uint32_t s_abf = 1;
NBL_CONSTEXPR uint32_t b_abfGridData = 0;
NBL_CONSTEXPR uint32_t b_abfVelField = 1;
NBL_CONSTEXPR uint32_t b_abfCM = 2;

#ifndef __HLSL_VERSION
NBL_CONSTEXPR IGPUDescriptorSetLayout::SBinding abfApplyForces_bs1[] = {
    {
        .binding = b_abfGridData,
        .type = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_abfVelField,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 3,
    },
    {
        .binding = b_abfCM,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    }
};
#endif

// diffusion
NBL_CONSTEXPR uint32_t s_d = 1;
NBL_CONSTEXPR uint32_t b_dGridData = 0;
NBL_CONSTEXPR uint32_t b_dCM = 1;
NBL_CONSTEXPR uint32_t b_dVel = 2;
NBL_CONSTEXPR uint32_t b_dAxisIn = 3;
NBL_CONSTEXPR uint32_t b_dAxisOut = 4;
NBL_CONSTEXPR uint32_t b_dDiff = 5;

#ifndef __HLSL_VERSION
NBL_CONSTEXPR IGPUDescriptorSetLayout::SBinding dAxisCM_bs1[] = {
    {
        .binding = b_dGridData,
        .type = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_dCM,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_dAxisOut,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    }
};
NBL_CONSTEXPR IGPUDescriptorSetLayout::SBinding dNeighborAxisCM_bs1[] = {
    {
        .binding = b_dGridData,
        .type = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_dAxisIn,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_dAxisOut,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    }
};
NBL_CONSTEXPR IGPUDescriptorSetLayout::SBinding dDiffuse_bs1[] = {
    {
        .binding = b_dGridData,
        .type = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_dCM,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_dVel,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 3,
    },
    {
        .binding = b_dAxisIn,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_dDiff,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    }
};
#endif

// pressureSolver
NBL_CONSTEXPR uint32_t s_ps = 1;
NBL_CONSTEXPR uint32_t b_psGridData = 0;
NBL_CONSTEXPR uint32_t b_psParams = 1;
NBL_CONSTEXPR uint32_t b_psCM = 2;
NBL_CONSTEXPR uint32_t b_psVel = 3;
NBL_CONSTEXPR uint32_t b_psDiv = 4;
NBL_CONSTEXPR uint32_t b_psPres = 5;

#ifndef __HLSL_VERSION
NBL_CONSTEXPR IGPUDescriptorSetLayout::SBinding psDivergence_bs1[] = {
    {
        .binding = b_psGridData,
        .type = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_psCM,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_psVel,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 3,
    },
    {
        .binding = b_psDiv,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    }
};
NBL_CONSTEXPR IGPUDescriptorSetLayout::SBinding psIteratePressure_bs1[] = {
    {
        .binding = b_psGridData,
        .type = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_psParams,
        .type = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_psCM,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_psDiv,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_psPres,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    }
};
NBL_CONSTEXPR IGPUDescriptorSetLayout::SBinding psUpdateVelPs_bs1[] = {
    {
        .binding = b_psGridData,
        .type = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_psParams,
        .type = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_psCM,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_psVel,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 3,
    },
    {
        .binding = b_psPres,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    }
};
#endif

// advectParticles
NBL_CONSTEXPR uint32_t s_ap = 1;
NBL_CONSTEXPR uint32_t b_apGridData = 0;
NBL_CONSTEXPR uint32_t b_apVelField = 1;
NBL_CONSTEXPR uint32_t b_apPrevVelField = 2;
NBL_CONSTEXPR uint32_t b_apVelSampler = 3;

#ifndef __HLSL_VERSION
NBL_CONSTEXPR IGPUDescriptorSetLayout::SBinding apAdvectParticles_bs1[] = {
    {
        .binding = b_apGridData,
        .type = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_apVelField,
        .type = asset::IDescriptor::E_TYPE::ET_SAMPLED_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 3,
    },
    {
        .binding = b_apPrevVelField,
        .type = asset::IDescriptor::E_TYPE::ET_SAMPLED_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 3,
    },
    {
        .binding = b_apVelSampler,
        .type = asset::IDescriptor::E_TYPE::ET_SAMPLER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    }
};
#endif

#endif