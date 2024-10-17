#ifndef _FLIP_EXAMPLE_BINDINGS_HLSL
#define _FLIP_EXAMPLE_BINDINGS_HLSL

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

// particlesInit
NBL_CONSTEXPR uint32_t s_pi = 1;
NBL_CONSTEXPR uint32_t b_piGridData = 0;
NBL_CONSTEXPR uint32_t b_piPBuffer = 1;

#ifndef __HLSL_VERSION
NBL_CONSTEXPR IGPUDescriptorSetLayout::SBinding piParticlesInit_bs1[] = {
    {
        .binding = b_piGridData,
        .type = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_piPBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
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
NBL_CONSTEXPR uint32_t b_gpvPBuffer = 2;
NBL_CONSTEXPR uint32_t b_gpvPVertBuffer = 3;

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
    },
    {
        .binding = b_gpvPBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_gpvPVertBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    }
};
#endif

// updateFluidCells
NBL_CONSTEXPR uint32_t s_ufc = 1;
NBL_CONSTEXPR uint32_t b_ufcGridData = 0;
NBL_CONSTEXPR uint32_t b_ufcPBuffer = 1;
NBL_CONSTEXPR uint32_t b_ufcGridIDBuffer = 2;
NBL_CONSTEXPR uint32_t b_ufcCMInBuffer = 3;
NBL_CONSTEXPR uint32_t b_ufcCMOutBuffer = 4;
NBL_CONSTEXPR uint32_t b_ufcVelBuffer = 5;
NBL_CONSTEXPR uint32_t b_ufcPrevVelBuffer = 6;

#ifndef __HLSL_VERSION
NBL_CONSTEXPR IGPUDescriptorSetLayout::SBinding ufcFluidCell_bs1[] = {
    {
        .binding = b_ufcGridData,
        .type = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_ufcGridIDBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_ufcCMOutBuffer,
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
        .binding = b_ufcCMInBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_ufcCMOutBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    }
};
NBL_CONSTEXPR IGPUDescriptorSetLayout::SBinding ufcParticleToCell_bs1[] = {
    {
        .binding = b_ufcGridData,
        .type = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_ufcPBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_ufcGridIDBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_ufcCMInBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_ufcVelBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 3,
    },
    {
        .binding = b_ufcPrevVelBuffer,
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
NBL_CONSTEXPR uint32_t b_abfVelFieldBuffer = 1;
NBL_CONSTEXPR uint32_t b_abfCMBuffer = 2;

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
        .binding = b_abfVelFieldBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 3,
    },
    {
        .binding = b_abfCMBuffer,
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
NBL_CONSTEXPR uint32_t b_dCMBuffer = 1;
NBL_CONSTEXPR uint32_t b_dVelBuffer = 2;
NBL_CONSTEXPR uint32_t b_dAxisInBuffer = 3;
NBL_CONSTEXPR uint32_t b_dAxisOutBuffer = 4;
NBL_CONSTEXPR uint32_t b_dDiffInBuffer = 5;
NBL_CONSTEXPR uint32_t b_dDiffOutBuffer = 6;

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
        .binding = b_dCMBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_dAxisOutBuffer,
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
        .binding = b_dAxisInBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_dAxisOutBuffer,
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
        .binding = b_dCMBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_dVelBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 3,
    },
    {
        .binding = b_dAxisInBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_dDiffInBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_dDiffOutBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    }
};
NBL_CONSTEXPR IGPUDescriptorSetLayout::SBinding dUpdateVelD_bs1[] = {
    {
        .binding = b_dGridData,
        .type = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_dCMBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_dVelBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 3,
    },
    {
        .binding = b_dDiffInBuffer,
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
NBL_CONSTEXPR uint32_t b_psCMBuffer = 2;
NBL_CONSTEXPR uint32_t b_psVelBuffer = 3;
NBL_CONSTEXPR uint32_t b_psDivBuffer = 4;
NBL_CONSTEXPR uint32_t b_psPresInBuffer = 5;
NBL_CONSTEXPR uint32_t b_psPresOutBuffer = 6;

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
        .binding = b_psCMBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_psVelBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 3,
    },
    {
        .binding = b_psDivBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    }
};
NBL_CONSTEXPR IGPUDescriptorSetLayout::SBinding psSolvePressure_bs1[] = {
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
        .binding = b_psCMBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_psDivBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_psPresInBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_psPresOutBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
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
        .binding = b_psCMBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_psVelBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 3,
    },
    {
        .binding = b_psPresInBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    }
};
#endif

// extrapolateVelocities
NBL_CONSTEXPR uint32_t s_ev = 1;
NBL_CONSTEXPR uint32_t b_evGridData = 0;
NBL_CONSTEXPR uint32_t b_evPBuffer = 1;
NBL_CONSTEXPR uint32_t b_evVelFieldBuffer = 2;
NBL_CONSTEXPR uint32_t b_evPrevVelFieldBuffer = 3;
NBL_CONSTEXPR uint32_t b_evVelSampler = 4;

#ifndef __HLSL_VERSION
NBL_CONSTEXPR IGPUDescriptorSetLayout::SBinding evExtrapolateVel_bs1[] = {
    {
        .binding = b_evGridData,
        .type = asset::IDescriptor::E_TYPE::ET_UNIFORM_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_evPBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_evVelFieldBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_SAMPLED_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 3,
    },
    {
        .binding = b_evPrevVelFieldBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_SAMPLED_IMAGE,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 3,
    },
    {
        .binding = b_evVelSampler,
        .type = asset::IDescriptor::E_TYPE::ET_SAMPLER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    }
};
#endif

// advectParticles
NBL_CONSTEXPR uint32_t s_ap = 1;
NBL_CONSTEXPR uint32_t b_apGridData = 0;
NBL_CONSTEXPR uint32_t b_apPBuffer = 1;
NBL_CONSTEXPR uint32_t b_apVelFieldBuffer = 2;
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
        .binding = b_apPBuffer,
        .type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
        .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
        .count = 1,
    },
    {
        .binding = b_apVelFieldBuffer,
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

// prepareCellUpdate
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