#pragma once

#include <nabla.h>

using namespace nbl;
using namespace nbl::video;
using namespace nbl::core;

class GPUPrefixSum
{

};

class GPURadixSort
{
public:
    void initialize();

    void sort();

private:
    smart_refctd_ptr<IGPUComputePipeline> m_localSortPipeline;
    smart_refctd_ptr<IGPUComputePipeline> m_globalMergePipeline;

    // buffers
};
