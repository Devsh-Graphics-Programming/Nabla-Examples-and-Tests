#include "nbl/application_templates/BasicMultiQueueApplication.hpp"
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"

#include "app_resources/descriptors.hlsl"
#include <nbl/builtin/hlsl/central_limit_blur/common.hlsl>


#include "nbl/builtin/CArchive.h"

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;

constexpr uint32_t WorkgroupSize = 256;
constexpr uint32_t PassesPerAxis = 4;

class BoxBlurDemo final
    : public application_templates::BasicMultiQueueApplication,
      public application_templates::
          MonoAssetManagerAndBuiltinResourceApplication {
  using base_t = application_templates::BasicMultiQueueApplication;
  using asset_base_t =
      application_templates::MonoAssetManagerAndBuiltinResourceApplication;

public:
  BoxBlurDemo(const path &_localInputCWD, const path &_localOutputCWD,
              const path &_sharedInputCWD, const path &_sharedOutputCWD)
      : system::IApplicationFramework(_localInputCWD, _localOutputCWD,
                                      _sharedInputCWD, _sharedOutputCWD) {}

  bool onAppInitialized(smart_refctd_ptr<ISystem> &&system) override {
    // Remember to call the base class initialization!
    if (!base_t::onAppInitialized(core::smart_refctd_ptr(system))) {
      return false;
    }
    if (!asset_base_t::onAppInitialized(std::move(system))) {
      return false;
    }

    auto checkedLoad =
        [&]<class T>(const char *filePath) -> smart_refctd_ptr<T> {
      IAssetLoader::SAssetLoadParams lparams = {};
      lparams.logger = m_logger.get();
      lparams.workingDirectory = "";
      // The `IAssetManager::getAsset` function is very complex, in essencee it:
      // 1. takes a cache key or an IFile, if you gave it an `IFile` skip to
      // step 3
      // 2. it consults the loader override about how to get an `IFile` from
      // your cache key
      // 3. handles any failure in opening an `IFile` (which is why it takes a
      // supposed filename), it allows the override to give a different file
      // 4. tries to derive a working directory if you haven't provided one
      // 5. looks for the assets in the cache if you haven't disabled that in
      // the loader parameters 5a. lets the override choose relevant assets from
      // the ones found under the cache key 5b. if nothing was found it lets the
      // override intervene one last time
      // 6. if there's no file to load from, return no assets
      // 7. try all loaders associated with a file extension
      // 8. then try all loaders by opening the file and checking if it will
      // load
      // 9. insert loaded assets into cache if required
      // 10. restore assets from dummy state if needed (more on that in other
      // examples) Take the docs with a grain of salt, the `getAsset` will be
      // rewritten to deal with restores better in the near future.
      nbl::asset::SAssetBundle bundle = m_assetMgr->getAsset(filePath, lparams);
      if (bundle.getContents().empty()) {
        m_logger->log("Asset %s failed to load! Are you sure it exists?",
                      ILogger::ELL_ERROR, filePath);
        return nullptr;
      }
      // All assets derive from `nbl::asset::IAsset`, and can be casted down if
      // the type matches
      static_assert(std::is_base_of_v<nbl::asset::IAsset, T>);
      // The type of the root assets in the bundle is not known until runtime,
      // so this is kinda like a `dynamic_cast` which will return nullptr on
      // type mismatch
      auto typedAsset = IAsset::castDown<T>(
          bundle.getContents()[0]); // just grab the first asset in the bundle
      if (!typedAsset) {
        m_logger->log("Asset type mismatch want %d got %d !",
                      ILogger::ELL_ERROR, T::AssetType, bundle.getAssetType());
      }
      return typedAsset;
    };

    auto textureToBlur = checkedLoad.operator()<nbl::asset::ICPUImage>(
        "../../media/GLI/kueken7_srgb8.png");
    if (!textureToBlur) {
      return logFail("Failed to load texture!\n");
    }
    const auto &inCpuTexInfo = textureToBlur->getCreationParameters();

    auto createGPUImages =
        [&](core::bitflag<IGPUImage::E_USAGE_FLAGS> usageFlags,
            asset::E_FORMAT format,
            std::string_view name) -> smart_refctd_ptr<nbl::video::IGPUImage> {
      video::IGPUImage::SCreationParams gpuImageCreateInfo;
      gpuImageCreateInfo.flags =
          inCpuTexInfo.flags | IImage::ECF_MUTABLE_FORMAT_BIT;
      gpuImageCreateInfo.type = inCpuTexInfo.type;
      gpuImageCreateInfo.extent = inCpuTexInfo.extent;
      gpuImageCreateInfo.mipLevels = inCpuTexInfo.mipLevels;
      gpuImageCreateInfo.arrayLayers = inCpuTexInfo.arrayLayers;
      gpuImageCreateInfo.samples = inCpuTexInfo.samples;
      gpuImageCreateInfo.tiling = video::IGPUImage::TILING::OPTIMAL;
      gpuImageCreateInfo.usage =
          usageFlags | asset::IImage::EUF_TRANSFER_DST_BIT;
      gpuImageCreateInfo.queueFamilyIndexCount = 0u;
      gpuImageCreateInfo.queueFamilyIndices = nullptr;

      gpuImageCreateInfo.format = format;
      // m_physicalDevice->promoteImageFormat({ inCpuTexInfo.format,
      // gpuImageCreateInfo.usage }, gpuImageCreateInfo.tiling);
      // gpuImageCreateInfo.viewFormats.set( E_FORMAT::EF_R8G8B8A8_SRGB );
      // gpuImageCreateInfo.viewFormats.set( E_FORMAT::EF_R8G8B8A8_UNORM );
      auto gpuImage = m_device->createImage(std::move(gpuImageCreateInfo));

      auto gpuImageMemReqs = gpuImage->getMemoryReqs();
      gpuImageMemReqs.memoryTypeBits &=
          m_physicalDevice->getDeviceLocalMemoryTypeBits();
      m_device->allocate(gpuImageMemReqs, gpuImage.get(),
                         video::IDeviceMemoryAllocation::EMAF_NONE);

      gpuImage->setObjectDebugName(name.data());
      return gpuImage;
    };
    smart_refctd_ptr<nbl::video::IGPUImage> gpuImg =
        createGPUImages(IImage::E_USAGE_FLAGS::EUF_TRANSFER_SRC_BIT |
                            IImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT |
                            IImage::E_USAGE_FLAGS::EUF_STORAGE_BIT,
                        E_FORMAT::EF_R8G8B8A8_SRGB, "GPU Image");
    const auto &gpuImgParams = gpuImg->getCreationParameters();

    smart_refctd_ptr<nbl::video::IGPUImageView> sampledView;
    smart_refctd_ptr<nbl::video::IGPUImageView> unormView;
    {
      sampledView = m_device->createImageView({
          .flags = IGPUImageView::ECF_NONE,
          .subUsages = IImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT,
          .image = gpuImg,
          .viewType = IGPUImageView::ET_2D,
          .format = E_FORMAT::EF_R8G8B8A8_SRGB,
      });
      sampledView->setObjectDebugName("Sampled sRGB view");

      unormView = m_device->createImageView({
          .flags = IGPUImageView::ECF_NONE,
          .subUsages = IImage::E_USAGE_FLAGS::EUF_STORAGE_BIT,
          .image = gpuImg,
          .viewType = IGPUImageView::ET_2D,
          .format = E_FORMAT::EF_R8G8B8A8_UNORM,
      });
      unormView->setObjectDebugName("UNORM view");
    }
    assert(gpuImg && sampledView && unormView);

    constexpr uint32_t WorkgroupSize =
        256; // TODO: Number of Passes as parameter
    smart_refctd_ptr<IGPUShader> shader;
    {
      auto computeMain = checkedLoad.operator()<nbl::asset::ICPUShader>(
          "app_resources/main.comp.hlsl");
      smart_refctd_ptr<ICPUShader> overridenUnspecialized =
          CHLSLCompiler::createOverridenCopy(
              computeMain.get(), "#define WORKGROUP_SIZE %s\n",
              std::to_string(WorkgroupSize).c_str());
      shader = m_device->createShader(overridenUnspecialized.get());
      if (!shader) {
        return logFail(
            "Creation of a GPU Shader to from CPU Shader source failed!");
      }
    }

    smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout;
    {
      NBL_CONSTEXPR_STATIC nbl::video::IGPUDescriptorSetLayout::SBinding
          bindings[] = {
              {.binding = inputViewBinding,
               .type =
                   nbl::asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
               .createFlags =
                   IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
               .stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
               .count = 1,
               .immutableSamplers = nullptr},
              {.binding = outputViewBinding,
               .type = nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
               .createFlags =
                   IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
               .stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
               .count = 1,
               .immutableSamplers = nullptr}};
      dsLayout = m_device->createDescriptorSetLayout(bindings);
      if (!dsLayout) {
        return logFail("Failed to create a Descriptor Layout!\n");
      }
    }

    const asset::SPushConstantRange pushConst[] = {
        {.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
         .offset = 0,
         .size = sizeof(nbl::hlsl::central_limit_blur::BoxBlurParams)}};
    smart_refctd_ptr<nbl::video::IGPUPipelineLayout> pplnLayout =
        m_device->createPipelineLayout(pushConst, smart_refctd_ptr(dsLayout));
    if (!pplnLayout) {
      return logFail("Failed to create a Pipeline Layout!\n");
    }

    smart_refctd_ptr<nbl::video::IGPUComputePipeline> pipeline;
    {
      IGPUComputePipeline::SCreationParams params = {};
      params.layout = pplnLayout.get();
      params.shader.entryPoint = "main";
      params.shader.shader = shader.get();
      if (!m_device->createComputePipelines(nullptr, {&params, 1}, &pipeline)) {
        return logFail(
            "Failed to create pipelines (compile & link shaders)!\n");
      }
    }
    smart_refctd_ptr<video::IGPUSampler> sampler = m_device->createSampler({});
    smart_refctd_ptr<nbl::video::IDescriptorPool> pool =
        m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE,
                                                   {&dsLayout.get(), 1});
    smart_refctd_ptr<nbl::video::IGPUDescriptorSet> ds =
        pool->createDescriptorSet(std::move(dsLayout));
    {
      // Views must be in the same layout because we read from them
      // simultaneously
      IGPUDescriptorSet::SDescriptorInfo info[2];
      info[0].desc = sampledView;
      info[0].info.combinedImageSampler.sampler = sampler;
      info[0].info.combinedImageSampler.imageLayout = IImage::LAYOUT::GENERAL;
      info[1].desc = unormView;
      info[1].info.image.imageLayout = IImage::LAYOUT::GENERAL;

      IGPUDescriptorSet::SWriteDescriptorSet writes[] = {
          {.dstSet = ds.get(),
           .binding = inputViewBinding,
           .arrayElement = 0,
           .count = 1,
           .info = &info[0]},
          {.dstSet = ds.get(),
           .binding = outputViewBinding,
           .arrayElement = 0,
           .count = 1,
           .info = &info[1]},
      };
      const bool success = m_device->updateDescriptorSets(writes, {});
      assert(success);
    }

    ds->setObjectDebugName("Box blur DS");
    pplnLayout->setObjectDebugName("Box Blur PPLN Layout");

    IQueue *queue = getComputeQueue();

    // Transfer stage
    auto transferSema = m_device->createSemaphore(0);
    IQueue::SSubmitInfo::SSemaphoreInfo transferDone[] = {
        {.semaphore = transferSema.get(),
         .value = 1,
         .stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS}};
    {

      smart_refctd_ptr<nbl::video::IGPUCommandBuffer> cmdbuf;
      smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool =
          m_device->createCommandPool(
              queue->getFamilyIndex(),
              IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
      if (!cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,
                                         1u, &cmdbuf)) {
        return logFail("Failed to create Command Buffers!\n");
      }

      cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

      const IGPUCommandBuffer::SImageMemoryBarrier<
          IGPUCommandBuffer::SOwnershipTransferBarrier>
          imgLayouts[] = {{
              .barrier =
                  {
                      .dep =
                          {// there's no need for a source synchronization
                           // because Host Ops become available and visible
                           // pre-submit
                           .srcStageMask = PIPELINE_STAGE_FLAGS::NONE,
                           .srcAccessMask = ACCESS_FLAGS::NONE,
                           .dstStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT,
                           .dstAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT},
                  },
              .image = gpuImg.get(),
              .subresourceRange = {.aspectMask = IImage::EAF_COLOR_BIT,
                                   .levelCount = 1,
                                   .layerCount = 1},
              .oldLayout = IImage::LAYOUT::UNDEFINED,
              .newLayout = IImage::LAYOUT::TRANSFER_DST_OPTIMAL,
          }};
      if (!cmdbuf->pipelineBarrier(nbl::asset::EDF_NONE,
                                   {.imgBarriers = imgLayouts})) {
        return logFail("Failed to issue barrier!\n");
      }

      queue->startCapture();
      IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[] = {
          {.cmdbuf = cmdbuf.get()}};
      core::smart_refctd_ptr<ISemaphore> imgFillSemaphore =
          m_device->createSemaphore(0);
      imgFillSemaphore->setObjectDebugName("Image Fill Semaphore");
      SIntendedSubmitInfo intendedSubmit = {
          .queue = queue,
          .waitSemaphores = {/*wait for no - one*/},
          .commandBuffers = cmdbufs,
          .scratchSemaphore = {.semaphore = imgFillSemaphore.get(),
                               .value = 0,
                               .stageMask =
                                   PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS}};
      std::cout << "pre submit\n";
      m_utils->updateImageViaStagingBufferAutoSubmit(
          intendedSubmit, textureToBlur->getBuffer()->getPointer(),
          inCpuTexInfo.format, gpuImg.get(),
          IImage::LAYOUT::TRANSFER_DST_OPTIMAL, textureToBlur->getRegions());

      queue->endCapture();

      // WARNING : Depending on OVerflows, `transferDone->value!=1` so if you
      // want to sync the compute submit against that, use `transferDone`
      // directly as the wait semaphore! const ISemaphore::SWaitInfo waitInfo =
      // {transferDone->semaphore,transferDone->value};
      // m_device->blockForSemaphores( { &waitInfo,1 } );
    }

    IImage::SSubresourceLayers subresourceLayers;
    subresourceLayers.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
    subresourceLayers.mipLevel = 0u;
    subresourceLayers.baseArrayLayer = 0u;
    subresourceLayers.layerCount = 1u;

    IImage::SBufferCopy bufferCopy;
    bufferCopy.bufferImageHeight = gpuImgParams.extent.height;
    bufferCopy.bufferRowLength = gpuImgParams.extent.width;
    bufferCopy.bufferOffset = 0u;
    bufferCopy.imageExtent = gpuImgParams.extent;
    bufferCopy.imageSubresource = subresourceLayers;

    nbl::video::IDeviceMemoryAllocator::SAllocation outputBufferAllocation = {};
    smart_refctd_ptr<IGPUBuffer> outputImageBuffer = nullptr;
    {
      IGPUBuffer::SCreationParams gpuBufCreationParams;
      gpuBufCreationParams.size = gpuImg->getImageDataSizeInBytes();
      // VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      // VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
      gpuBufCreationParams.usage =
          IGPUBuffer::E_USAGE_FLAGS::EUF_TRANSFER_DST_BIT;
      outputImageBuffer =
          m_device->createBuffer(std::move(gpuBufCreationParams));
      if (!outputImageBuffer)
        return logFail("Failed to create a GPU Buffer of size %d!\n",
                       gpuBufCreationParams.size);

      // Naming objects is cool because not only errors (such as Vulkan
      // Validation Layers) will show their names, but RenderDoc captures too.
      outputImageBuffer->setObjectDebugName("Output Image Buffer");

      nbl::video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs =
          outputImageBuffer->getMemoryReqs();
      // you can simply constrain the memory requirements by AND-ing the type
      // bits of the host visible memory types
      reqs.memoryTypeBits &= m_physicalDevice->getHostVisibleMemoryTypeBits();

      outputBufferAllocation =
          m_device->allocate(reqs, outputImageBuffer.get(),
                             nbl::video::IDeviceMemoryAllocation::
                                 E_MEMORY_ALLOCATE_FLAGS::EMAF_NONE);
      if (!outputBufferAllocation.isValid())
        return logFail("Failed to allocate Device Memory compatible with our "
                       "GPU Buffer!\n");
    }

    constexpr size_t StartedValue = 0;
    constexpr size_t FinishedValue = 45;
    static_assert(StartedValue < FinishedValue);
    smart_refctd_ptr<ISemaphore> progress =
        m_device->createSemaphore(StartedValue);

    smart_refctd_ptr<nbl::video::IGPUCommandBuffer> cmdbuf;
    smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool =
        m_device->createCommandPool(
            queue->getFamilyIndex(),
            IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
    if (!cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,
                                       1u, &cmdbuf)) {
      return logFail("Failed to create Command Buffers!\n");
    }

    hlsl::central_limit_blur::BoxBlurParams pushConstData = {
        .radius = 4.f,
        .direction = 0,
        .channelCount = nbl::asset::getFormatChannelCount(gpuImgParams.format),
        .wrapMode = hlsl::central_limit_blur::WrapMode::WRAP_MODE_CLAMP_TO_EDGE,
        .borderColorType = hlsl::central_limit_blur::BorderColor::
            BORDER_COLOR_FLOAT_OPAQUE_WHITE,
    };

    cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
    cmdbuf->beginDebugMarker("Box Blur dispatches",
                             core::vectorSIMDf(0, 1, 0, 1));

    const IGPUCommandBuffer::SImageMemoryBarrier<
        IGPUCommandBuffer::SOwnershipTransferBarrier>
        barriers[] = {{
            .barrier =
                {
                    .dep =
                        {
                            .srcStageMask =
                                nbl::asset::PIPELINE_STAGE_FLAGS::COPY_BIT,
                            .srcAccessMask =
                                nbl::asset::ACCESS_FLAGS::TRANSFER_WRITE_BIT,
                            .dstStageMask = nbl::asset::PIPELINE_STAGE_FLAGS::
                                                COMPUTE_SHADER_BIT |
                                            nbl::asset::PIPELINE_STAGE_FLAGS::
                                                ALL_TRANSFER_BITS,
                            .dstAccessMask =
                                nbl::asset::ACCESS_FLAGS::STORAGE_WRITE_BIT,
                        },
                },
            .image = gpuImg.get(),
            .subresourceRange = {.aspectMask = IImage::EAF_COLOR_BIT,
                                 .levelCount = 1,
                                 .layerCount = 1},
            .oldLayout = IImage::LAYOUT::UNDEFINED,
            .newLayout = IImage::LAYOUT::GENERAL,
        }};
    if (!cmdbuf->pipelineBarrier(nbl::asset::EDF_NONE,
                                 {.imgBarriers = barriers}))
      return logFail("Failed to issue barrier!\n");

    cmdbuf->bindComputePipeline(pipeline.get());
    cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, pplnLayout.get(), 0, 1,
                               &ds.get());
    cmdbuf->pushConstants(pplnLayout.get(),
                          IShader::E_SHADER_STAGE::ESS_COMPUTE, 0,
                          sizeof(pushConstData), &pushConstData);

    for (int j = 0; j < 1; j++) {
      cmdbuf->dispatch(1, gpuImgParams.extent.height, 1);

      // const nbl::asset::SMemoryBarrier barriers3[] = {
      // 	{
      // 		.srcStageMask =
      // nbl::asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT, 		.srcAccessMask =
      // nbl::asset::ACCESS_FLAGS::SHADER_WRITE_BITS, 		.dstStageMask =
      // nbl::asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT, 		.dstAccessMask=
      // nbl::asset::ACCESS_FLAGS::SHADER_READ_BITS,
      // 	}
      // };
      // // TODO: you don't need a pipeline barrier just before the end of the
      // last command buffer to be submitted
      // // Timeline semaphore takes care of all the memory deps between a
      // signal and a wait if( !cmdbuf->pipelineBarrier( nbl::asset::EDF_NONE, {
      // .memBarriers = barriers3 } ) )
      // {
      // 	return logFail( "Failed to issue barrier!\n" );
      // }

      // pushConstData.direction = 1;
      // cmdbuf->pushConstants( pplnLayout.get(),
      // IShader::E_SHADER_STAGE::ESS_COMPUTE, 0, sizeof( pushConstData ),
      // &pushConstData ); cmdbuf->dispatch(1, gpuImgParams.extent.width, 1);
    }

    const IGPUCommandBuffer::SImageMemoryBarrier<
        IGPUCommandBuffer::SOwnershipTransferBarrier>
        barriers2[] = {{
            .barrier =
                {
                    .dep =
                        {
                            .srcStageMask = nbl::asset::PIPELINE_STAGE_FLAGS::
                                COMPUTE_SHADER_BIT,
                            .srcAccessMask =
                                nbl::asset::ACCESS_FLAGS::STORAGE_WRITE_BIT,
                            .dstStageMask = nbl::asset::PIPELINE_STAGE_FLAGS::
                                ALL_TRANSFER_BITS,
                            .dstAccessMask =
                                nbl::asset::ACCESS_FLAGS::MEMORY_READ_BITS,
                        },
                },
            .image = gpuImg.get(),
            .subresourceRange = {.aspectMask = IImage::EAF_COLOR_BIT,
                                 .levelCount = 1,
                                 .layerCount = 1},
            .oldLayout = IImage::LAYOUT::UNDEFINED,
            .newLayout = IImage::LAYOUT::TRANSFER_SRC_OPTIMAL,
        }};
    if (!cmdbuf->pipelineBarrier(nbl::asset::EDF_NONE,
                                 {.imgBarriers = barriers2}))
      return logFail("Failed to issue barrier!\n");

    // Copy the resulting image to a buffer.
    cmdbuf->copyImageToBuffer(gpuImg.get(),
                              IImage::LAYOUT::TRANSFER_SRC_OPTIMAL,
                              outputImageBuffer.get(), 1u, &bufferCopy);

    cmdbuf->endDebugMarker();
    cmdbuf->end();

    {
      const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[] = {
          {.cmdbuf = cmdbuf.get()}};
      const IQueue::SSubmitInfo::SSemaphoreInfo signals[] = {
          {.semaphore = progress.get(),
           .value = FinishedValue,
           .stageMask = asset::PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS}};
      IQueue::SSubmitInfo submitInfos[] = {
          {.waitSemaphores = {/*transferDone*/},
           .commandBuffers = cmdbufs,
           .signalSemaphores = signals}};

      // This is super useful for debugging multi-queue workloads and by default
      // RenderDoc delimits captures only by Swapchain presents.
      queue->startCapture();
      queue->submit(submitInfos);
      queue->endCapture();
    }
    const ISemaphore::SWaitInfo waitInfos[] = {
        {.semaphore = progress.get(), .value = FinishedValue}};
    m_device->blockForSemaphores(waitInfos);

    // Map memory, so contents of `outputImageBuffer` will be host visible.
    const ILogicalDevice::MappedMemoryRange memoryRange(
        outputBufferAllocation.memory.get(), 0ull,
        outputBufferAllocation.memory->getAllocationSize());
    auto imageBufferMemPtr = outputBufferAllocation.memory->map(
        {0ull, outputBufferAllocation.memory->getAllocationSize()},
        IDeviceMemoryAllocation::EMCAF_READ);
    if (!imageBufferMemPtr)
      return logFail("Failed to map the Device Memory!\n");

    // If the mapping is not coherent the range needs to be invalidated to pull
    // in new data for the CPU's caches.
    if (!outputBufferAllocation.memory->getMemoryPropertyFlags().hasFlags(
            IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
      m_device->invalidateMappedMemoryRanges(1, &memoryRange);

    // While JPG/PNG/BMP/EXR Loaders create ICPUImages because they cannot
    // disambiguate colorspaces, 2D_ARRAY vs 2D and even sometimes formats
    // (looking at your PNG normalmaps!), the writers are always meant to be fed
    // by ICPUImageViews.
    ICPUImageView::SCreationParams params = {};
    {

      // ICPUImage isn't really a representation of a GPU Image in itself, more
      // of a recipe for creating one from a series of ICPUBuffer to ICPUImage
      // copies. This means that an ICPUImage has no internal storage or memory
      // bound for its texels and rather references separate ICPUBuffer ranges
      // to provide its contents, which also means it can be sparsely(with gaps)
      // specified.
      params.image = ICPUImage::create(gpuImgParams);
      {
        // CDummyCPUBuffer is used for creating ICPUBuffer over an already
        // existing memory, without any memcopy operations or taking over the
        // memory ownership. CDummyCPUBuffer cannot free its memory.
        auto cpuOutputImageBuffer =
            core::make_smart_refctd_ptr<CDummyCPUBuffer>(
                gpuImg->getImageDataSizeInBytes(), imageBufferMemPtr,
                core::adopt_memory_t());
        ICPUImage::SBufferCopy region = {};
        region.imageSubresource.aspectMask =
            IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = gpuImgParams.extent;

        //
        params.image->setBufferAndRegions(
            std::move(cpuOutputImageBuffer),
            core::make_refctd_dynamic_array<
                core::smart_refctd_dynamic_array<ICPUImage::SBufferCopy>>(
                1, region));
      }
      // Only DDS and KTX support exporting layered views.
      params.viewType = ICPUImageView::ET_2D;
      params.format = gpuImgParams.format;
      params.subresourceRange.aspectMask =
          IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
      params.subresourceRange.layerCount = 1;
    }
    auto cpuImageView = ICPUImageView::create(std::move(params));
    asset::IAssetWriter::SAssetWriteParams writeParams(cpuImageView.get());
    m_assetMgr->writeAsset("blit.png", writeParams);

    // Even if you forgot to unmap, it would unmap itself when
    // `outputBufferAllocation.memory` gets dropped by its last reference and
    // its destructor runs.
    outputBufferAllocation.memory->unmap();

    return true;
  }

  // Platforms like WASM expect the main entry point to periodically return
  // control, hence if you want a crossplatform app, you have to let the
  // framework deal with your "game loop"
  void workLoopBody() override {}

  // Whether to keep invoking the above. In this example because its headless
  // GPU compute, we do all the work in the app initialization.
  bool keepRunning() override { return false; }

  // Just to run destructors in a nice order
  bool onAppTerminated() override { return base_t::onAppTerminated(); }
};

NBL_MAIN_FUNC(BoxBlurDemo)