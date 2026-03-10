/******************************************************************************/
/* DrawResourcesFiller: This class provides important functionality to manage resources needed for a draw.
/******************************************************************************/
#pragma once

#if __has_include("glm/glm/glm.hpp") // legacy
#include "glm/glm/glm.hpp"
#else
#include "glm/glm.hpp" // new build system
#endif
#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/vector.hlsl>
#include <nbl/builtin/hlsl/limits.hlsl>
#include <nbl/core/containers/LRUCache.h>
#include <nbl/video/utilities/SIntendedSubmitInfo.h>
#include "CTriangleMesh.h"
#include "Shaders/globals.hlsl"

using namespace nbl;
using namespace nbl::video;
using namespace nbl::core;
using namespace nbl::asset;

static_assert(sizeof(DrawObject) == 16u);

// ! DrawResourcesFiller
// ! This class provides important functionality to manage resources needed for a draw.
// ! Drawing new objects (polylines, hatches, etc.) should go through this function.
// ! Contains all the scene resources (buffers and images)
// ! In the case of overflow (i.e. not enough remaining v-ram) will auto-submit/render everything recorded so far,
//   and additionally makes sure relavant data needed for those draw calls are present in memory
struct DrawResourcesFiller
{
	struct DrawCallData
	{
		uint64_t indexBufferOffset;
		uint64_t indexCount;
		uint64_t triangleMeshVerticesBaseAddress;
		uint32_t triangleMeshMainObjectIndex;
	};

public:
	
	// We pack multiple data types in a single buffer, we need to makes sure each offset starts aligned to avoid mis-aligned accesses
	static constexpr size_t GPUStructsMaxNaturalAlignment = 8u;
	static constexpr size_t MinimumDrawResourcesMemorySize = 512u * 1 << 20u; // 512MB

	/**
	 * @brief Attempts to allocate draw resources within a given VRAM budget, retrying with progressively smaller sizes on failure.
	 *
	 * This function preserves the initial image-to-buffer memory ratio. If the initial sizes are too small,
	 * it scales them up to meet a minimum required threshold. On allocation failure, it reduces the memory
	 * sizes by a specified percentage and retries, until it either succeeds or the number of attempts exceeds `maxTries`.
	 *
	 * @param logicalDevice Pointer to the logical device used for allocation.
	 * @param maxImageMemorySize Initial image memory size (in bytes) to attempt allocation with.
	 * @param maxBufferMemorySize Initial buffer memory size (in bytes) to attempt allocation with.
	 * @param memoryTypeIndexTryOrder Ordered list of memory type indices to attempt allocation with, in the order they should be tried.
	 * @param reductionPercent The percentage by which to reduce the memory sizes after each failed attempt (e.g., 10 means reduce by 10%).
	 * @param maxTries Maximum number of attempts to try reducing and allocating memory.
	 *
	 * @return true if the allocation succeeded at any iteration; false if all attempts failed.
	 */
	bool allocateDrawResourcesWithinAvailableVRAM(ILogicalDevice* logicalDevice, size_t maxImageMemorySize, size_t maxBufferMemorySize, std::span<uint32_t> memoryTypeIndexTryOrder, uint32_t reductionPercent = 10u, uint32_t maxTries = 32u);

	/// @brief call this function before submitting to ensure all buffer and textures resourcesCollection requested via drawing calls are copied to GPU
	/// records copy command into intendedNextSubmit's active command buffer and might possibly submits if fails allocation on staging upload memory.
	bool pushAllUploads(SIntendedSubmitInfo& intendedNextSubmit);

	/// @brief general parent struct for 1.ReservedCompute and 2.CPUGenerated Resources
	struct ResourceBase
	{
		static constexpr size_t InvalidBufferOffset = ~0u;
		size_t bufferOffset = InvalidBufferOffset; // set when copy to gpu buffer is issued
		virtual size_t getCount() const = 0;
		virtual size_t getStorageSize() const = 0;
		virtual size_t getAlignedStorageSize() const { return core::alignUp(getStorageSize(), GPUStructsMaxNaturalAlignment); }
	};

	/// @brief ResourceBase reserved for compute shader stages input/output
	template <typename T>
	struct ReservedComputeResource : ResourceBase
	{
		size_t count = 0ull;
		size_t getCount() const override { return count; }
		size_t getStorageSize() const override  { return count * sizeof(T); }
	};

	/// @brief ResourceBase which is filled by CPU, packed and sent to GPU
	template <typename T>
	struct CPUGeneratedResource : ResourceBase
	{
		core::vector<T> vector;
		size_t getCount() const { return vector.size(); }
		size_t getStorageSize() const { return vector.size() * sizeof(T); }
		
		/// @return pointer to start of the data to be filled, up to additionalCount
		T* increaseCountAndGetPtr(size_t additionalCount) 
		{
			size_t offset = vector.size();
			vector.resize(offset + additionalCount);
			return &vector[offset];
		}

		/// @brief increases size of general-purpose resources that hold bytes
		/// @param alignment: Alignment of the pointer returned to be filled, should be PoT and <= GPUStructsMaxNaturalAlignment, only use this if storing raw bytes in vector
		/// @return pointer to start of the data to be filled, up to additional size
		size_t increaseSizeAndGetOffset(size_t additionalSize, size_t alignment) 
		{
			assert(core::isPoT(alignment) && alignment <= GPUStructsMaxNaturalAlignment);
			size_t offset = core::alignUp(vector.size(), alignment);
			vector.resize(offset + additionalSize);
			return offset;
		}
		
		uint32_t addAndGetOffset(const T& val)
		{
			vector.push_back(val);
			return vector.size() - 1u;
		}

		T* data() { return vector.data(); }
	};

	/// @brief struct to hold all resources
	// TODO: rename to staged resources buffers or something like that
	struct ResourcesCollection
	{
		// auto-submission level 1 buffers (mainObj that drawObjs references, if all drawObjs+idxBuffer+geometryInfo doesn't fit into mem this will be broken down into many)
		//CPUGeneratedResource<MainObject> mainObjects;

		// auto-submission level 2 buffers
		CPUGeneratedResource<DrawObject> drawObjects;
		CPUGeneratedResource<uint32_t> indexBuffer; // TODO: this is going to change to ReservedComputeResource where index buffer gets filled by compute shaders
		CPUGeneratedResource<uint8_t> geometryInfo; // general purpose byte buffer for custom data for geometries (eg. line points, bezier definitions, aabbs)

		// Get Total memory consumption, If all ResourcesCollection get packed together with GPUStructsMaxNaturalAlignment
		// used to decide the remaining memory and when to overflow
		size_t calculateTotalConsumption() const
		{
			return
				drawObjects.getAlignedStorageSize() +
				indexBuffer.getAlignedStorageSize() +
				geometryInfo.getAlignedStorageSize();
		}
	};
	
	DrawResourcesFiller();

	DrawResourcesFiller(smart_refctd_ptr<video::ILogicalDevice>&& device, smart_refctd_ptr<IUtilities>&& bufferUploadUtils, IQueue* copyQueue, core::smart_refctd_ptr<system::ILogger>&& logger);

	typedef std::function<void(SIntendedSubmitInfo&)> SubmitFunc;
	void setSubmitDrawsFunction(const SubmitFunc& func);

	// Must be called at the end of each frame.
	// right before submitting the main draw that uses the currently queued geometry, images, or other objects/resources.
	// Registers the semaphore/value that will signal completion of this frame�s draw,
	// This allows future frames to safely deallocate or evict resources used in the current frame by waiting on this signal before reuse or destruction.
	// `drawSubmitWaitValue` should reference the wait value of the draw submission finishing this frame using the `intendedNextSubmit`; 
	void markFrameUsageComplete(uint64_t drawSubmitWaitValue);
	
	void drawTriangleMesh(
		const CTriangleMesh& mesh,
		SIntendedSubmitInfo& intendedNextSubmit);

	/// @brief  resets staging buffers and images
	void reset()
	{
		drawCalls.clear();
	}

	/// @brief collection of all the resources that will eventually be reserved or copied to in the resourcesGPUBuffer, will be accessed via individual BDA pointers in shaders
	const ResourcesCollection& getResourcesCollection() const { return resourcesCollection; }
	/// @brief buffer containing all non-texture type resources
	nbl::core::smart_refctd_ptr<IGPUBuffer> getResourcesGPUBuffer() const { return resourcesGPUBuffer; }
	/// @return how far resourcesGPUBuffer was copied to by `finalizeAllCopiesToGPU` in `resourcesCollection` 
	const size_t getCopiedResourcesSize() { return copiedResourcesSize; }
	const core::vector<DrawCallData>& getDrawCalls() const { return drawCalls; }

private:
	/**
	 * @brief Attempts to allocate a single contiguous device-local memory block for draw resources, divided into image and buffer sections.
	 *
	 * The function allocates a single memory block and splits it into image and buffer arenas.
	 *
	 * @param logicalDevice Pointer to the logical device used for memory allocation and resource creation.
	 * @param requiredImageMemorySize The size in bytes of the memory required for images.
	 * @param requiredBufferMemorySize The size in bytes of the memory required for buffers.
	 * @param memoryTypeIndexTryOrder Ordered list of memory type indices to attempt allocation with, in the order they should be tried.
	 *
	 * @return true if the memory allocation and resource setup succeeded; false otherwise.
	 */
	bool allocateDrawResources(ILogicalDevice* logicalDevice, size_t requiredImageMemorySize, size_t requiredBufferMemorySize, std::span<uint32_t> memoryTypeIndexTryOrder);

	/// @brief Records GPU copy commands for all staged buffer resourcesCollection into the active command buffer.
	bool pushBufferUploads(SIntendedSubmitInfo& intendedNextSubmit, ResourcesCollection& resourcesCollection);

private:
	nbl::system::logger_opt_smart_ptr m_logger = nullptr;

	smart_refctd_ptr<video::ILogicalDevice> m_device;
	core::smart_refctd_ptr<video::IUtilities> m_bufferUploadUtils;

	IQueue* m_copyQueue;

	// FrameIndex used as a criteria for resource/image eviction in case of limitations
	uint32_t currentFrameIndex = 0u;

	// DrawCalls Data
	core::vector<DrawCallData> drawCalls;

	// ResourcesCollection and packed into GPUBuffer
	ResourcesCollection resourcesCollection;
	IDeviceMemoryAllocator::SAllocation buffersMemoryArena;
	nbl::core::smart_refctd_ptr<IGPUBuffer> resourcesGPUBuffer;
	size_t copiedResourcesSize;

	SubmitFunc submitDraws;
};