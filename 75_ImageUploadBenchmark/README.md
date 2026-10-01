# Image Upload Benchmark

Measures the throughput of uploading 128x128 pixel tiles into a 2048x2048 `EF_R8G8B8A8_UNORM` image with `OPTIMAL` tiling in `DEVICE_LOCAL` memory. Every permutation moves the same payload: 256 tiles (16 MiB) per frame, 1000 frames, 15.6 GiB total. 

## Benchmark permutations

Staging memory types:

- **SysRAM**: `HOST_VISIBLE`, not `DEVICE_LOCAL`, not `HOST_CACHED`. Write combined system RAM, the GPU pulls from it over PCIe.
- **BAR/VRAM**: `HOST_VISIBLE` and `DEVICE_LOCAL`, not `HOST_CACHED`. The CPU memcpys directly into GPU device-local RAM, and the GPU then reads local VRAM. Depending on the device this memory type can be an integrated GPU's unified memory, the whole VRAM exposed through resizable BAR, or a small BAR staging window. Skipped when no such memory type exists.

Upload methods:

### CopyBufferToImage

The CPU memcpys the frame's 16 MiB into its staging partition, records one `copyBufferToImage` with 256 regions (one per tile), and submits. The copy engine converts linear rows into optimal tiling.

### Snake Compute

A compute dispatch replaces the copy command. Each thread loads one pixel from the staging buffer through BDA (`vk::RawBufferLoad`) and stores it to the image bound as a storage texture. Tiles are packed linearly in row major order in the buffer; the thread's global coordinate is decomposed into (tile index, position in tile) with shifts and masks since every dimension is a power of two. Destination tile coordinates come from a small lookup buffer of packed (x,y) pairs. Workgroup size 128x4.

### Morton Compute

Same read side, but each 16x16 workgroup handles one 16x16 block of a tile and the local write position is a Morton (Z-order) decode of the thread's 8 bit linear index. Writes then follow a swizzled pattern closer to how GPUs lay out optimal tiling. Workgroup size 16x16.

### HostImageCopy

The `VK_EXT_host_image_copy` path, run only when the `hostImageCopy` limit is enabled. `ILogicalDevice::copyMemoryToImage` with `EHICF_NONE` and the same 256 tile regions per call, reading straight from a CPU vector. No staging buffer, no command buffer, no queue: the driver's CPU code reads the linear tiles and swizzles them into optimal tiling. The image is created with `EUF_HOST_TRANSFER_BIT` and transitioned to `GENERAL` once on the host via `transitionImageLayout`.

### HostImageCopy (MEMCPY)

Same entry point with `EHICF_MEMCPY_BIT` and a single region spanning the whole subresource. The host buffer already holds the image's raw optimal tiling bytes (see below), so the driver performs a flat memcpy with no swizzle. The spec requires MEMCPY copies to cover the full subresource with zero `imageOffset` and zero `memoryRowLength`/`memoryImageHeight`.

## How the numbers are computed

The queue permutations run a frames-in-flight loop:

- 4 command pools/buffers and one timeline semaphore. Before reusing a slot the CPU blocks until the submit from 4 frames ago completes, so at most 4 frames are in flight.
- The 64 MiB staging buffer is split into 4 partitions of 16 MiB. Frame N writes partition N mod 4 while older frames still execute.
- Per frame: wait for the slot, memcpy the source data into the partition (plus a flush on non-coherent memory), record barriers and the copy or dispatch, submit. Each phase is bracketed with `high_resolution_clock` and the per-phase sums print as the Wait/Memcpy/Record/Submit breakdown.

Columns of the results table:

- **Wall GB/s** ("CPU submit throughput"): total bytes over wall time from loop start until right after the last submit, not until the GPU drains. This measures the sustained rate at which the CPU can feed the queue.
- **GPU GB/s** ("GPU only throughput"): `writeTimestamp` before and after the copy/dispatch in every command buffer. After the loop the timestamps of the last 4 frames are read back, averaged per frame, and scaled to 1000 frames ("GPU time (extrapolated)").
- **Memcpy GB/s**: total bytes over the summed memcpy phase alone.

The host image copy rows involve no queue, so the GPU and Memcpy columns are zero. They time a plain loop of 1000 `copyMemoryToImage` calls after one untimed warmup call; wall time is the only number.

## MEMCPY test and the PNG dumps

The MEMCPY row needs a buffer holding the image's opaque optimal tiling bytes. Producing that buffer also gives a way to visualize the tiling:

1. A gradient (R = x/(W-1), G = y/(H-1), B = 0, A = 1, packed with `packUnorm4x8`) is uploaded with `EHICF_NONE`, so the image holds known content.
2. `getImageSubresourceLayout` reports `hostMemcpySize` for mip 0, the size of the raw tiled representation of the subresource. It is logged next to the linear size W\*H\*4.
3. `copyImageToMemory` with `EHICF_MEMCPY_BIT` downloads into a buffer of `hostMemcpySize` bytes. With this flag the driver skips de-swizzling and returns the bytes exactly as they sit in device memory.
4. Those bytes are re-uploaded with `EHICF_MEMCPY_BIT` 1000 times to produce the "HostImageCopy (MEMCPY)" row. Comparing it against the `EHICF_NONE` rows isolates the cost of the driver's CPU swizzle.
5. A second MEMCPY download is memcmp'd against the first and PASS/FAIL is logged, proving the timed upload reproduced the exact tiled bytes.
6. Two PNGs are written to the working directory. `writeImagePNG` wraps the bytes in an `ICPUImage` over an adopted `ICPUBuffer` (no copy), wraps that in an `ICPUImageView`, and writes it with the asset manager:
   - `hostcopy_gradient.png`: the gradient as uploaded, W x H.
   - `hostcopy_optimal_tiling_raw.png`: the MEMCPY download reinterpreted as linear RGBA8 rows, W wide and ceil(hostMemcpySize / (W\*4)) tall, zero padded at the tail. It contains the same colors as the gradient but rearranged; the rearrangement is the driver's optimal tiling pattern.
