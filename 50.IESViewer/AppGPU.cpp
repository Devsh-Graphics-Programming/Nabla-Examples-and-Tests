// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "App.hpp"

core::smart_refctd_ptr<IGPUImageView> IESViewer::createImageView(const size_t width, const size_t height, E_FORMAT format, std::string name, bitflag<IImage::E_USAGE_FLAGS> usage, bitflag<IImage::E_ASPECT_FLAGS> aspectFlags)
{
    IGPUImage::SCreationParams imageParams{};
    imageParams.type = IImage::E_TYPE::ET_2D;
    imageParams.extent.height = height;
    imageParams.extent.width = width;
    imageParams.extent.depth = 1u;
    imageParams.format = format;
    imageParams.mipLevels = 1u;
    imageParams.flags = IImage::ECF_NONE;
    imageParams.arrayLayers = 1u;
    imageParams.samples = IImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT;
    imageParams.usage = usage;

    auto image = m_device->createImage(std::move(imageParams));
    image->setObjectDebugName(name.c_str());

    if (!image)
    {
        m_logger->log("Failed to create \"%s\" image!", system::ILogger::ELL_ERROR, name.c_str());
        return nullptr;
    }

    auto allocation = m_device->allocate(image->getMemoryReqs(), image.get(), nbl::video::IDeviceMemoryAllocation::EMAF_NONE);
    if (!allocation.isValid())
    {
        m_logger->log("Failed to allocate device memory for \"%s\" image!", system::ILogger::ELL_ERROR, name.c_str());
        return nullptr;
    }

    IGPUImageView::SCreationParams viewParams{};
    viewParams.image = std::move(image);
    viewParams.format = format;
    viewParams.viewType = IGPUImageView::ET_2D;
    viewParams.flags = IImageViewBase::ECF_NONE;
    viewParams.subresourceRange.baseArrayLayer = 0u;
    viewParams.subresourceRange.baseMipLevel = 0u;
    viewParams.subresourceRange.layerCount = 1u;
    viewParams.subresourceRange.levelCount = 1u;
    viewParams.subresourceRange.aspectMask = aspectFlags;

    auto imageView = m_device->createImageView(std::move(viewParams));

    if (not imageView)
        m_logger->log("Failed to create image view for \"%s\" image!", system::ILogger::ELL_ERROR, name.c_str());

    return imageView;
}

core::smart_refctd_ptr<IGPUBuffer> IESViewer::createBuffer(const core::vector<asset::CIESProfile::IES_STORAGE_FORMAT>& in, std::string name)
{
    IGPUBuffer::SCreationParams bufferParams = {};
    bufferParams.usage = core::bitflag(asset::IBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT) | IGPUBuffer::EUF_TRANSFER_DST_BIT /*TODO: <- double check*/;;
    bufferParams.size = sizeof(asset::CIESProfile::IES_STORAGE_FORMAT) * in.size();

    auto buffer = m_device->createBuffer(std::move(bufferParams));
    buffer->setObjectDebugName(name.c_str());

    if (not buffer)
    {
        m_logger->log("Failed to create \"%s\" buffer!", ILogger::ELL_ERROR, name.c_str());
        return nullptr;
    }

    auto memoryReqs = buffer->getMemoryReqs();

    if (m_utils)
        memoryReqs.memoryTypeBits &= m_utils->getLogicalDevice()->getPhysicalDevice()->getUpStreamingMemoryTypeBits();

    auto allocation = m_device->allocate(memoryReqs, buffer.get(), core::bitflag<video::IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS>(video::IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT));
    if (not allocation.isValid())
    {
        m_logger->log("Failed to allocate \"%s\" buffer!", ILogger::ELL_ERROR, name.c_str());
        return nullptr;
    }

    auto* mappedPointer = allocation.memory->map({ 0ull, memoryReqs.size }, IDeviceMemoryAllocation::EMCAF_READ_AND_WRITE);

    if (not mappedPointer)
    {
        m_logger->log("Failed to map device memory for \"%s\" buffer!", ILogger::ELL_ERROR, name.c_str());
        return nullptr;
    }

    memcpy(mappedPointer, in.data(), buffer->getSize());

    if (not allocation.memory->unmap())
    {
        m_logger->log("Failed to unmap device memory for \"%s\" buffer!", ILogger::ELL_ERROR, name.c_str());
        return nullptr;
    }

    return buffer;
}