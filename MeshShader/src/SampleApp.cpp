#include "SampleApp.h"

    bool MeshSampleApp::onAppInitialized(smart_refctd_ptr<ISystem>&& system) {
        if (!asset_base_t::onAppInitialized(smart_refctd_ptr(system)))
            return false;
        if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
            return false;

        m_semaphore = m_device->createSemaphore(m_realFrameIx);
        if (!m_semaphore)
            return logFail("Failed to Create a Semaphore!");

        auto pool = m_device->createCommandPool(getGraphicsQueue()->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
        for (auto i = 0u; i<MaxFramesInFlight; i++)
        {
            if (!pool)
                return logFail("Couldn't create Command Pool!");
            if (!pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,{m_cmdBufs.data()+i,1}))
                return logFail("Couldn't create Command Buffer!");
        }
        
        const uint32_t addtionalBufferOwnershipFamilies[] = {getGraphicsQueue()->getFamilyIndex()};


        m_scene = CGeometryCreatorScene::create(
            {
                .transferQueue = getTransferUpQueue(),
                .utilities = m_utils.get(),
                .logger = m_logger.get(),
                .addtionalBufferOwnershipFamilies = addtionalBufferOwnershipFamilies
            },
            CSimpleDebugRenderer::DefaultPolygonGeometryPatch
        );

        
        // for the scene drawing pass
        {
            IGPURenderpass::SCreationParams params = {};
            const IGPURenderpass::SCreationParams::SDepthStencilAttachmentDescription depthAttachments[] = {
                {{
                    {
                        .format = sceneRenderDepthFormat,
                        .samples = IGPUImage::ESCF_1_BIT,
                        .mayAlias = false
                    },
                    /*.loadOp = */{IGPURenderpass::LOAD_OP::CLEAR},
                    /*.storeOp = */{IGPURenderpass::STORE_OP::STORE},
                    /*.initialLayout = */{IGPUImage::LAYOUT::UNDEFINED},
                    /*.finalLayout = */{IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL}
                }},
                IGPURenderpass::SCreationParams::DepthStencilAttachmentsEnd
            };
            params.depthStencilAttachments = depthAttachments;
            const IGPURenderpass::SCreationParams::SColorAttachmentDescription colorAttachments[] = {
                {{
                    {
                        .format = finalSceneRenderFormat,
                        .samples = IGPUImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT,
                        .mayAlias = false
                    },
                    /*.loadOp = */IGPURenderpass::LOAD_OP::CLEAR,
                    /*.storeOp = */IGPURenderpass::STORE_OP::STORE,
                    /*.initialLayout = */IGPUImage::LAYOUT::UNDEFINED,
                    /*.finalLayout = */ IGPUImage::LAYOUT::READ_ONLY_OPTIMAL // ImGUI shall read
                }},
                IGPURenderpass::SCreationParams::ColorAttachmentsEnd
            };
            params.colorAttachments = colorAttachments;
            IGPURenderpass::SCreationParams::SSubpassDescription subpasses[] = {
                {},
                IGPURenderpass::SCreationParams::SubpassesEnd
            };
            subpasses[0].depthStencilAttachment = {{.render={.attachmentIndex=0,.layout=IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL}}};
            subpasses[0].colorAttachments[0] = {.render={.attachmentIndex=0,.layout=IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL}};
            params.subpasses = subpasses;
            params.dependencies = {};
            m_renderpass = m_device->createRenderpass(std::move(params));
            if (!m_renderpass)
                return logFail("Failed to create Scene Renderpass!");
        }

        const auto& geometries = m_scene->getInitParams().geometries;
        m_renderer = MeshDebugRenderer::create(m_assetMgr.get(), m_renderpass.get(), 0, { &geometries.front().get(),geometries.size() });

        // Create ImGUI
        {
            auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
            ext::imgui::UI::SCreationParameters params = {};
            params.resources.texturesInfo = {.setIx=0u,.bindingIx=TexturesImGUIBindingIndex};
            params.resources.samplersInfo = {.setIx=0u,.bindingIx=1u};

            params.utilities = m_utils;
            params.transfer = getTransferUpQueue();
            params.pipelineLayout = ext::imgui::UI::createDefaultPipelineLayout(m_utils->getLogicalDevice(),params.resources.texturesInfo,params.resources.samplersInfo,MaxImGUITextures);
            params.assetManager = make_smart_refctd_ptr<IAssetManager>(smart_refctd_ptr(m_system));
            params.renderpass = smart_refctd_ptr<IGPURenderpass>(scRes->getRenderpass());
            params.subpassIx = 0u;
            params.pipelineCache = nullptr;
            interface.imGUI = ext::imgui::UI::create(std::move(params));
            if (!interface.imGUI) {
                return logFail("Failed to create `nbl::ext::imgui::UI` class");
            }
        }

        // create rest of User Interface
        {
            auto* imgui = interface.imGUI.get();
            // create the suballocated descriptor set
            {
                // note that we use default layout provided by our extension, but you are free to create your own by filling ext::imgui::UI::S_CREATION_PARAMETERS::resources
                const auto* layout = imgui->getPipeline()->getLayout()->getDescriptorSetLayout(0u);
                auto pool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::E_CREATE_FLAGS::ECF_UPDATE_AFTER_BIND_BIT,{&layout,1});
                auto ds = pool->createDescriptorSet(smart_refctd_ptr<const IGPUDescriptorSetLayout>(layout));
                if (ds) {
                    interface.subAllocDS = make_smart_refctd_ptr<SubAllocatedDescriptorSet>(std::move(ds));
                }
                else {
                    interface.subAllocDS = nullptr;
                }
                if (!interface.subAllocDS) {
                    return logFail("Failed to create the descriptor set");
                }
                // make sure Texture Atlas slot is taken for eternity
                {
                    auto dummy = SubAllocatedDescriptorSet::invalid_value;
                    interface.subAllocDS->multi_allocate(0,1,&dummy);
                    assert(dummy==ext::imgui::UI::FontAtlasTexId); //?
                }
                // write constant descriptors, note we don't create info & write pair for the samplers because UI extension's are immutable and baked into DS layout
                IGPUDescriptorSet::SDescriptorInfo info = {};
                info.desc = smart_refctd_ptr<nbl::video::IGPUImageView>(interface.imGUI->getFontAtlasView());
                info.info.image.imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL;
                const IGPUDescriptorSet::SWriteDescriptorSet write = {
                    .dstSet = interface.subAllocDS->getDescriptorSet(),
                    .binding = TexturesImGUIBindingIndex,
                    .arrayElement = ext::imgui::UI::FontAtlasTexId,
                    .count = 1,
                    .info = &info
                };
                if (!m_device->updateDescriptorSets({&write,1},{}))
                    return logFail("Failed to write the descriptor set");
            }



            nbl::video::IGPUBuffer::SCreationParams gpubuff_params = {};
            gpubuff_params.size = sizeof(MeshletObjectData) * MeshDataBuffer::MaxObjectCount + sizeof(hlsl::float32_t4x4) * MeshDataBuffer::MaxInstanceCount;
            // While the usages on `ICPUBuffers` are mere hints to our automated CPU-to-GPU conversion systems which need to be patched up anyway,
            // the usages on an `IGPUBuffer` are crucial to specify correctly.
            gpubuff_params.usage = IGPUBuffer::EUF_UNIFORM_BUFFER_BIT;
            meshGPUBuffer = m_device->createBuffer(std::move(gpubuff_params));
            meshGPUBuffer->setObjectDebugName("mesh data buffer");

            nbl::video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = meshGPUBuffer->getMemoryReqs();
            // you can simply constrain the memory requirements by AND-ing the type bits of the host visible memory types
            reqs.memoryTypeBits &= m_device->getPhysicalDevice()->getHostVisibleMemoryTypeBits();
            mesh_allocation = m_device->allocate(reqs, meshGPUBuffer.get(), nbl::video::IDeviceMemoryAllocation::EMAF_NONE);
            if (!mesh_allocation.isValid()) {
                return logFail("failed to allocate device memory");
            }
            assert(meshGPUBuffer->getBoundMemory().memory == mesh_allocation.memory.get());



// This is a cool utility you can use instead of counting up how much of each descriptor type you need to N_i allocate descriptor sets with layout L_i from a single pool
            smart_refctd_ptr<nbl::video::IDescriptorPool> pool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE, { &m_renderer->mesh_layout.get(),1 });

            //i dont really want to move the layout but it seems like im at the mercy of the compiler
            auto layout_smart_ptr_copy = m_renderer->mesh_layout; 
            m_renderer->getInitParams().meshDescriptor = pool->createDescriptorSet(layout_smart_ptr_copy);
            m_renderer->getInitParams().meshDescriptor->setObjectDebugName("mesh descriptor");
            {
                IGPUDescriptorSet::SDescriptorInfo info[1];
                info[0].desc = smart_refctd_ptr(meshGPUBuffer); // bad API, too late to change, should just take raw-pointers since not consumed
                info[0].info.buffer = { .offset = 0,.size = gpubuff_params.size };
                IGPUDescriptorSet::SWriteDescriptorSet writes[1] = {
                    {
                        .dstSet = m_renderer->getInitParams().meshDescriptor.get(),
                        .binding = 0,
                        .arrayElement = 0,
                        .count = 1,
                        .info = info
                    }
                };
                m_device->updateDescriptorSets(writes, {});
            }

            interface.meshMemoryRange.memory = mesh_allocation.memory.get();
            interface.meshMemoryRange.offset = 0;
            interface.meshMemoryRange.length = mesh_allocation.memory->getAllocationSize();
            interface.meshMemoryRange.range = { interface.meshMemoryRange.offset, interface.meshMemoryRange.length };
            if (!mesh_allocation.memory->map(interface.meshMemoryRange.range, IDeviceMemoryAllocation::EMCAF_WRITE)) {
                return logFail("failed to map device memory");
            }
            interface.mesh_mapped_memory = mesh_allocation.memory->getMappedPointer();
            if (!interface.mesh_mapped_memory) {
                return logFail("failed to map device memory");
            }

            memcpy(interface.mesh_mapped_memory, m_renderer->m_geoms.meshData, sizeof(MeshletObjectData) * MeshDataBuffer::MaxObjectCount);

            imgui->registerListener([this](){interface();});

        }
        
        interface.objectNames = {
            "Cube",
            "Rectangle",
            "Disk",
            "Sphere",
            "Cylinder",
            "Cone",
            "Icosphere"
            //magicenum reflection?
        };

        const hlsl::matrix<float, 4, 4> fillVal{
            1.f, 0.f, 0.f, 0.f,
            0.f, 1.f, 0.f, 0.f,
            0.f, 0.f, 1.f, 0.f,
            0.f, 0.f, 0.f, 1.f
        };
        interface.transforms.fill(fillVal);

        //load up the ICPUGeometry, then convert it to GPU geometry

        interface.camera.mapKeysToArrows();

        onAppInitializedFinish();
        return true;
    }

   

    bool MeshSampleApp::onAppTerminated() {
        SubAllocatedDescriptorSet::value_type fontAtlasDescIx = ext::imgui::UI::FontAtlasTexId;
        IGPUDescriptorSet::SDropDescriptorSet dummy[1];
        interface.subAllocDS->multi_deallocate(dummy,TexturesImGUIBindingIndex,1,&fontAtlasDescIx);
        return device_base_t::onAppTerminated();
    }

    IQueue::SSubmitInfo::SSemaphoreInfo MeshSampleApp::renderFrame(const std::chrono::microseconds nextPresentationTimestamp) {
        // CPU events
        update(nextPresentationTimestamp);

        const auto& virtualWindowRes = interface.sceneResolution;
        if (!m_framebuffer || m_framebuffer->getCreationParameters().width!=virtualWindowRes[0] || m_framebuffer->getCreationParameters().height!=virtualWindowRes[1])
            recreateFramebuffer(virtualWindowRes);

        const auto resourceIx = m_realFrameIx % MaxFramesInFlight;

        auto* const cb = m_cmdBufs.data()[resourceIx].get();
        cb->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
        cb->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
        // clear to black for both things
        const IGPUCommandBuffer::SClearColorValue clearValue = { .float32 = {0.f,0.f,0.f,1.f} };
        if (m_framebuffer)
        {
            cb->beginDebugMarker("UISampleApp Scene Frame");
            {
                const IGPUCommandBuffer::SClearDepthStencilValue farValue = { .depth = 0.f };
                const IGPUCommandBuffer::SRenderpassBeginInfo renderpassInfo{
                    .framebuffer = m_framebuffer.get(),
                    .colorClearValues = &clearValue,
                    .depthStencilClearValues = &farValue,
                    .renderArea = {
                        .offset = {0,0},
                        .extent = {virtualWindowRes[0],virtualWindowRes[1]}
                    }
                };
                beginRenderpass(cb, renderpassInfo);
            }
            // draw scene
            UpdateScene(cb);
            cb->endRenderPass();
            cb->endDebugMarker();
        }
        {
            cb->beginDebugMarker("UISampleApp IMGUI Frame");
            { //begin imgui subpass
                auto scRes = static_cast<CDefaultSwapchainFramebuffers*>(m_surface->getSwapchainResources());
                const IGPUCommandBuffer::SRenderpassBeginInfo renderpassInfo = {
                    .framebuffer = scRes->getFramebuffer(device_base_t::getCurrentAcquire().imageIndex),
                    .colorClearValues = &clearValue,
                    .depthStencilClearValues = nullptr,
                    .renderArea = {
                        .offset = {0,0},
                        .extent = {m_window->getWidth(),m_window->getHeight()}
                    }
                };
                beginRenderpass(cb, renderpassInfo);
            }
            // draw ImGUI
            {
                auto* imgui = interface.imGUI.get();
                auto* pipeline = imgui->getPipeline();
                cb->bindGraphicsPipeline(pipeline);
                // note that we use default UI pipeline layout where uiParams.resources.textures.setIx == uiParams.resources.samplers.setIx
                const auto* ds = interface.subAllocDS->getDescriptorSet();
                cb->bindDescriptorSets(EPBP_GRAPHICS,pipeline->getLayout(),imgui->getCreationParameters().resources.texturesInfo.setIx,1u,&ds);
                // a timepoint in the future to release streaming resources for geometry
                const ISemaphore::SWaitInfo drawFinished = {.semaphore=m_semaphore.get(),.value=m_realFrameIx+1u};
                if (!imgui->render(cb,drawFinished))
                {
                    m_logger->log("TODO: need to present acquired image before bailing because its already acquired.",ILogger::ELL_ERROR);
                    return {};
                }
            }
            cb->endRenderPass();
            cb->endDebugMarker();
        }
        cb->end();

        IQueue::SSubmitInfo::SSemaphoreInfo retval =
        {
            .semaphore = m_semaphore.get(),
            .value = ++m_realFrameIx,
            .stageMask = PIPELINE_STAGE_FLAGS::ALL_GRAPHICS_BITS
        };
        const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[] =
        {
            {.cmdbuf = cb }
        };
        const IQueue::SSubmitInfo::SSemaphoreInfo acquired[] = {
            {
                .semaphore = device_base_t::getCurrentAcquire().semaphore,
                .value = device_base_t::getCurrentAcquire().acquireCount,
                .stageMask = PIPELINE_STAGE_FLAGS::NONE
            }
        };
        const IQueue::SSubmitInfo infos[] =
        {
            {
                .waitSemaphores = acquired,
                .commandBuffers = commandBuffers,
                .signalSemaphores = {&retval,1}
            }
        };
        
        if (getGraphicsQueue()->submit(infos) != IQueue::RESULT::SUCCESS)
        {
            retval.semaphore = nullptr; // so that we don't wait on semaphore that will never signal
            m_realFrameIx--;
        }


        m_window->setCaption("[Nabla Engine] Mesh Shader Demo");
        return retval;
    }

    void MeshSampleApp::UpdateDescriptor() {
        m_renderer.get()->getInitParams().subAllocDS;
    }

    const video::IGPURenderpass::SCreationParams::SSubpassDependency* MeshSampleApp::getDefaultSubpassDependencies() const {
        // Subsequent submits don't wait for each other, but they wait for acquire and get waited on by present
        const static IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] = {
            // don't want any writes to be available, we'll clear, only thing to worry about is the layout transition
            {
                .srcSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
                .dstSubpass = 0,
                .memoryBarrier = {
                    .srcStageMask = PIPELINE_STAGE_FLAGS::NONE, // should sync against the semaphore wait anyway 
                    .srcAccessMask = ACCESS_FLAGS::NONE,
                    // layout transition needs to finish before the color write
                    .dstStageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
                    .dstAccessMask = ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
                }
                // leave view offsets and flags default
            },
            // want layout transition to begin after all color output is done
            {
                .srcSubpass = 0,
                .dstSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
                .memoryBarrier = {
                    // last place where the color can get modified, depth is implicitly earlier
                    .srcStageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
                    // only write ops, reads can't be made available
                    .srcAccessMask = ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
                    // spec says nothing is needed when presentation is the destination
                }
                // leave view offsets and flags default
            },
            IGPURenderpass::SCreationParams::DependenciesEnd
        };
        return dependencies;
    }


    void MeshSampleApp::UpdateScene(nbl::video::IGPUCommandBuffer* cb) {
        float32_t4x4 viewProjMatrix;
        // TODO: get rid of legacy matrices //<-- camera.getViewMatrix returns matrix3x4SIMD
        {
            const auto& camera = interface.camera;
            memcpy(&viewProjMatrix, camera.getConcatenatedMatrix().pointer(), sizeof(viewProjMatrix));
        }
        if (interface.transposeCameraViewProj) {
            viewProjMatrix = hlsl::transpose(viewProjMatrix);
        }
        const auto viewParams = MeshDebugRenderer::SViewParams(viewProjMatrix, interface.objectCount);

        m_renderer->render(cb, viewParams);
    }


    void MeshSampleApp::update(const std::chrono::microseconds nextPresentationTimestamp)
    {
        auto& camera = interface.camera;
        camera.setMoveSpeed(interface.moveSpeed);
        camera.setRotateSpeed(interface.rotateSpeed);


        m_inputSystem->getDefaultMouse(&mouse);
        m_inputSystem->getDefaultKeyboard(&keyboard);

        struct
        {
            std::vector<SMouseEvent> mouse{};
            std::vector<SKeyboardEvent> keyboard{};
        } uiEvents;

        // TODO: should be a member really
        static std::chrono::microseconds previousEventTimestamp{};

        // I think begin/end should always be called on camera, just events shouldn't be fed, why?
        // If you stop begin/end, whatever keys were up/down get their up/down values frozen leading to
        // `perActionDt` becoming obnoxiously large the first time the even processing resumes due to
        // `timeDiff` being computed since `lastVirtualUpTimeStamp` 
        camera.beginInputProcessing(nextPresentationTimestamp);
        {
            mouse.consumeEvents([&](const IMouseEventChannel::range_t& events) -> void
                {
                    if (interface.move)
                        camera.mouseProcess(events); // don't capture the events, only let camera handle them with its impl

                    for (const auto& e : events) // here capture
                    {
                        if (e.timeStamp < previousEventTimestamp)
                            continue;

                        previousEventTimestamp = e.timeStamp;
                        uiEvents.mouse.emplace_back(e);

                        if (e.type==nbl::ui::SMouseEvent::EET_SCROLL)
                        {
                            interface.gcIndex += int16_t(core::sign(e.scrollEvent.verticalScroll));
                        }
                    }
                },
                m_logger.get()
            );
            keyboard.consumeEvents([&](const IKeyboardEventChannel::range_t& events) -> void
                {
                    if (interface.move)
                        camera.keyboardProcess(events); // don't capture the events, only let camera handle them with its impl

                    for (const auto& e : events) // here capture
                    {
                        if (e.timeStamp < previousEventTimestamp)
                            continue;

                        previousEventTimestamp = e.timeStamp;
                        uiEvents.keyboard.emplace_back(e);
                    }
                },
                m_logger.get()
            );
        }
        camera.endInputProcessing(nextPresentationTimestamp);

        const auto cursorPosition = m_window->getCursorControl()->getPosition();

        ext::imgui::UI::SUpdateParameters params = 
        {
            .mousePosition = float32_t2(cursorPosition.x,cursorPosition.y) - float32_t2(m_window->getX(),m_window->getY()),
            .displaySize = {m_window->getWidth(),m_window->getHeight()},
            .mouseEvents = uiEvents.mouse,
            .keyboardEvents = uiEvents.keyboard
        };

        interface.imGUI->update(params);



        auto* countMem = reinterpret_cast<MeshletObjectData*>(interface.mesh_mapped_memory);
        //i only need to set the meslet object data once on initialization
        //memcpy(countMem, interface.objectCount.data(), sizeof(MeshletObjectData) * MeshDataBuffer::MaxObjectCount);
        countMem += MeshDataBuffer::MaxObjectCount;
        auto* matrixMem = reinterpret_cast<hlsl::matrix<float, 4, 4>*>(countMem);
        memcpy(matrixMem, interface.transforms.data(), interface.transforms.size() * sizeof(hlsl::matrix<float, 4, 4>));
        m_device->flushMappedMemoryRanges(1, &interface.meshMemoryRange);
    }

    void MeshSampleApp::recreateFramebuffer(const uint16_t2 resolution)
    {
        auto createImageAndView = [&](E_FORMAT format)->smart_refctd_ptr<IGPUImageView>
        {
            auto image = m_device->createImage({{
                .type = IGPUImage::ET_2D,
                .samples = IGPUImage::ESCF_1_BIT,
                .format = format,
                .extent = {resolution.x,resolution.y,1},
                .mipLevels = 1,
                .arrayLayers = 1,
                .usage = IGPUImage::EUF_RENDER_ATTACHMENT_BIT|IGPUImage::EUF_SAMPLED_BIT
            }});
            if (!m_device->allocate(image->getMemoryReqs(),image.get()).isValid())
                return nullptr;
            IGPUImageView::SCreationParams params = {
                .image = std::move(image),
                .viewType = IGPUImageView::ET_2D,
                .format = format
            };
            params.subresourceRange.aspectMask = isDepthOrStencilFormat(format) ? IGPUImage::EAF_DEPTH_BIT:IGPUImage::EAF_COLOR_BIT;
            return m_device->createImageView(std::move(params));
        };
        
        smart_refctd_ptr<IGPUImageView> colorView;
        // detect window minimization
        if (resolution.x<0x4000 && resolution.y<0x4000)
        {
            colorView = createImageAndView(finalSceneRenderFormat);
            auto depthView = createImageAndView(sceneRenderDepthFormat);
            m_framebuffer = m_device->createFramebuffer(
                { 
                    {
                        .renderpass = m_renderpass,
                        .depthStencilAttachments = &depthView.get(),
                        .colorAttachments = &colorView.get(),
                        .width = resolution.x,
                        .height = resolution.y
                    }   
                }
            );
        }
        else {
            m_framebuffer = nullptr;
        }
        // release previous slot and its image
        interface.subAllocDS->multi_deallocate(0,1,&interface.renderColorViewDescIndex,{.semaphore=m_semaphore.get(),.value=m_realFrameIx});
        //
        if (colorView)
        {
            interface.subAllocDS->multi_allocate(0,1,&interface.renderColorViewDescIndex);
            // update descriptor set
            IGPUDescriptorSet::SDescriptorInfo info = {};
            info.desc = colorView;
            info.info.image.imageLayout = IGPUImage::LAYOUT::READ_ONLY_OPTIMAL;
            const IGPUDescriptorSet::SWriteDescriptorSet write = {
                .dstSet = interface.subAllocDS->getDescriptorSet(),
                .binding = TexturesImGUIBindingIndex,
                .arrayElement = interface.renderColorViewDescIndex,
                .count = 1,
                .info = &info
            };
            m_device->updateDescriptorSets({&write,1},{});
        }
        interface.transformParams.sceneTexDescIx = interface.renderColorViewDescIndex;
    }

    void MeshSampleApp::beginRenderpass(IGPUCommandBuffer* cb, const IGPUCommandBuffer::SRenderpassBeginInfo& info)
    {
        cb->beginRenderPass(info,IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
        cb->setScissor(0,1,&info.renderArea);
        const SViewport viewport = {
            .x = 0,
            .y = 0,
            .width = static_cast<float>(info.renderArea.extent.width),
            .height = static_cast<float>(info.renderArea.extent.height)
        };
        cb->setViewport(0u,1u,&viewport);
    }

    auto addMatrixTable = [&](const char* topText, const char* tableName, const int rows, const int columns, const float* pointer, const bool withSeparator = true)
        {
            ImGui::Text(topText);
            if (ImGui::BeginTable(tableName, columns))
            {
                for (int y = 0; y < rows; ++y)
                {
                    ImGui::TableNextRow();
                    for (int x = 0; x < columns; ++x)
                    {
                        ImGui::TableSetColumnIndex(x);
                        ImGui::Text("%.3f", *(pointer + (y * columns) + x));
                    }
                }
                ImGui::EndTable();
            }

            if (withSeparator)
                ImGui::Separator();
        };

    void MeshSampleApp::CInterface::DrawMeshControls() {
        //this was for learning hlsl, given the shader takes like 2 minutes to compile, might as well just relaunch
        //if (ImGui::Button("reload mesh shader")) {
            //printf("test shader result - %d\n", CreateTestShaderFuncPtr());
        //}

        ImGui::DragInt("current transform editting", &currentTransform, 1, 0, MeshDataBuffer::MaxObjectCount * MeshDataBuffer::MaxInstanceCount);

        for (uint8_t i = 0; i < objectNames.size(); i++) {
            const std::string objNameWithCount = objectNames[i] + " {" + std::to_string(objectCount[i]) + '}';
            if (ImGui::TreeNode(objNameWithCount.c_str())) {
                const std::string objCountDraggerName = ("object count ##") + objectNames[i];

                int imguiCopy = objectCount[i];
                //ImGui::DragInt(objCountDraggerName.c_str(), &imguiCopy, 1, 0, localMax);
                ImGui::SliderInt(objCountDraggerName.c_str(), &imguiCopy, 0, MeshDataBuffer::MaxInstanceCount);
                objectCount[i] = imguiCopy;

                for (uint64_t j = 0; j < objectCount[i]; j++) {
                    const std::string treeName = std::string("transform[") + std::to_string(j) + "]##" + objectNames[i];
                    if (ImGui::TreeNode(treeName.c_str())) {
                        const std::size_t transformIndex = i * MeshDataBuffer::MaxInstanceCount + j;
                        addMatrixTable("model", "", 4, 4, &transforms[transformIndex][0][0]);

                        //imguizmo overwrites these changes
                        //for (uint8_t x = 0; x < 4; x++) {
                        //    const std::size_t rowIndex = transformIndex * 4 + x;
                        //    const std::string rowName = std::string("##") + std::to_string(rowIndex);
                        //    ImGui::DragFloat4(rowName.c_str(), &transforms[transformIndex][x][0], 0.1f, -100.f, 100.f);
                        //}
                        ImGui::TreePop();
                    }
                }

                ImGui::TreePop();
            }
        }
    }

    void MeshSampleApp::CInterface::DrawCameraControls() {
        ImGuiIO& io = ImGui::GetIO();

        ImGui::Text("Camera");
        bool viewDirty = false;

        ImGui::Checkbox("transpose view proj (holy space intermixing)", &transposeCameraViewProj);

        if (ImGui::RadioButton("LH", isLH))
            isLH = true;

        ImGui::SameLine();

        if (ImGui::RadioButton("RH", !isLH))
            isLH = false;

        if (ImGui::RadioButton("Perspective", isPerspective))
            isPerspective = true;

        ImGui::SameLine();

        if (ImGui::RadioButton("Orthographic", !isPerspective))
            isPerspective = false;

        ImGui::Checkbox("Enable \"view manipulate\"", &transformParams.enableViewManipulate);
        ImGui::Checkbox("Enable camera movement", &move);
        ImGui::SliderFloat("Move speed", &moveSpeed, 0.1f, 10.f);
        ImGui::SliderFloat("Rotate speed", &rotateSpeed, 0.1f, 10.f);

        // ImGui::Checkbox("Flip Gizmo's Y axis", &flipGizmoY); // let's not expose it to be changed in UI but keep the logic in case

        if (isPerspective)
            ImGui::SliderFloat("Fov", &fov, 20.f, 150.f);
        else
            ImGui::SliderFloat("Ortho width", &viewWidth, 1, 20);

        ImGui::SliderFloat("zNear", &zNear, 0.1f, zFar);
        ImGui::SliderFloat("zFar", &zFar, zNear, 10000.f);

        viewDirty |= ImGui::SliderFloat("Distance", &transformParams.camDistance, 1.f, 69.f);

        if (viewDirty || firstFrame)
        {
            core::vectorSIMDf cameraPosition(cosf(camYAngle) * cosf(camXAngle) * transformParams.camDistance, sinf(camXAngle) * transformParams.camDistance, sinf(camYAngle) * cosf(camXAngle) * transformParams.camDistance);
            core::vectorSIMDf cameraTarget(0.f, 0.f, 0.f);
            const static core::vectorSIMDf up(0.f, 1.f, 0.f);

            camera.setPosition(cameraPosition);
            camera.setTarget(cameraTarget);
            camera.setBackupUpVector(up);

            camera.recomputeViewMatrix();
        }
        firstFrame = false;

        ImGui::Text("X: %f Y: %f", io.MousePos.x, io.MousePos.y);
        if (ImGuizmo::IsUsing())
        {
            ImGui::Text("Using gizmo");
        }
        else {
            ImGui::Text(ImGuizmo::IsOver() ? "Over gizmo" : "");
            ImGui::SameLine();
            ImGui::Text(ImGuizmo::IsOver(ImGuizmo::TRANSLATE) ? "Over translate gizmo" : "");
            ImGui::SameLine();
            ImGui::Text(ImGuizmo::IsOver(ImGuizmo::ROTATE) ? "Over rotate gizmo" : "");
            ImGui::SameLine();
            ImGui::Text(ImGuizmo::IsOver(ImGuizmo::SCALE) ? "Over scale gizmo" : "");
        }

        const auto& view = camera.getViewMatrix(); // a hack, correct way would be to use inverse matrix and get position + target because now it will bring you back to last position & target when switching from gizmo move to manual move (but from manual to gizmo is ok)

        auto const& projection = camera.getProjectionMatrix();
        if (ImGui::TreeNode("camera matrices")) {
            addMatrixTable("View", "ViewMatrixTable", 3, 4, view.pointer());
            addMatrixTable("Projection", "ViewProjectionMatrixTable", 4, 4, projection.pointer(), false);
            ImGui::TreePop();
        }
    }

    void MeshSampleApp::CInterface::UpdateImguizmo() {
        /*
            * ImGuizmo expects view & perspective matrix to be column major both with 4x4 layout
            * and Nabla uses row major matricies - 3x4 matrix for view & 4x4 for projection

            *
            * the ViewManipulate final call (inside EditTransform) returns world space column major matrix for an object,
            * note it also modifies input view matrix but projection matrix is immutable
            */

    // TODO: do all computation using `hlsl::matrix` and its `hlsl::float32_tNxM` aliases
        static struct
        {
            core::matrix4SIMD view, projection, model;
        } imguizmoM16InOut;

        ImGuizmo::SetID(0u);

        imguizmoM16InOut.view = core::transpose(matrix4SIMD(camera.getViewMatrix()));
        imguizmoM16InOut.projection = core::transpose(camera.getProjectionMatrix());

        if (currentTransform < 0) {
            currentTransform = 0;
        }

        if (currentTransform >= 0 && currentTransform < transforms.size()) {
            //auto transposedTemp = core::transpose(transforms[currentTransform]);
            //the model is a double matrix, so a memcpy or reinterpret wont work
            //i might have the x and y backwards it doesnt matter as long as its x:x and y:y
            //skipping the transform from example 61
            for (uint8_t x = 0; x < 4; x++) {
                for (uint8_t y = 0; y < 4; y++) {
                    imguizmoM16InOut.model[x][y] = transforms[currentTransform][x][y];
                }
            }
        }
        {
            transformParams.editTransformDecomposition = true;
            static TransformWidget transformWidget{};
            const auto tempForConversion = transformWidget.Update(imguizmoM16InOut.view.pointer(), imguizmoM16InOut.projection.pointer(), imguizmoM16InOut.model.pointer(), transformParams);
            sceneResolution = { tempForConversion.x, tempForConversion.y };

        }

        if (currentTransform >= 0 && currentTransform < transforms.size()) {
            for (uint8_t x = 0; x < 4; x++) {
                for (uint8_t y = 0; y < 4; y++) {
                    //tranposed
                    transforms[currentTransform][x][y] = imguizmoM16InOut.model[x][y];
                }
            }
        }
        const auto& view = camera.getViewMatrix();
        const_cast<core::matrix3x4SIMD&>(view) = core::transpose(imguizmoM16InOut.view).extractSub3x4(); // a hack, correct way would be to use inverse matrix and get position + target because now it will bring you back to last position & target when switching from gizmo move to manual move (but from manual to gizmo is ok)
        const auto& projection = camera.getProjectionMatrix();
        camera.setProjectionMatrix(projection); //this recalcs viewproj

    }

    void MeshSampleApp::CInterface::operator()() {
        ImGuiIO& io = ImGui::GetIO();
        //io.ConfigDebugIsDebuggerPresent = true;

        //camera
        matrix4SIMD projection;
        {
            const float viewHeight = viewWidth * io.DisplaySize.x / io.DisplaySize.y;

            if (isPerspective) {
                if (isLH) {
                    projection = matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(fov), viewHeight, zNear, zFar);
                }
                else {
                    projection = matrix4SIMD::buildProjectionMatrixPerspectiveFovRH(core::radians(fov), viewHeight, zNear, zFar);
                }
            }
            else
            {
                if (isLH) {
                    projection = matrix4SIMD::buildProjectionMatrixOrthoLH(viewWidth, 1.f / viewHeight, zNear, zFar);
                }
                else {
                    projection = matrix4SIMD::buildProjectionMatrixOrthoRH(viewWidth, 1.f / viewHeight, zNear, zFar);
                }
            }
            camera.setProjectionMatrix(projection);
        } //end camera
        
        ImGuizmo::SetOrthographic(false);
        ImGuizmo::BeginFrame();
        

        ImGui::SetNextWindowPos(ImVec2(1024, 100), ImGuiCond_Appearing);
        ImGui::SetNextWindowSize(ImVec2(256, 256), ImGuiCond_Appearing);

        // create a window and insert the inspector
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Appearing);
        ImGui::SetNextWindowSize(ImVec2(320, 340), ImGuiCond_Appearing);

        if (meshControlSeparated) {
            if (ImGui::Begin("mesh controls", &meshControlSeparated)) {
                meshControlSeparated = !ImGui::Button("Rejoin mesh control");
                DrawMeshControls();
            }
            ImGui::End();
        }
        if (cameraControlSeparated) {
            if (ImGui::Begin("camera controls", &cameraControlSeparated)) {
                cameraControlSeparated = !ImGui::Button("Rejoin camera control");
                DrawCameraControls();
            }
            ImGui::End();
        }
        if(ImGui::Begin("Editor")) {

            if (!meshControlSeparated) {
                meshControlSeparated = ImGui::Button("Separate mesh control");
                DrawMeshControls();
                ImGui::Separator();
            }

            if (!cameraControlSeparated) {
                cameraControlSeparated = ImGui::Button("Separate camera controls");
                DrawCameraControls();
                ImGui::Separator();
            }

            ImGui::Checkbox("update guizmo", &guizmoEnabled);
            UpdateImguizmo();
        } //end editor window
        ImGui::End();
    }
