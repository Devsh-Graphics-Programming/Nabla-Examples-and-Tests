## 9/30/2025 - GDBobby
Here's the current plan, front to back

1. Remove all unnecessary parts from my copy of example 61.

    1.1 figure out what IS necessary.
    
    1.2 trace the graphics pipeline used, so I can figure out how the mesh pipeline should look

2. i dont have much experience with viewports and scissors yet, so I'd like to change
    how the imgui viewport is handled just for the fun of it. 61 mentions it's rendered to a
    temporary color attachment which is then sourced as a texture in imgui. id like to change it so
    that imgui literally just puts a box around a viewport thats rendered to directly

3. Create the Mesh Pipeline.

    3.1. I want to support generative (procedural) mesh shaders, which take 0 input vertices
    
    3.2. I want to support meshlets - small meshes that are defined by pre-existing vertices
    
    3.3. I want to re-compile the mesh shader into a compute and vertex shader combo, 
        which can be used on machines that don't support the mesh shader extension 
        (mostly GPUs older than 2016)


I think, to prevent controlling two different branches in two different repos, I'll stuff everything into this example in the beginning. 
Once everything start to come together, I'll start moving things, like the Mesh Pipeline class, into more appropriate places, like Nabla itself.


## 9/31
I'll create a mesh shader tomorrow. I don't really know what to do yet but I'll start with procedural gen.

I think I'll also make a different pipeline object that supports the geometry from example 61?

I had my fun with viewports. idk what i expected tbh

I need to search a little deeper in the spec for other mesh pipeline related rules. I need to research subpasses as well.


## 10/3
Beginning shader experimentation. Setting up easy reload of shaders so I don't have to relaunch every test iteration.

## 10/6
I need to add CPU side verification that mesh shader vert and prim count are below vulkan limits. The same as how work group size is verified.

Mesh and Task shaders having branches where the output is not defined is incorrect. The glslc compiler won't warn the user, I'll have to check for DXC. Nvidia will assume it's a 0 group output, but AMD will get DEVICE_LOST. If it's possible, having a warning or compile check for that would be nice. Most likely outside the scope of Nabla, but possibly not. I'll have to ask.

## 10/9
On the bug hunting phase. Should be finished shortly, then I'll hit the cleanup phase.