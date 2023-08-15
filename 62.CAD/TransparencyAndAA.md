The Algorithm used here for AntiAliasing and Transparency for 2D Lines and Curves relies on `VK_EXT_fragment_shader_interlock`; enabling this capability provides a critical section for fragment shaders to avoid overlapping pixels being processed at the same time, and certain guarantees about the ordering of fragment shader invocations of fragments of overlapping pixels.

Such a guarantee is useful for applications like blending in the fragment shader, where an application requires that fragment values to be composited in the framebuffer in primitive order.

For example Programmable blending operations in the fragment shader, where the destination buffer is read via image loads and the final value is written via image stores.

Alpha value isn't the only thing we store in our R32_UINT texture. We also store an `object id` so that we can avoid objects self intersections (think of polylines crossing over themselves) with 24 bits for ID and 8 bits for alpha.

In more details:
1. Every fragment being processed checks if it's object id is the same as the one in the Read/Write R32_UINT Texture.
    - if it's the same then it does a MAX operation from the current calculate alpha and the one existing in the Texture.
    - if it's not the same then it **resolves**:
        - It renders the pixel using the "Previous!" object's style that was in the texture

2. There is a last fullscreen pass that resolves anything unresolved