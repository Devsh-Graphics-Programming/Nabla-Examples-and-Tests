
//this is user defined data sent from the task shader to the mesh shader
//1 packet is sent, but it can use arrays so that each workgroup can receive customized data
struct TaskToMeshPayload {
    uint objectType[INSTANCE_COUNT];
};

//1 is cone, 2 is for fan, anything else for trangle list without the special normal calc.
//cone can be handled in the task shader or the mesh shader, I'm going to handle it in the task shader
//#define OTHER_OBJECTS 0
#define CONE_OBJECT_TYPE 1
#define T_FAN_OBJECT_TYPE 2
struct MeshData{
    uint vertCount;
    uint primCount;
    uint objType; 
	uint positionView;
	uint normalView;
};

[[vk::binding(1)]] cbuffer MeshDataBuffer {
    
    MeshData meshData[OBJECT_COUNT];
    float4x4 transform[INSTANCE_COUNT]; //this is goign to be based on device limits
};

#define PushDescCount (0x1<<16)-1
struct SPushConstants {
	float4x4 viewProj;
    uint objectCount[OBJECT_COUNT];
};

//im not keen on trying to figure out how the push constant abstraction worked before without documentation
[[vk::push_constant]] SPushConstants pc;