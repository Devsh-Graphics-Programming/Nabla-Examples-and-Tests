
#include "task_mesh_common.hlsl"

groupshared TaskToMeshPayload taskToMeshPayload;

[numthreads(1,1,1)]
void main(
	in uint3 id : SV_DispatchThreadID,
	in uint3 groupThreadId : SV_GroupThreadID
	//out payload TaskToMeshPayload taskToMeshPayload, interestingly, thats not how it's done here
){
	uint objectCount = 0;
	for(uint i = 0; i < OBJECT_COUNT; i++){
		for(uint j = 0; j < pc.objectCount[i]; j++){
			taskToMeshPayload.objectType[objectCount] = i;
			objectCount++;
		}
	}

    printf("dispatching meshes - %u", objectCount);
	DispatchMesh(objectCount, 1, 1, taskToMeshPayload);
}