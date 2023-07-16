import org.DevshGraphicsProgramming.Agent
import org.DevshGraphicsProgramming.BuilderInfo
import org.DevshGraphicsProgramming.IBuilder

class CGeometryCreatorBuilder extends IBuilder
{
	public CGeometryCreatorBuilder(Agent _agent, _info)
	{
		super(_agent, _info)
	}
	
	@Override
	public boolean prepare(Map axisMapping)
	{
		return true
	}
	
	@Override
  	public boolean build(Map axisMapping)
	{
		IBuilder.CONFIGURATION config = axisMapping.get("CONFIGURATION")
		IBuilder.BUILD_TYPE buildType = axisMapping.get("BUILD_TYPE")
		
		def nameOfBuildDirectory = getNameOfBuildDirectory(buildType)
		def nameOfConfig = getNameOfConfig(config)
		
		agent.execute("cmake --build ${info.rootProjectPath}/${nameOfBuildDirectory}/${info.targetProjectPathRelativeToRoot} --target ${info.targetBaseName} --config ${nameOfConfig} -j12 -v")
		
		return true
	}
	
	@Override
  	public boolean test(Map axisMapping)
	{
		return true
	}
	
	@Override
	public boolean install(Map axisMapping)
	{
		return true
	}
}

def create(Agent _agent, _info)
{
	return new CGeometryCreatorBuilder(_agent, _info)
}

return this