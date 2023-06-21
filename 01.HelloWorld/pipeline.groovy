import org.DevshGraphicsProgramming.Agent
import org.DevshGraphicsProgramming.IBuilder

class CExampleBuilder extends IBuilder
{
	public CExampleBuilder(Agent _agent, _targetBaseName, _projectPathRelativeToNabla)
	{
		super(_agent)
		
		targetBaseName = _targetBaseName
		projectPathRelativeToNabla = _projectPathRelativeToNabla
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
		
		agent.execute("cmake --build ../../${nameOfBuildDirectory}/${projectPathRelativeToNabla} --target ${targetBaseName} --config ${nameOfConfig} -j12 -v")
		
		return true
	}
	
	@Override
  	public boolean test(Map axisMapping)
	{
		return true
	}
	
	@Override
	public boolean deploy(Map axisMapping)
	{
		return true
	}
	
	public final def targetBaseName //! base name of CMake target
	public final def projectPathRelativeToNabla //! relative path of the project to Nabla root
}

def create(Agent _agent, _targetBaseName, _projectPathRelativeToNabla)
{
	return new CExampleBuilder(_agent, _targetBaseName, _projectPathRelativeToNabla)
}

return this
