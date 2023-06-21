import org.DevshGraphicsProgramming.Agent
import org.DevshGraphicsProgramming.IBuilder

class CExampleBuilder extends IBuilder
{
	public CExampleBuilder(Agent _agent, _targetBaseName, _projectDirectoryName)
	{
		super(_agent)
		
		targetBaseName = _targetBaseName
		projectDirectoryName = _projectDirectoryName
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
		
		agent.execute("cmake --build ../../${nameOfBuildDirectory}/examples_tests/${projectDirectoryName} --target ${targetBaseName} --config ${nameOfConfig} -j12 -v")
		
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
	public final def projectDirectoryName //! parent directory name
}

def create(Agent _agent, _targetBaseName, _projectDirectoryName)
{
	return new CExampleBuilder(_agent, _targetBaseName, _projectDirectoryName)
}

return this
