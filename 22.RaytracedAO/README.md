# Using the Renderer

## How the Renderer works

* You have control over how the renderer behaves via the following options passed to the `-PROCESS_SENSORS` command line option:


| Option							| Behaviour																																																																		       |
|-----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `RenderAllThenInteractive ID`		| It starts by rendering the scene with each sensor starting at the given ID, and will stop when enough samples are taken. To skip this part, press the `END` Key (more detail below). Then you'll have full control of the camera, you can take snapshots, move around and have fun :)|
| `RenderAllThenTerminate ID`		| Same as above but it will exit after rendering with each sensor.																																																					   |
| `RenderSensorThenInteractive ID`	| This will render only with the passed sensor ID. If the sensor ID is not valid, then it defaults to the first one in Mitsuba metadata. After the rendering you'll be in interactive mode.																							   |
| `InteractiveAtSensor ID`			| This will skip all rendering and you'll go straight to the interactive mode with the given sensor ID.																																												   |


* For all the above options, if the ID is not passed, then it defaults to 0.
* Before Exiting from the Renderer, the very last view will be rendered and denoised to files named like `LastView_spaceship_Sensor_0`

## CommandLine Help
```
Parameters:
-SCENE=sceneMitsubaXMLPathOrZipAndXML
-PROCESS_SENSORS ID

Description and usage: 

-SCENE:
	some/path extra/path which will make it skip the file choose dialog

	NOTE: If the scene path contains space, put it between quotation marks

-PROCESS_SENSORS ID:
	It will control the behaviour of sensors in the app as detailed above.
	If the option is not passed, then it defaults to RenderAllThenInteractive.
	If the ID is not passed, then it defaults to 0.
	
Example Usages :
	raytracedao.exe -SCENE=../../media/kitchen.zip scene.xml
	raytracedao.exe -SCENE=../../media/kitchen.zip scene.xml -PROCESS_SENSORS RenderAllThenInteractive
	raytracedao.exe -SCENE="../../media/my good kitchen.zip" scene.xml -PROCESS_SENSORS RenderAllThenTerminate 0
	raytracedao.exe -SCENE="../../media/my good kitchen.zip scene.xml" -PROCESS_SENSORS RenderSensorThenInteractive 1
	raytracedao.exe -SCENE="../../media/extraced folder/scene.xml" -PROCESS_SENSORS InteractiveAtSensor 2
```


## New mitsuba properties and tags 
Multiple Sensor tags in mitsuba XML's is now supported. This feature helps you have multiple views with different camera and film parameters without needing to execute the renderer and load again.

You can switch between those sensors using `PAGE UP/DOWN` Keys defined in more detail below.

### Properties added to \<sensor\>:

| Property Name | Description           | Type  | Default Value                            |
|---------------|-----------------------|-------|------------------------------------------|
|   moveSpeed   | Camera Movement Speed | float | NaN -> Will be deduced from scene bounds |
|   zoomSpeed   | Camera Zoom Speed     | float | NaN -> Will be deduced from scene bounds |
|  rotateSpeed  | Camera Rotation Speed | float | 300.0                                    |

#### Properties added to \<sensor type="perspective"\>:

| Property Name | Description                                                              | Type  | Default Value |
|---------------|--------------------------------------------------------------------------|-------|---------------|
|    shiftX     | Right/Left Lens-Shift in NDC units, 1.0 moves screen center to the edge. | float |      0.0      |
|    shiftY     | Up/Down Lens-Shift in NDC units, 1.0 moves screen center to the edge.    | float |      0.0      |

### Properties added to \<film\>
| Property Name  | Description                                                                                                                             | Type    | Default Value                                                                                                                                                            |
|----------------|-----------------------------------------------------------------------------------------------------------------------------------------|---------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|      outputFilePath      | Final Render Output Path;<br>Denoised Render will have "_denoised" suffix added to it.                                        | string  | Render_{SceneName}_Sensor_{SensorIdx}.exr<br>{SceneName} is the filename of the xml or zip loaded.<br>{SensorIdx} is the index of the Sensor in xml used for the render. |
|       cascadeCount       | The number of Luminance Cascades to use for the Re-Weighting Monte Carlo 2018 paper, 6 was the paper's default.               | integer | 1 (disable RWMC)                                                                                                                                                         |
|  cascadeLuminanceStart   | The Luminance value of the first Cascade, 1.0 was the paper's default.                                                        | float   | NaN (gets replaced by rfilter's Emin)                                                                                                                                    |
|   cascadeLuminanceBase   | The magnitude factor between subsequent Luminance Cascades, 8.0 was the paper's default.                                      | float   | NaN (gets replaced by the N-th root of the ratio of Maximum Emitter Radiance's luminance over cascadeLuminanceStart, where N is `cascadeCount-1`)                        |
|        bloomScale        | Denoiser Bloom Scale                                                                                                          | float   | 0.1                                                                                                                                                                      |
|      bloomIntensity      | Denoiser Bloom Intensity                                                                                                      | float   | 0.1                                                                                                                                                                      |
|       bloomFilePath      | Lens Flare File Path                                                                                                          | string  | "../../media/kernels/physical_flare_512.exr"                                                                                                                             |
|        tonemapper        | Tonemapper Settings for Denoiser                                                                                              | string  | "ACES=0.4,0.8"                                                                                                                                                           |
| cropOffsetX, cropOffsetY | Used to control the offset for cropping cubemap renders (instead of highQualityEdges)                                         | int     | 0                                                                                                                                                                        |
|  cropWidth, cropHeight   | Used to control the size for cropping cubemap renders (instead of highQualityEdges)                                           | int     | width-cropOffsetX, height-cropOffsetY                                                                                                                                    |
|envmapRegularizationFactor| Fractional blend between guiding paths based on just the BxDF (0.0) or the product of the BxDF and the Environment Map (1.0)<br>Valid parameter ranges are between 0.0 and 0.8 as guiding fully by the product produces extreme fireflies from indirect light or local lights. | float  | 0.5                      |      


### Properties added to \<film\>
| Property Name | Description                                                                                                      | Type  | Default Value |
|---------------|------------------------------------------------------------------------------------------------------------------|-------|---------------|
|     kappa     | Parameter from Re-weighting Monte Carlo 2018 paper where its an integer, high values reject more aggressively    | float | 0.0 (disable) |
|     Emin      | Threshold of absolute luminance below which a sample is always considered reliable. Default taken from the paper.| float | 0.05          |

### Example of a sensor using all new properties described above.
```xml
<sensor type="perspective" >
	<float name="fov" value="60" />
	<float name="moveSpeed" value="100.0" />
	<float name="zoomSpeed" value="1.0" />
	<float name="rotateSpeed" value="300.0" />
	<transform name="toWorld" >
		<matrix value="-0.89874 -0.0182716 -0.4381 1.211 0 0.999131 -0.0416703 1.80475 0.438481 -0.0374507 -0.89796 3.85239 0 0 0 1"/>
	</transform>
	<sampler type="sobol" >
		<integer name="sampleCount" value="1024" />
	</sampler>
	<film type="ldrfilm" >
		<string name="outputFilePath" value="C:\Users\MyUser\Desktop\MyRender.exr" />
		<integer name="width" value="1920" />
		<integer name="height" value="1080" />
		<string name="fileFormat" value="png" />
		<string name="pixelFormat" value="rgb" />
		<float name="gamma" value="2.2" />
		<boolean name="banner" value="false" />
		<integer name="cascadeCount" value="6" />
		<float name="cascadeLuminanceBase" value="2.0" />
		<float name="cascadeLuminanceStart" value="0.5" />
		<float name="bloomScale" value="0.1" />
		<float name="bloomIntensity" value="0.1" />
		<string name="bloomFilePath" value="../../media/kernels/physical_flare_512.exr" />
		<string name="tonemapper" value="ACES=0.4,0.8" />
		<rfilter type="tent" >
			<float name="kappa" value="1.0" />
			<float name="Emin" value="0.025" />
		</rfilter>
	</film>
</sensor>
```
### Example of Cubemap Render
```xml
<sensor type="spherical" >
		<transform name="toWorld" >
			<matrix value="-0.89874 -0.0182716 -0.4381 1.211 0 0.999131 -0.0416703 1.80475 0.438481 -0.0374507 -0.89796 3.85239 0 0 0 1"/>
		</transform>
		<sampler type="sobol" >
			<integer name="sampleCount" value="128" />
		</sampler>
		<film type="ldrfilm" >
			<string name="outputFilePath" value="C:\Users\Erfan\Desktop\Renders\MyCubeMapRender.exr" />
			<integer name="width" value="1152" />
			<integer name="height" value="1152" />
			<integer name="cropWidth" value="1024" />
			<integer name="cropHeight" value="1024" />
			<integer name="cropOffsetX" value="64" />
			<integer name="cropOffsetY" value="64" />
			<string name="fileFormat" value="png" />
			<string name="pixelFormat" value="rgb" />
			<float name="gamma" value="2.2" />
			<boolean name="banner" value="false" />
			<float name="bloomScale" value="0.1" />
			<float name="bloomIntensity" value="0.1" />
			<string name="bloomFilePath" value="../../media/kernels/physical_flare_512.exr" />
			<string name="tonemapper" value="ACES=0.4,0.8" />
			<rfilter type="tent" />
		</film>
</sensor>
```

Example above renders to 1024x1024 cubemap sides and uses 64 offset pixels for higher quality. 
So the full width, height are 1152x1152 (64+1024+64=1152)

## Mouse

| Button              | Description                             |
|---------------------|-------------------------------------------------|
| Left Mouse Button   | Drag to Look around                             |
| Mouse Wheel Scroll  | Zoom In/Out (you can set the speed via mitsuba) |
| Right Mouse Button  | Drag to Move around                             |
| Middle Mouse Button | Drag to Move around                             |

## Keyboard
| Key       | Description                                                                                                            |
|-----------|------------------------------------------------------------------------------------------------------------------------|
| Q         | Press to Quit the Renderer                                                                                             |
| END       | Press to Skip Current Render and "free" the camera                                                                     |
| PAGE_UP   | Press to switch view to the 'next' sensor defined in mitsuba.                                                          |
| PAGE_DOWN | Press to switch view to the 'previous' sensor defined in mitsuba.                                                      |
| HOME      | Press to reset the camera to the initial view. (Usefull when you're lost and you want to go back to where you started) |
| P         | Press to take a snapshot when moving around (will be denoised)                                                         |
| L         | Press to log the current progress percentage and samples rendered.                                                     |
| B         | Toggle between Path Tracing and Albedo preview, allows you to position the camera more responsively in complex scenes. |

## Denoiser Hook
`denoiser_hook.bat` is a script that you can call to denoise your rendered images.

Example:
```
denoiser_hook.bat "Render_scene_Sensor_0.exr" "Render_scene_Sensor_0_albedo.exr" "Render_scene_Sensor_0_normal.exr" "../../media/kernels/physical_flare_512.exr" 0.1 0.3 "ACES=0.4,0.8"
```

Parameters:
1. ColorFile
2. AlbedoFile
3. NormalFile
4. BloomPsfFilePath
5. BloomScale
6. BloomIntensity
7. TonemapperArgs(string)


## Testing in batch

Run `test.bat` to batch render all of the files referenced in `test_scenes.txt`

Here is an example of  `test_scenes.txt`:
```
"../../media/mitsuba/staircase2.zip scene.xml"
"C:\Mitsuba\CoolLivingRoom\scene.xml"
"C:\Mitsuba\CoolKitchen.zip scene.xml"
"../../MitsubaFiles/spaceship.zip scene.xml"
; Here is my Commented line that batch file will skip (started with semicolons)
; "relative/dir/from/bin/folder/to/scene.zip something.xml
```
lines with semicolons will be skipped.
