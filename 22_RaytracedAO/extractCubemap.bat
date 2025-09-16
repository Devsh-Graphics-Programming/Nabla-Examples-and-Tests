REM @echo off

REM examplary usage:
REM mergeCubemap.bat 64 64 mergedImage.png stripeFormat.png

set cropOffsetX0=%1
set cropOffsetY0=%2

set in=%3
set out=%4

REM set extracted image size
for /f "tokens=*" %%s in ('magick identify -format "%%w" %in%') do set sz=%%s
set /a paddedSize = sz/3

set /a realSize = paddedSize-2*cropOffsetX0

set /a cropOffsetX1 = cropOffsetX0+paddedSize
set /a cropOffsetX2 = cropOffsetX0+paddedSize*2
set /a cropOffsetX3 = cropOffsetX0+paddedSize*3
set /a cropOffsetX4 = cropOffsetX0+paddedSize*4
set /a cropOffsetX5 = cropOffsetX0+paddedSize*5
set /a cropOffsetY1 = paddedSize+64

set /a x0 = 0
set /a x1 = realSize
set /a x2 = 2*realSize
set /a x3 = 3*realSize
set /a x4 = 5*realSize
set /a x5 = 4*realSize

set /a stripWidth = realSize*6
magick convert -size %stripWidth%x%realSize% canvas:none ^
( %in% -crop %realSize%x%realSize%+%cropOffsetX0%+%cropOffsetY1% -matte -virtual-pixel transparent -geometry %realSize%x%realSize%+%x0%+0 ) -composite ^
( %in% -crop %realSize%x%realSize%+%cropOffsetX2%+%cropOffsetY1% -matte -virtual-pixel transparent -geometry %realSize%x%realSize%+%x1%+0 ) -composite ^
( %in% -crop %realSize%x%realSize%+%cropOffsetX1%+%cropOffsetY0% -matte -virtual-pixel transparent -geometry %realSize%x%realSize%+%x2%+0 ) -composite ^
( %in% -crop %realSize%x%realSize%+%cropOffsetX2%+%cropOffsetY0% -matte -virtual-pixel transparent -geometry %realSize%x%realSize%+%x3%+0 ) -composite ^
( %in% -crop %realSize%x%realSize%+%cropOffsetX0%+%cropOffsetY0% -matte -virtual-pixel transparent -geometry %realSize%x%realSize%+%x4%+0 ) -composite ^
( %in% -crop %realSize%x%realSize%+%cropOffsetX1%+%cropOffsetY1% -matte -virtual-pixel transparent -geometry %realSize%x%realSize%+%x5%+0 ) -composite ^
%out%