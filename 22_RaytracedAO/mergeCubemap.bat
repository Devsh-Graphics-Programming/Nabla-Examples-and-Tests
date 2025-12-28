@echo off

REM the ordering of the cubemap faces is irrelevant as long as extractCubemap knows what has been merged together here
set first=%1
set second=%2
set third=%3
set fourth=%4
set fifth=%5
set sixth=%6
set output=%~dpn7

REM examplary usage: 
REM mergeCubemap.bat first.png second.png third.png fourth.png fifth.png sixth.png outputImageName

REM set image size
for /f "tokens=*" %%s in ('magick identify -format "%%w" %first%') do set sz=%%s

REM set image fromat
for /f "tokens=*" %%s in ('magick identify -format "%%m" %first%') do set format=%%s

set /a szx2=2*sz
set /a outputWidth=3*sz
set /a outputHeight=2*sz

magick convert -size %outputwidth%x%outputHeight% canvas:none ^
-draw "image over  0,0 0,0 '%sixth%'" ^
-draw "image over  %sz%,0 0,0 '%fourth%'" ^
-draw "image over  %szx2%,0 0,0 '%third%'" ^
-draw "image over  0,%sz% 0,0 '%first%'" ^
-draw "image over  %sz%,%sz% 0,0 '%fifth%'" ^
-draw "image over  %szx2%,%sz% 0,0 '%second%'" ^
%output%.%format%