@REM @echo off

@REM if "%1" == "" (
@REM     echo "Error! Input video path missing."
@REM )
@REM if "%2" == "" (
@REM     echo "Error! Output video path missing."
@REM )
@REM if "%4" == "" (
@REM     echo "Error! Debugging value missing." 
@REM )
@REM if "%6" =="" (
@REM     echo "Error! Cars value missing."
@REM )

@REM if "%3" == "--debug" (
@REM     set flag=0 
@REM     if "%4" == "0" (
@REM         echo "ana hena!"
@REM         set flag=1
@REM         echo %flag%
@REM     ) 
@REM     if "%4" == "1" (
@REM         set flag=1
@REM     ) 
@REM     if "%flag%" == "1" (
@REM         set flag=0
@REM         if "%5" == "--cars" (
@REM             if "%6" == "0" (
@REM                 set flag=1
@REM             )  
@REM             if "%6" == "1" (
@REM                 set flag=1
@REM             ) 
@REM             if "%flag%" == "1" (
@REM                 conda activate base
@REM                 python main.py %1 %2 %4 %6
@REM             ) else (
@REM                 echo "Error! Cars value should be 0 or 1 only."
@REM             )
@REM         ) else (
@REM             echo "Error! --cars keyword missing."
@REM         )
@REM     ) else (
@REM         echo "Error! Debugging value should be 0 or 1 only."
@REM     ) 
@REM ) else (
@REM     echo "Error! --debug keyword missing."
@REM )
@REM pause


@echo off

if "%1" == "" (
    echo "Error! Input video path missing."
)
if "%2" == "" (
    echo "Error! Output video path missing."
)
if "%3" == "" (
    echo "Error! --mode keyword missing."
)
if "%4" == "" (
    echo "Error! mode value missing." 
)

if "%3" == "--mode" (
    if "%4" == "0" (
        python main.py %1 %2 %4
        pause
    )
    if "%4" == "1" (
        python main.py %1 %2 %4
        pause
    )
    if "%4" == "2" (
        python main.py %1 %2 %4
        pause
    )
    if "%4" == "3" (
        python main.py %1 %2 %4
        pause
    ) else (
        echo "Error! mode value must be 0, 1, 2, or 3."
    )
    
) else (
    echo "Error! --mode keyword missing."
)
pause