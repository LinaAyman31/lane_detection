@echo off

if "%1" == "" (
    echo "Error! Input video path missing."
)
if "%2" == "" (
    echo "Error! Output video path missing."
)
if "%4" == "" (
    echo "Error! Debugging value missing." 
)
if "%6" =="" (
    echo "Error! Cars value missing."
)

if "%3" == "--debug" (
    set flag = false
    if "%4" == "0" (
        set flag = true
    ) if "%4" == "1" (
        set flag = true
    ) if flag == true (
        set flag = false
        if "%5" == "--cars" (
            if "%6" == "0" (
                set flag = true
            )  if "%6" == "1" (
                set flag = true
            ) if flag == true (
                conda activate base
                python main.py %1 %2 %4 %6
            ) else (
                echo "Error! Cars value should be 0 or 1 only."
            )
        ) else (
            echo "Error! --cars keyword missing."
        )
    ) else (
        echo "Error! Debugging value should be 0 or 1 only."
    ) 
) else (
    echo "Error! --debug keyword missing."
)
pause