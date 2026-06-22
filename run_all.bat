@echo off
setlocal
python train.py
if errorlevel 1 (
    echo [오류] train.py가 실패했습니다.
    exit /b 1
)
python eval.py
if errorlevel 1 (
    echo [오류] eval.py가 실패했습니다.
    exit /b 1
)
python utils/generate_all_plots.py
if errorlevel 1 (
    echo [오류] generate_all_plots.py가 실패했습니다.
    exit /b 1
)
echo 모든 단계가 정상적으로 완료되었습니다.
endlocal
