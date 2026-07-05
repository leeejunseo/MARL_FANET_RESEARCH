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
python visualize_attack.py --policy trained --demo-60 --demo-seconds 60 --fps 3 --output logs/demo_60s.gif --event-log logs/demo_60s_link_events.csv --no-show
if errorlevel 1 (
    echo [오류] visualize_attack.py 데모 생성이 실패했습니다.
    exit /b 1
)
echo 모든 단계가 정상적으로 완료되었습니다.
endlocal
