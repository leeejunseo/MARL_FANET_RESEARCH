@echo off
setlocal

REM Usage:
REM   run_ns3_bridge_pipeline.bat logs\ns3_events.csv

set EVENTS_ARG=
if not "%~1"=="" set EVENTS_ARG=--events "%~1"

python run_ns3_bridge_pipeline.py %EVENTS_ARG% --trace logs/ns3_link_trace.csv --num-drones 3 --step-seconds 1.0 --seeds 42,43,44,45,46 --policy trained --scenario Default --gif logs/ns3_bridge_view.gif
if errorlevel 1 (
    echo [오류] ns-3 브리지 파이프라인 실행 실패
    exit /b 1
)

echo [완료] ns-3 브리지 파이프라인 실행 성공
endlocal
