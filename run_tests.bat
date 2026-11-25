@echo off
echo =======================================================================
echo     Running End-to-End Test Suite for Road Accident Severity Project
echo =======================================================================
"%~dp0venv\Scripts\python.exe" "%~dp0tests\test_pipeline.py"
pause

