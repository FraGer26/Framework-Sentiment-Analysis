@echo off
REM Script di avvio per Spark con Java 11
REM Imposta JAVA_HOME per usare Java 11 compatibile con Spark

set JAVA_HOME=C:\Java\jdk-11.0.24+8
set PATH=%JAVA_HOME%\bin;%PATH%

echo Usando Java: %JAVA_HOME%
java -version
echo.
echo Avvio Dashboard con Spark...
streamlit run app/app.py
