@echo off
echo === Git Quick Push ===

git add .
git status

set /p msg="Commit message: "
if "%msg%"=="" set msg=update

git commit -m "%msg%"
git push

echo === Done ===
pause
