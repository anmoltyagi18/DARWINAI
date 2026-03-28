@echo off
echo ==============================================
echo AIGOFIN - Full System Installer ^& Launcher
echo ==============================================
echo.
echo Step 1: Installing Root Dependencies...
call npm install
echo.
echo Step 2: Installing Frontend Dependencies...
cd frontend
call npm install
cd ..
echo.
echo Step 3: Starting Backend and Frontend!
echo The backend URL will show in this console.
echo Next.js will typically run on http://localhost:3000
echo.
call npm run dev
pause
