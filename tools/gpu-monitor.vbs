' Launch the GPU monitor with no console flash.
' (wscript itself has no console window, and Run(..., 0, ...) hides the child.)
Dim sh, fso, dir
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
dir = fso.GetParentFolderName(WScript.ScriptFullName)
sh.Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & dir & "\gpu-monitor.ps1""", 0, False
