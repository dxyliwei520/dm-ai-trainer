# ai-toolkit deps: install pinned diffusers from the local source dir, then the rest.
$ErrorActionPreference = 'Continue'
$root = 'E:\ai-train'
$py = "$root\env-aitk\Scripts\python.exe"
$idx = 'https://mirrors.cloud.tencent.com/pypi/simple'
function Say($m) { Write-Output ("[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $m) }

Say '=== aitk deps (2nd attempt) ==='
Say 'step 1/3: pinned diffusers from local source'
& $py -m pip install --no-deps "$root\src\diffusers-pinned" --timeout 180 2>&1 |
    ForEach-Object { if ($_ -match 'Successfully|ERROR|error:|Building|Installing') { Say $_ } }

Say 'step 2/3: requirements (minus the git+diffusers line)'
& $py -m pip install -r "$root\req-aitk.txt" --index-url $idx --timeout 180 2>&1 |
    ForEach-Object { if ($_ -match 'Successfully installed|ERROR|error:|no matching|Conflict|requires ') { Say $_ } }

Say 'step 3/3: pip check'
& $py -m pip check 2>&1 | Select-Object -First 25
Say '=== aitk deps done ==='
