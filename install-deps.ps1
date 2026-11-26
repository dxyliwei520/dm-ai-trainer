param([Parameter(Mandatory=$true)][ValidateSet('musubi','aitk')][string]$Target)
# Install python deps for one tool. Torch is already installed by install-torch.ps1.
$ErrorActionPreference = 'Continue'
$root = 'E:\ai-train'
$idx = 'https://mirrors.cloud.tencent.com/pypi/simple'
function Say($m) { Write-Output ("[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $m) }

if ($Target -eq 'musubi') {
    $py = "$root\env-musubi\Scripts\python.exe"
    $repo = "$root\src\musubi-tuner"
    Say '=== musubi deps start ==='
    Push-Location $repo
    & $py -m pip install . --index-url $idx --timeout 180 2>&1 |
        ForEach-Object { if ($_ -match 'Collecting|Installing collected|Successfully installed|ERROR|error:|no matching') { Say $_ } }
    Pop-Location
    Say '--- triton-windows (optional, for ConvRot INT8 fused kernels) ---'
    & $py -m pip install triton-windows --index-url $idx --timeout 180 2>&1 |
        ForEach-Object { if ($_ -match 'Successfully|ERROR|no matching') { Say $_ } }
    Say '=== musubi deps done ==='
} else {
    $py = "$root\env-aitk\Scripts\python.exe"
    $repo = "$root\src\ai-toolkit"
    Say '=== ai-toolkit deps start ==='
    Push-Location $repo
    & $py -m pip install --no-deps -r requirements.txt --index-url $idx --timeout 180 2>&1 |
        ForEach-Object { if ($_ -match 'Collecting|Successfully installed|ERROR|error:|no matching') { Say $_ } }
    Say '--- second pass: resolve deps of the packages above (torch excluded) ---'
    & $py -m pip install -r requirements.txt --index-url $idx --timeout 180 2>&1 |
        ForEach-Object { if ($_ -match 'Collecting|Successfully installed|ERROR|error:|no matching') { Say $_ } }
    Pop-Location
    Say '=== ai-toolkit deps done ==='
}
Say 'pip check:'
& $py -m pip check 2>&1 | Select-Object -First 20
