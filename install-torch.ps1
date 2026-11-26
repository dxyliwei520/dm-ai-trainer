param([Parameter(Mandatory=$true)][string]$EnvName)
# Install torch (cu130) + torchvision + torchaudio into E:\ai-train\<EnvName>
# Torch wheels from SJTU mirror (~5 MB/s), deps from Tencent PyPI mirror.
$ErrorActionPreference = 'Continue'
$py = "E:\ai-train\$EnvName\Scripts\python.exe"
$log = "E:\ai-train\install-$EnvName.log"
function Say($m) { $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $m; Write-Output $line }

Say "=== $EnvName : torch install start ==="
& $py -m pip install --upgrade pip --index-url https://mirrors.cloud.tencent.com/pypi/simple 2>&1 | Select-Object -Last 3
Say "pip: " + (& $py -m pip --version)

$torchArgs = @(
  'install',
  'torch==2.13.0', 'torchvision==0.28.0', 'torchaudio==2.11.0',
  '--index-url', 'https://mirror.sjtu.edu.cn/pytorch-wheels/cu130/',
  '--extra-index-url', 'https://mirrors.cloud.tencent.com/pypi/simple',
  '--timeout', '120'
)
Say "running: pip install torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 (cu130)"
$sw = [Diagnostics.Stopwatch]::StartNew()
& $py -m pip @torchArgs 2>&1 | ForEach-Object { if ($_ -match 'Downloading|Collecting torch|Successfully installed|ERROR|error:') { Say $_ } }
$sw.Stop()
Say ("torch stage finished in " + [math]::Round($sw.Elapsed.TotalMinutes,1) + " min")

Say "verify:"
& $py -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available())" 2>&1
Say "=== $EnvName : torch stage done ==="
