# Verify both training environments without loading any big model.
# Checks: python / torch / CUDA / key imports / CLI --help of the two trainers.
$ErrorActionPreference = 'Continue'
$root = 'E:\ai-train'
function Line($t) { Write-Output ('') ; Write-Output ('==== ' + $t + ' ====') }

foreach ($env in @('env-musubi', 'env-aitk')) {
    $py = "$root\$env\Scripts\python.exe"
    Line "$env"
    if (-not (Test-Path $py)) { Write-Output 'MISSING venv'; continue }
    & $py -c "import sys;print('python',sys.version.split()[0],sys.executable)" 2>&1
    & $py -c "import torch;print('torch',torch.__version__,'| cuda',torch.version.cuda,'| available',torch.cuda.is_available())" 2>&1
    & $py -c "import torch;print('device',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a');print('capability',torch.cuda.get_device_capability(0) if torch.cuda.is_available() else '')" 2>&1
}

Line 'musubi-tuner: imports'
$m = "$root\env-musubi\Scripts\python.exe"
& $m -c "import diffusers,transformers,accelerate,safetensors,av,cv2,numpy;print('diffusers',diffusers.__version__,'| transformers',transformers.__version__,'| accelerate',accelerate.__version__)" 2>&1
& $m -c "import bitsandbytes;print('bitsandbytes ok')" 2>&1
& $m -c "import triton;print('triton',triton.__version__)" 2>&1

Line 'musubi-tuner: CLI (minimax_h3_train_network.py --help)'
Push-Location "$root\src\musubi-tuner"
& $m minimax_h3_train_network.py --help 2>&1 | Select-Object -First 12
Pop-Location

Line 'ai-toolkit: imports'
$a = "$root\env-aitk\Scripts\python.exe"
& $a -c "import diffusers,transformers,torchao,peft,accelerate;print('diffusers',diffusers.__version__,'| transformers',transformers.__version__,'| peft',peft.__version__)" 2>&1
& $a -c "import bitsandbytes;print('bitsandbytes ok')" 2>&1

Line 'ai-toolkit: run.py --help'
Push-Location "$root\src\ai-toolkit"
& $a run.py --help 2>&1 | Select-Object -First 15
Pop-Location

Line 'DONE'
