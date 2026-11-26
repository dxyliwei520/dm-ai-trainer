# Download ai-toolkit + musubi-tuner zips through a China-friendly mirror and extract.
# ASCII only. Safe to re-run: skips if the target folder already exists.
$ErrorActionPreference = 'Continue'
$root = 'E:\ai-train'
$src  = Join-Path $root 'src'
New-Item -ItemType Directory -Force -Path $src | Out-Null

$jobs = @(
    @{ name = 'musubi-tuner'; repo = 'kohya-ss/musubi-tuner'; zip = 'musubi-tuner-main.zip' },
    @{ name = 'ai-toolkit';   repo = 'ostris/ai-toolkit';   zip = 'ai-toolkit-main.zip' }
)

$mirrors = @('https://ghfast.top/', 'https://gh-proxy.com/', 'https://codeload.github.com/')

foreach ($j in $jobs) {
    $dest = Join-Path $src $j.name
    if (Test-Path (Join-Path $dest 'requirements.txt')) { Write-Output ("SKIP " + $j.name + " (already present)"); continue }
    $zipPath = Join-Path $env:TEMP $j.zip
    $ok = $false
    foreach ($m in $mirrors) {
        if ($m -like '*codeload*') {
            $url = 'https://codeload.github.com/' + $j.repo + '/zip/refs/heads/main'
        } else {
            $url = $m + 'https://github.com/' + $j.repo + '/archive/refs/heads/main.zip'
        }
        Write-Output ("GET " + $j.name + " via " + $m)
        & curl.exe -sL --max-time 900 -o $zipPath $url
        if ((Test-Path $zipPath) -and ((Get-Item $zipPath).Length -gt 1MB)) {
            try {
                $tmp = Join-Path $env:TEMP ('x_' + $j.name)
                if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
                Expand-Archive -Path $zipPath -DestinationPath $tmp -Force
                $inner = Get-ChildItem $tmp -Directory | Select-Object -First 1
                if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
                Move-Item $inner.FullName $dest
                Write-Output ("OK   " + $j.name + " -> " + $dest + "  (" + [math]::Round(((Get-ChildItem $dest -Recurse -File | Measure-Object Length -Sum).Sum / 1MB), 1) + " MB)")
                $ok = $true
                break
            } catch {
                Write-Output ("EXTRACT FAIL " + $j.name + " : " + $_.Exception.Message)
            }
        } else {
            Write-Output ("SLOW/FAIL " + $m + " (size=" + (Get-Item $zipPath -ErrorAction SilentlyContinue).Length + ")")
        }
    }
    if (-not $ok) { Write-Output ("FAILED " + $j.name) }
}
Write-Output '=== zip stage done ==='
