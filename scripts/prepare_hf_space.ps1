# Copies exactly what the backend Docker image needs into a cloned Hugging Face Space folder.
# Usage (from the project root):   .\scripts\prepare_hf_space.ps1 -SpaceDir deploy\cyclone-api
# Targets a free Gradio-SDK Space (Docker Spaces are paid). The Dockerfile in the project root is for other hosts.
param([Parameter(Mandatory = $true)][string]$SpaceDir)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $SpaceDir ".git"))) { throw "$SpaceDir is not a git clone of your Space. Clone it first (step B4)." }

function Copy-Tree($src, $dst, [string[]]$exclude = @()) {
    $from = Join-Path $root $src
    $to = Join-Path $SpaceDir $dst
    New-Item -ItemType Directory -Force $to | Out-Null
    robocopy $from $to /E /NFL /NDL /NJH /NJS /NP /XD __pycache__ /XF $exclude | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "copy failed: $src" }
}

# Gradio-SDK Space (free tier): entry point + Space-specific requirements
Copy-Item (Join-Path $root "deploy_space\app.py") $SpaceDir -Force
Copy-Item (Join-Path $root "deploy_space\requirements.txt") $SpaceDir -Force
Copy-Tree "ml" "ml"
Copy-Tree "backend" "backend" @("*.pyc")
Copy-Tree "models\weights" "models\weights" @("*_last.pt", "MC_*_best.pt")
Copy-Tree "data\processed" "data\processed"
Copy-Tree "data\raw\era5" "data\raw\era5"

# Space card: Gradio SDK, Python 3.12, entry file app.py
$readme = @"
---
title: Cyclone Decision Support API
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
pinned: false
---

Backend API for the SIH 2026 PS 26070 prototype (research prototype, not an official IMD/NDMA product).
Health: /api/health - Docs: /docs
"@
[IO.File]::WriteAllText((Join-Path (Resolve-Path $SpaceDir) "README.md"), $readme, (New-Object Text.UTF8Encoding $false))

# Large binary files go through Git LFS
$attrs = @"
*.pt filter=lfs diff=lfs merge=lfs -text
*.npz filter=lfs diff=lfs merge=lfs -text
*.npy filter=lfs diff=lfs merge=lfs -text
*.parquet filter=lfs diff=lfs merge=lfs -text
*.nc filter=lfs diff=lfs merge=lfs -text
*.png filter=lfs diff=lfs merge=lfs -text
"@
[IO.File]::WriteAllText((Join-Path (Resolve-Path $SpaceDir) ".gitattributes"), $attrs, (New-Object Text.UTF8Encoding $false))

$size = (Get-ChildItem $SpaceDir -Recurse -File | Where-Object { $_.FullName -notmatch "\\.git\\" } | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("Prepared {0} ({1:N0} MB). Next: git add . ; git commit ; git push  (step B6)" -f $SpaceDir, $size)
exit 0  # robocopy leaves exit code 1 for "files copied", which is success
