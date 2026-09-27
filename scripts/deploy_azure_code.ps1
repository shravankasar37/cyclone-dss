# Deploys the full backend (all features) to Azure App Service as Python code (no Docker needed).
# Azure installs the Python packages on its own network ("Oryx" build), so only ~230 MB is uploaded.
#
# Usage (from the project root, after `az login`):
#   powershell -ExecutionPolicy Bypass -File .\scripts\deploy_azure_code.ps1 -Name cyclonesk26 -GeminiKey "<key>"
# Test packaging only (no Azure calls):  ... -Name cyclonesk26 -PackageOnly
param(
    [Parameter(Mandatory = $true)][ValidatePattern('^[a-z0-9]{5,20}$')][string]$Name,
    [string]$Location = "centralindia",
    [string]$Sku = "B1",                       # 1 vCPU, 1.75 GB RAM (backend peaks at ~1.0 GB)
    [string]$GeminiKey = "",
    [string]$CorsOrigins = "http://localhost:3000",
    [switch]$PackageOnly
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$rg = "rg-$Name"; $plan = "$Name-plan"; $app = "$Name-api"

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
# locate the Azure CLI even in a window opened before it was installed
$azExe = (Get-Command az.cmd -ErrorAction SilentlyContinue).Source
if (-not $azExe) { $azExe = "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd" }
# read-only lookups: value, or nothing if it does not exist (Windows PowerShell 5.1 treats redirected stderr as errors)
function Get-AzValue { $ErrorActionPreference = "Continue"; $out = & $azExe @args 2>$null; if ($LASTEXITCODE -eq 0) { $out } }
function Invoke-AzCli { & $azExe @args; if ($LASTEXITCODE -ne 0) { throw "az $($args -join ' ') failed (exit $LASTEXITCODE)" } }

# ---------------------------------------------------------------- package
Step "Packaging backend (code + models + data)"
$stage = Join-Path $env:TEMP "cyclone_app_stage"
$zip = Join-Path $env:TEMP "cyclone_app.zip"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
if (Test-Path $zip) { Remove-Item -Force $zip }
New-Item -ItemType Directory $stage | Out-Null

function Copy-Tree($src, [string[]]$exclude = @()) {
    $to = Join-Path $stage $src
    New-Item -ItemType Directory -Force $to | Out-Null
    robocopy (Join-Path $root $src) $to /E /NFL /NDL /NJH /NJS /NP /XD __pycache__ /XF $exclude | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "copy failed: $src" }
}
Copy-Tree "ml" @("*.pyc")
Copy-Tree "backend" @("*.pyc", "requirements.txt")
Copy-Tree "models\weights" @("*_last.pt", "MC_*_best.pt")
Copy-Tree "data\processed"
Copy-Tree "data\raw\era5"
# requirements at the root tells Azure this is a Python app; it pulls CPU-only PyTorch from the PyTorch index
Copy-Item (Join-Path $root "deploy_space\requirements.txt") (Join-Path $stage "requirements.txt")

# tar.exe (built into Windows 10/11) writes zip paths with forward slashes, which Linux needs
Push-Location $stage
try { $items = @(Get-ChildItem -Name); & (Join-Path $env:SystemRoot "System32\tar.exe") -a -c -f $zip @items; if ($LASTEXITCODE -ne 0) { throw "zip failed" } }
finally { Pop-Location }
Write-Host ("Package: {0} ({1:N0} MB)" -f $zip, ((Get-Item $zip).Length / 1MB))
$global:LASTEXITCODE = 0
if ($PackageOnly) { Write-Host "PackageOnly: stopping before any Azure calls."; exit 0 }

# ---------------------------------------------------------------- azure
Step "Checking login"
$sub = Get-AzValue account show --query name -o tsv
if (-not $sub) { throw "Not logged in. Run: az login" }
Write-Host "Subscription: $sub"

Step "Registering the Web provider (first time only)"
Invoke-AzCli provider register --namespace Microsoft.Web --wait | Out-Null

Step "Resource group $rg in $Location"
Invoke-AzCli group create --name $rg --location $Location -o none

Step "Removing the unused container registry from the Docker attempt (saves credit)"
$acr = Get-AzValue acr show --name "${Name}acr" --resource-group $rg --query name -o tsv
if ($acr) { Invoke-AzCli acr delete --name "${Name}acr" --resource-group $rg --yes -o none; Write-Host "deleted ${Name}acr" } else { Write-Host "none" }

Step "App Service plan $plan ($Sku, Linux)"
if (-not (Get-AzValue appservice plan show --name $plan --resource-group $rg --query name -o tsv)) {
    Invoke-AzCli appservice plan create --name $plan --resource-group $rg --is-linux --sku $Sku --location $Location -o none
}

Step "Web app $app (Python 3.13)"
if (-not (Get-AzValue webapp show --name $app --resource-group $rg --query name -o tsv)) {
    Invoke-AzCli webapp create --name $app --resource-group $rg --plan $plan --runtime "PYTHON:3.13" -o none
}

Step "Settings (build on deploy, CORS, Gemini key, start command, always on)"
$settings = @("SCM_DO_BUILD_DURING_DEPLOYMENT=true", "SCM_COMMAND_IDLE_TIMEOUT=1800", "WEBSITES_CONTAINER_START_TIME_LIMIT=1800",
              "DB_PATH=/tmp/app.db", "MPLCONFIGDIR=/tmp/matplotlib", "CORS_ORIGINS=$CorsOrigins")
if ($GeminiKey) { $settings += "GEMINI_API_KEY=$GeminiKey" }
Invoke-AzCli webapp config appsettings set --name $app --resource-group $rg --settings @settings -o none
Invoke-AzCli webapp config set --name $app --resource-group $rg --always-on true `
    --startup-file "cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000" -o none

Step "Uploading the package (~230 MB) - Azure then installs packages (~5-10 min)"
Invoke-AzCli webapp deploy --name $app --resource-group $rg --src-path $zip --type zip --async true -o none

$url = "https://$app.azurewebsites.net"
Write-Host "`nUploaded. Azure is now installing packages and starting the app (10-15 min)." -ForegroundColor Green
Write-Host "Backend URL : $url"
Write-Host "Health check: $url/api/health   (should list 7 models)"
Write-Host "Live log    : az webapp log tail --name $app --resource-group $rg"
