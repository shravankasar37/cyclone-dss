# Deploys the full backend (all features) to Azure App Service using the project's Dockerfile.
# The image is built in Azure (az acr build), so only the source + models (~270 MB) are uploaded.
#
# Usage (from the project root, after `az login`):
#   powershell -ExecutionPolicy Bypass -File .\scripts\deploy_azure.ps1 -Name cyclone<yourinitials><2digits> `
#       -GeminiKey "<new gemini key>" -CorsOrigins "http://localhost:3000"
param(
    [Parameter(Mandatory = $true)][ValidatePattern('^[a-z0-9]{5,20}$')][string]$Name,   # lowercase letters/digits, globally unique
    [string]$Location = "centralindia",
    [string]$Sku = "B1",                       # B1 = 1 vCPU, 1.75 GB RAM (backend peaks at ~1.0 GB)
    [string]$GeminiKey = "",
    [string]$CorsOrigins = "http://localhost:3000",
    [string]$Tag = "v1"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$rg = "rg-$Name"; $acr = "${Name}acr"; $plan = "$Name-plan"; $app = "$Name-api"
$image = "$acr.azurecr.io/cyclone-api:$Tag"

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
# locate the Azure CLI even in a window opened before it was installed
$azExe = (Get-Command az.cmd -ErrorAction SilentlyContinue).Source
if (-not $azExe) { $azExe = "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd" }
if (-not (Test-Path $azExe)) { throw "Azure CLI not found. Install it: winget install -e --id Microsoft.AzureCLI" }
# named Invoke-AzCli: PowerShell names are case-insensitive, so a function called "Az" would call itself
# read-only lookups: returns the value, or nothing if the resource does not exist (never stops the script).
# Windows PowerShell 5.1 turns redirected stderr of native tools into errors, so relax that locally.
function Get-AzValue { $ErrorActionPreference = "Continue"; $out = & $azExe @args 2>$null; if ($LASTEXITCODE -eq 0) { $out } }
function Invoke-AzCli { & $azExe @args; if ($LASTEXITCODE -ne 0) { throw "az $($args -join ' ') failed (exit $LASTEXITCODE)" } }

Step "Checking login"
$sub = Get-AzValue account show --query name -o tsv
if (-not $sub) { throw "Not logged in. Run: az login" }
Write-Host "Subscription: $sub"

Step "Registering resource providers (first time only, can take a minute)"
foreach ($ns in "Microsoft.ContainerRegistry", "Microsoft.Web") { Invoke-AzCli provider register --namespace $ns --wait | Out-Null }

Step "Resource group $rg in $Location"
Invoke-AzCli group create --name $rg --location $Location -o none

Step "Container registry $acr"
$exists = Get-AzValue acr show --name $acr --resource-group $rg --query name -o tsv
if (-not $exists) { Invoke-AzCli acr create --name $acr --resource-group $rg --sku Basic --admin-enabled true --location $Location -o none }

Step "Building the image in Azure (uploads ~270 MB, then builds ~10-15 min)"
Push-Location $root
try { Invoke-AzCli acr build --registry $acr --image "cyclone-api:$Tag" --file Dockerfile . }
finally { Pop-Location }

Step "App Service plan $plan ($Sku, Linux)"
$exists = Get-AzValue appservice plan show --name $plan --resource-group $rg --query name -o tsv
if (-not $exists) { Invoke-AzCli appservice plan create --name $plan --resource-group $rg --is-linux --sku $Sku --location $Location -o none }

Step "Web app $app"
$user = Get-AzValue acr credential show --name $acr --query username -o tsv
$pass = Get-AzValue acr credential show --name $acr --query "passwords[0].value" -o tsv
if (-not $user -or -not $pass) { throw "Could not read the registry credentials for $acr" }
$exists = Get-AzValue webapp show --name $app --resource-group $rg --query name -o tsv
if (-not $exists) { Invoke-AzCli webapp create --name $app --resource-group $rg --plan $plan --container-image-name $image -o none }
Invoke-AzCli webapp config container set --name $app --resource-group $rg --container-image-name $image `
    --container-registry-url "https://$acr.azurecr.io" --container-registry-user $user --container-registry-password $pass -o none

Step "App settings (port, start-up time, CORS, Gemini key)"
$settings = @("WEBSITES_PORT=7860", "WEBSITES_CONTAINER_START_TIME_LIMIT=1800", "CORS_ORIGINS=$CorsOrigins")
if ($GeminiKey) { $settings += "GEMINI_API_KEY=$GeminiKey" }
Invoke-AzCli webapp config appsettings set --name $app --resource-group $rg --settings @settings -o none
Invoke-AzCli webapp config set --name $app --resource-group $rg --always-on true --http20-enabled true -o none
Invoke-AzCli webapp restart --name $app --resource-group $rg -o none

$url = "https://$app.azurewebsites.net"
Write-Host "`nDone. The first start pulls a 3.35 GB image and can take 5-10 minutes." -ForegroundColor Green
Write-Host "Backend URL : $url"
Write-Host "Health check: $url/api/health   (should list 7 models)"
Write-Host "Use this URL as NEXT_PUBLIC_API_URL on Vercel."
