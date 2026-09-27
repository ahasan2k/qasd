<#
.SYNOPSIS
  Installs Qasd Gateway on Windows and keeps it running in the background.

.DESCRIPTION
  - Installs Python 3.12 with winget if no Python 3.11+ is found
  - Downloads Qasd from GitHub into the install folder (or updates it)
  - Creates a virtual environment and installs Qasd into it
  - Writes a .env file with a new gateway key and admin key (kept on later runs)
  - Registers a scheduled task "Qasd Gateway" that starts Qasd at boot and restarts it if it stops
  - Starts it and checks /healthz

  Works in Windows PowerShell 5.1 and PowerShell 7. Run it again at any time to update.

.EXAMPLE
  # Run from an elevated PowerShell (needed once, for the scheduled task):
  powershell -ExecutionPolicy Bypass -File .\install-windows.ps1

.EXAMPLE
  # Update code only, keep keys and settings:
  powershell -ExecutionPolicy Bypass -File .\install-windows.ps1 -Update
#>
[CmdletBinding()]
param(
    [string]$InstallDir = "$env:ProgramData\Qasd",
    [int]$Port = 8787,
    # Listen on all interfaces instead of localhost only. Opens the Windows firewall for the port.
    [switch]$Listen,
    [switch]$Update,
    [string]$Repo = "ahasan2k/qasd",
    [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"
$TaskName = "Qasd Gateway"

function Say($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Warn($msg) { Write-Host "!!  $msg" -ForegroundColor Yellow }

function Test-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    return (New-Object Security.Principal.WindowsPrincipal($id)).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function New-Key([int]$bytes = 24) {
    $buf = New-Object byte[] $bytes
    [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($buf)
    return "qasd-" + (($buf | ForEach-Object { $_.ToString("x2") }) -join "")
}

function Read-Secret([string]$prompt) {
    $secure = Read-Host $prompt -AsSecureString
    $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try { return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr) }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr) }
}

function Find-Python {
    foreach ($cmd in @("py -3.12", "py -3.11", "python")) {
        try {
            $parts = $cmd.Split(" ")
            $exe = $parts[0]
            $args_ = @()
            if ($parts.Length -gt 1) { $args_ = $parts[1..($parts.Length - 1)] }
            $ver = & $exe @args_ -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($LASTEXITCODE -eq 0 -and $ver) {
                $v = [version]$ver.Trim()
                if ($v -ge [version]"3.11") {
                    $path = & $exe @args_ -c "import sys; print(sys.executable)"
                    return $path.Trim()
                }
            }
        } catch { }
    }
    return $null
}

if (-not (Test-Admin)) {
    throw "Please run this from an elevated PowerShell (Run as administrator). It is needed once to register the startup task."
}

# 1. Python ----------------------------------------------------------------
$python = Find-Python
if (-not $python) {
    Say "Python 3.11+ not found. Installing Python 3.12 with winget..."
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "winget is not available. Install Python 3.12 from https://www.python.org/downloads/ (tick 'Add to PATH'), then run this again."
    }
    winget install --id Python.Python.3.12 --scope machine --silent --accept-package-agreements --accept-source-agreements | Out-Null
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
    $python = Find-Python
    if (-not $python) { throw "Python installed but not found on PATH. Open a new PowerShell window and run this again." }
}
Say "Using Python at $python"

# 2. Code ------------------------------------------------------------------
$AppDir = Join-Path $InstallDir "app"
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

# Stop a running instance before replacing files
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

Say "Downloading $Repo ($Branch) from GitHub..."
$zip = Join-Path $env:TEMP "qasd-$Branch.zip"
$unz = Join-Path $env:TEMP "qasd-unzip"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
Invoke-WebRequest -UseBasicParsing -Uri "https://github.com/$Repo/archive/refs/heads/$Branch.zip" -OutFile $zip
if (Test-Path $unz) { Remove-Item -Recurse -Force $unz }
Expand-Archive -Path $zip -DestinationPath $unz -Force
$src = Get-ChildItem $unz | Select-Object -First 1

# Keep a customised routing.yaml across updates
$routing = Join-Path $AppDir "config\routing.yaml"
$routingBackup = $null
if (Test-Path $routing) {
    $routingBackup = Join-Path $InstallDir "routing.yaml.keep"
    Copy-Item $routing $routingBackup -Force
}
if (Test-Path $AppDir) { Remove-Item -Recurse -Force $AppDir }
Move-Item $src.FullName $AppDir
if ($routingBackup) { Copy-Item $routingBackup $routing -Force; Remove-Item $routingBackup }
Remove-Item -Recurse -Force $unz, $zip -ErrorAction SilentlyContinue

# 3. Virtual environment ----------------------------------------------------
$venv = Join-Path $InstallDir "venv"
$venvPy = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Say "Creating virtual environment..."
    & $python -m venv $venv
}
Say "Installing Qasd and its dependencies (first run takes a few minutes)..."
& $venvPy -m pip install --quiet --upgrade pip
& $venvPy -m pip install --quiet --upgrade $AppDir
if ($LASTEXITCODE -ne 0) { throw "pip install failed." }

# 4. Settings (.env) -------------------------------------------------------
$envFile = Join-Path $InstallDir ".env"
$bindHost = "127.0.0.1"
if ($Listen) { $bindHost = "0.0.0.0" }

if (-not (Test-Path $envFile)) {
    Say "Creating settings with new keys..."
    $appKey = New-Key
    $adminKey = New-Key
    Write-Host ""
    Write-Host "Provider keys are optional here; add or change them later in $envFile" -ForegroundColor Gray
    $anthropic = Read-Secret "ANTHROPIC_API_KEY (press Enter to skip)"
    $openai = Read-Secret "OPENAI_API_KEY (press Enter to skip)"
    $gemini = Read-Secret "GEMINI_API_KEY (press Enter to skip)"
    $dbPath = (Join-Path $InstallDir "qasd.db") -replace "\\", "/"
    $lines = @(
        "# Qasd settings. Restart the 'Qasd Gateway' scheduled task after editing.",
        "QASD_API_KEYS=$appKey",
        "QASD_ADMIN_KEY=$adminKey",
        "QASD_HOST=$bindHost",
        "QASD_PORT=$Port",
        "QASD_DATABASE_URL=sqlite+aiosqlite:///$dbPath",
        "QASD_ROUTING_FILE=config/routing.yaml",
        "",
        "ANTHROPIC_API_KEY=$anthropic",
        "OPENAI_API_KEY=$openai",
        "GEMINI_API_KEY=$gemini",
        "# OLLAMA_API_BASE=http://localhost:11434",
        "",
        "# QASD_SUMMARY_MODEL=claude-haiku-4-5",
        "# QASD_REDIS_URL=redis://localhost:6379/0"
    )
    Set-Content -Path $envFile -Value $lines -Encoding ASCII
    # Only Administrators and SYSTEM can read the keys
    icacls $envFile /inheritance:r /grant:r "Administrators:F" "SYSTEM:F" | Out-Null
} else {
    Say "Keeping existing settings in $envFile"
    if ($PSBoundParameters.ContainsKey("Listen") -or $PSBoundParameters.ContainsKey("Port")) {
        $content = Get-Content $envFile
        $content = $content -replace "^QASD_HOST=.*", "QASD_HOST=$bindHost" -replace "^QASD_PORT=.*", "QASD_PORT=$Port"
        Set-Content -Path $envFile -Value $content -Encoding ASCII
    }
}

# 5. Runner script ---------------------------------------------------------
$runner = Join-Path $InstallDir "run-qasd.ps1"
$runnerBody = @'
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Get-Content (Join-Path $root ".env") | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
        $i = $line.IndexOf("=")
        $name = $line.Substring(0, $i).Trim()
        $value = $line.Substring($i + 1).Trim()
        if ($value) { [Environment]::SetEnvironmentVariable($name, $value, "Process") }
    }
}
Set-Location (Join-Path $root "app")
$log = Join-Path $root "qasd.log"
if ((Test-Path $log) -and ((Get-Item $log).Length -gt 20MB)) { Move-Item $log "$log.old" -Force }
& (Join-Path $root "venv\Scripts\python.exe") -m qasd *>> $log
'@
Set-Content -Path $runner -Value $runnerBody -Encoding ASCII

# 6. Scheduled task: start at boot, restart on failure ----------------------
Say "Registering scheduled task '$TaskName'..."
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$runner`""
$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Principal $principal -Description "Qasd token-saving AI gateway" -Force | Out-Null

# 7. Firewall ----------------------------------------------------------------
$ruleName = "Qasd Gateway $Port"
if ($Listen) {
    if (-not (Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue)) {
        New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Protocol TCP -LocalPort $Port -Action Allow | Out-Null
    }
    Warn "Qasd now accepts connections from other machines on port $Port over plain HTTP."
    Warn "Only do this on a private network or VPN, or put it behind HTTPS (for example Caddy or Cloudflare Tunnel)."
} else {
    Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue | Remove-NetFirewallRule
}

# 8. Start and check ---------------------------------------------------------
Say "Starting Qasd..."
Start-ScheduledTask -TaskName $TaskName
$ok = $false
for ($i = 0; $i -lt 40; $i++) {
    Start-Sleep -Seconds 2
    try {
        $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 -Uri "http://127.0.0.1:$Port/healthz"
        if ($r.StatusCode -eq 200) { $ok = $true; break }
    } catch { }
}

$keys = @{}
Get-Content $envFile | ForEach-Object {
    if ($_ -match "^(QASD_API_KEYS|QASD_ADMIN_KEY)=(.*)$") { $keys[$Matches[1]] = $Matches[2] }
}

Write-Host ""
if ($ok) {
    Write-Host "Qasd is running." -ForegroundColor Green
} else {
    Warn "Qasd did not answer on port $Port yet. Check the log: $(Join-Path $InstallDir 'qasd.log')"
}
Write-Host ""
Write-Host "  OpenAI-compatible URL : http://localhost:$Port/v1"
Write-Host "  Anthropic URL         : http://localhost:$Port   (ANTHROPIC_BASE_URL)"
Write-Host "  Dashboard             : http://localhost:$Port/dashboard"
Write-Host "  App key               : $($keys['QASD_API_KEYS'])"
Write-Host "  Admin key (dashboard) : $($keys['QASD_ADMIN_KEY'])"
Write-Host "  Settings              : $envFile"
Write-Host "  Log                   : $(Join-Path $InstallDir 'qasd.log')"
Write-Host ""
Write-Host "After editing settings, restart with:  Stop-ScheduledTask '$TaskName'; Start-ScheduledTask '$TaskName'"