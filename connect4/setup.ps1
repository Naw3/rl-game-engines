# =============================================================================
# setup.ps1 - Robust setup for the Connect4 pipeline on Windows.
#
# Installs & Configures:
#   1. Python 3.11+ & packages (via requirements.txt)
#   2. Rust toolchain (via winget/rustup)
#   3. CUDA Toolkit 12.6 (via official installer)
#   4. cuDNN 9.x (via pip wheel)
#   5. Adds critical paths permanently to the User environment PATH.
#
# Run from the project root in a regular PowerShell:
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1
# =============================================================================

# --- Self-elevate to admin if not already ----------------------------------
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    $scriptPath = $MyInvocation.MyCommand.Path
    Write-Host '[setup] Not running as admin - re-launching elevated to allow system modifications.'
    # Correctly set location to PSScriptRoot when running elevated to prevent running from C:\Windows\System32
    $argList = "-NoProfile -ExecutionPolicy Bypass -Command `"Set-Location -Path '$PSScriptRoot'; & '$scriptPath' -Elevated`""
    try {
        $proc = Start-Process -FilePath powershell -ArgumentList $argList -Verb RunAs -WindowStyle Normal -PassThru
        $proc.WaitForExit()
        exit $proc.ExitCode
    } catch {
        Write-Warning "[setup] Elevation failed (e.g. headless/non-interactive shell): $_"
        Write-Warning "[setup] Proceeding without elevation. Install steps requiring admin rights may fail."
    }
}

$ErrorActionPreference = "Stop"

Write-Host "[setup] Connect4 pipeline setup starting..."
Write-Host "[setup] Project root: $PSScriptRoot"
Write-Host ""

# --- Helper: Refresh environment PATH from registry ------------------------
function Refresh-Environment {
    Write-Host "[setup] Refreshing environment variables from registry..."
    $machinePath = [Environment]::GetEnvironmentVariable("PATH", "Machine")
    $userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    
    $newPath = @()
    foreach ($p in ($machinePath + ";" + $userPath).Split(';')) {
        $pClean = $p.Trim()
        if ($pClean -and $newPath -notcontains $pClean) {
            $newPath += $pClean
        }
    }
    $env:PATH = $newPath -join ';'
}

# --- Helper: Add to User PATH permanently and update current session ------
function Add-ToUserPath {
    param(
        [string]$PathToAdd
    )
    if (-not (Test-Path $PathToAdd)) {
        Write-Warning "[setup] Cannot add path that does not exist: $PathToAdd"
        return
    }
    
    $userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    $paths = $userPath.Split(';') | ForEach-Object { $_.Trim() }
    
    if ($paths -notcontains $PathToAdd.Trim()) {
        Write-Host "[setup] Adding '$PathToAdd' to User PATH permanently..."
        $newUserPath = ($paths + $PathToAdd.Trim()) -join ';'
        [Environment]::SetEnvironmentVariable("PATH", $newUserPath, "User")
        Refresh-Environment
    } else {
        # Ensure it's in the current process path too
        $processPaths = $env:PATH.Split(';') | ForEach-Object { $_.Trim() }
        if ($processPaths -notcontains $PathToAdd.Trim()) {
            $env:PATH = "$PathToAdd;$env:PATH"
        }
    }
}

# Derive the original user's home directory from the script path.
# This prevents losing the user's context when running elevated (where $env:USERPROFILE might point to Administrator).
$realUserHome = $env:USERPROFILE
if ($PSScriptRoot -match '^([a-zA-Z]:\\Users\\[^\\]+)') {
    $realUserHome = $Matches[1]
}

# Start logging to setup.log in the script directory
$logFile = Join-Path $PSScriptRoot "setup.log"
Write-Host "[setup] Logging output to $logFile"
Start-Transcript -Path $logFile -Append -ErrorAction SilentlyContinue

try {
    Write-Host "[setup] Connect4 pipeline setup starting..."
    Write-Host "[setup] Project root: $PSScriptRoot"
    Write-Host "[setup] Derived original user home directory: $realUserHome"
    Write-Host ""

    # --- 1. Python Check & Install ----------------------------------------------
    $pythonExe = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonExe) {
        Write-Host "[setup] Python not found on active PATH. Searching common locations..."
        $localPython = Get-ChildItem "$realUserHome\AppData\Programs\Python" -Filter "python.exe" -Recurse -Depth 2 -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($localPython) {
            $pyDir = Split-Path $localPython.FullName
            Write-Host "[setup] Found Python installed at $pyDir. Registering in PATH."
            Add-ToUserPath $pyDir
            Add-ToUserPath (Join-Path $pyDir "Scripts")
        } else {
            Write-Host "[setup] Python not found. Installing Python 3.11 via winget..."
            winget install Python.Python.3.11 --accept-source-agreements --accept-package-agreements
            Refresh-Environment
        }
    } else {
        Write-Host "[setup] Python is already available: $($pythonExe.Source)"
    }

    # --- 2. Rust/Cargo Check & Install -------------------------------------------
    $cargoExe = Get-Command cargo -ErrorAction SilentlyContinue
    if (-not $cargoExe) {
        $cargoBin = Join-Path $realUserHome ".cargo\bin"
        if (Test-Path $cargoBin) {
            Write-Host "[setup] Found cargo at $cargoBin. Registering in PATH."
            Add-ToUserPath $cargoBin
        } else {
            Write-Host "[setup] Cargo not found. Installing Rustup via winget..."
            winget install Rustlang.Rustup --accept-source-agreements --accept-package-agreements
            Refresh-Environment
            if (Test-Path $cargoBin) {
                Add-ToUserPath $cargoBin
            }
        }
    } else {
        Write-Host "[setup] Rust is already available: $($cargoExe.Source)"
    }

    # Run rustup to ensure toolchain is stable
    if (Get-Command rustup -ErrorAction SilentlyContinue) {
        Write-Host "[setup] Ensuring stable Rust toolchain..."
        rustup default stable
    }

    # --- 3. CUDA Toolkit 12.6 -----------------------------------------------------
    $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
    if (Test-Path (Join-Path $cudaRoot "bin\cublasLt64_12.dll")) {
        Write-Host "[setup] CUDA 12.6 already installed at $cudaRoot - registering in PATH."
        Add-ToUserPath (Join-Path $cudaRoot "bin")
    } else {
        $installer = "$realUserHome\Downloads\cuda_12.6.3_windows_network.exe"
        if (-not (Test-Path $installer)) {
            Write-Host "[setup] Downloading CUDA 12.6 network installer (~30 MB)..."
            Invoke-WebRequest -Uri "https://developer.download.nvidia.com/compute/cuda/12.6.3/network_installers/cuda_12.6.3_windows_network.exe" -OutFile $installer
        }
        Write-Host "[setup] Running CUDA 12.6 installer (silent install, may take a few minutes)..."
        $proc = Start-Process -FilePath $installer -ArgumentList "-s -n" -Wait -PassThru -NoNewWindow
        if ($proc.ExitCode -ne 0) {
            Write-Warning "[setup] CUDA installer exited with code $($proc.ExitCode)"
        }
        if (-not (Test-Path (Join-Path $cudaRoot "bin\cublasLt64_12.dll"))) {
            Write-Error "[setup] FAILED: CUDA 12.6 install did not produce expected DLL at $cudaRoot\bin"
            Write-Error "[setup] Please run the installer at $installer manually to see any error messages."
            exit 1
        }
        Write-Host "[setup] CUDA 12.6 installed successfully."
        Add-ToUserPath (Join-Path $cudaRoot "bin")
    }

    # --- 4. Python Deps & cuDNN --------------------------------------------------
    # Check if uv is installed, otherwise install it via pip
    $uvExe = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uvExe) {
        Write-Host "[setup] uv package installer not found. Installing uv via pip..."
        python -m pip install uv
        Refresh-Environment
    } else {
        Write-Host "[setup] uv is already available: $($uvExe.Source)"
    }

    # Detect if we are in a virtual environment
    $inVenv = python -c "import sys; print(sys.prefix != sys.base_prefix)" 2>$null
    $uvArgs = @("pip", "install")
    if ($inVenv -ne "True") {
        Write-Host "[setup] Running outside of virtual environment. Using --system flag for uv."
        $uvArgs += "--system"
    }
    $uvArgs += @("-r", "$PSScriptRoot\src_python\requirements.txt")

    Write-Host "[setup] Installing Python dependencies from requirements.txt via uv..."
    uv @uvArgs
    Write-Host "[setup] Python dependencies installed."

    # Find and register cuDNN bin path from the installed pip package
    Write-Host "[setup] Locating nvidia-cudnn-cu12 package path..."
    $cudnnPath = python -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__))" 2>$null
    if ($cudnnPath) {
        # Find the bin folder containing the DLLs
        $cudnnDll = Get-ChildItem -Path $cudnnPath -Filter "cudnn64_9.dll" -Recurse -File -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($cudnnDll) {
            $cudnnBinPath = Split-Path $cudnnDll.FullName
            Write-Host "[setup] Found cuDNN bin path at $cudnnBinPath."
            Add-ToUserPath $cudnnBinPath
        } else {
            Write-Warning "[setup] Could not locate cudnn64_9.dll inside $cudnnPath."
        }
    } else {
        Write-Warning "[setup] nvidia-cudnn-cu12 is not imported correctly."
    }

    # --- Verification & Summary -------------------------------------------------
    Write-Host ""
    Write-Host "================================================================="
    Write-Host "  SETUP VERIFICATION"
    Write-Host "================================================================="
    Refresh-Environment

    $pyVersion = & python --version 2>&1
    $cargoVersion = & cargo --version 2>&1
    Write-Host "[verify] Python: $pyVersion"
    Write-Host "[verify] Rust:   $cargoVersion"

    $cudaBinPath = Join-Path $cudaRoot "bin"
    if (($env:PATH.Split(';') | ForEach-Object { $_.Trim() }) -contains $cudaBinPath) {
        Write-Host "[verify] CUDA bin on active PATH: Yes"
    } else {
        Write-Warning "[verify] CUDA bin on active PATH: NO (Ensure you reopen your terminal)"
    }

    Write-Host "================================================================="
    Write-Host "[setup] Completed successfully! Please RESTART your terminal/editor to reload the updated PATH."
    Write-Host "================================================================="

} catch {
    Write-Error "[setup] An error occurred during setup: $_"
    exit 1
} finally {
    Stop-Transcript -ErrorAction SilentlyContinue
    
    # Prompt the user to press Enter if the window is elevated or run non-interactively, so it doesn't auto-close
    if ([System.Environment]::UserInteractive -and ($args -contains '-Elevated' -or $env:prompt -eq $null)) {
        Write-Host ""
        Write-Host "Press Enter to exit..." -ForegroundColor Cyan
        [void](Read-Host)
    }
}