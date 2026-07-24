# =============================================================================
# setup.ps1 - Interactive & transparent setup for Connect4 pipeline on Windows.
#
# Installs & Configures:
#   1. Python 3.11+ & packages (via src_python/requirements.txt)
#   2. Rust toolchain (via rustup/cargo)
#   3. CUDA Toolkit 12.x (detects existing v12.x or guides install)
#   4. cuDNN 9.x (via pip wheel) & registers DLL path into User PATH
#   5. uv package manager (detects or installs)
# =============================================================================

$ErrorActionPreference = "Stop"

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  Connect4 Pipeline Setup" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "[setup] Project root: $PSScriptRoot"

# Derive original user home directory
$realUserHome = $env:USERPROFILE
if ($PSScriptRoot -match '^([a-zA-Z]:\\Users\\[^\\]+)') {
    $realUserHome = $Matches[1]
}

# --- Helper: Refresh environment PATH from registry ------------------------
function Refresh-Environment {
    $machinePath = [Environment]::GetEnvironmentVariable("PATH", "Machine")
    $userPath    = [Environment]::GetEnvironmentVariable("PATH", "User")
    
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
    param([string]$PathToAdd)
    
    if (-not (Test-Path $PathToAdd)) {
        Write-Warning "[setup] Cannot add path that does not exist: $PathToAdd"
        return
    }
    
    $PathToAddClean = $PathToAdd.Trim()
    $userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    $paths = if ($userPath) { $userPath.Split(';') | ForEach-Object { $_.Trim() } } else { @() }
    
    if ($paths -notcontains $PathToAddClean) {
        Write-Host "[PATH] Adding to User PATH: $PathToAddClean" -ForegroundColor Green
        $newUserPath = ($paths + $PathToAddClean) -join ';'
        [Environment]::SetEnvironmentVariable("PATH", $newUserPath, "User")
        Refresh-Environment
    }
    
    # Ensure active session has it too
    $processPaths = $env:PATH.Split(';') | ForEach-Object { $_.Trim() }
    if ($processPaths -notcontains $PathToAddClean) {
        $env:PATH = "$PathToAddClean;$env:PATH"
    }
}

try {
    # -------------------------------------------------------------------------
    # STEP 1: Python Executable & Virtual Environment
    # -------------------------------------------------------------------------
    Write-Host "`n[1/5] Checking Python environment..." -ForegroundColor Yellow
    $venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    $pythonExePath = $null

    if (Test-Path $venvPy) {
        Write-Host "  -> Found local venv: $venvPy" -ForegroundColor Green
        $pythonExePath = $venvPy
    } else {
        $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
        if ($pythonCmd) {
            $pythonExePath = $pythonCmd.Source
            Write-Host "  -> Using system Python: $pythonExePath" -ForegroundColor Green
        } else {
            $localPython = Get-ChildItem "$realUserHome\AppData\Local\Programs\Python" -Filter "python.exe" -Recurse -Depth 2 -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($localPython) {
                $pyDir = Split-Path $localPython.FullName
                Add-ToUserPath $pyDir
                Add-ToUserPath (Join-Path $pyDir "Scripts")
                $pythonExePath = $localPython.FullName
                Write-Host "  -> Found Python at: $pythonExePath" -ForegroundColor Green
            }
        }
    }

    if (-not $pythonExePath) {
        Write-Host "  -> Python not found. Installing Python 3.11 via winget..." -ForegroundColor Yellow
        winget install Python.Python.3.11 --accept-source-agreements --accept-package-agreements
        Refresh-Environment
        $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
        if ($pythonCmd) { $pythonExePath = $pythonCmd.Source }
    }

    # -------------------------------------------------------------------------
    # STEP 2: Rust & Cargo
    # -------------------------------------------------------------------------
    Write-Host "`n[2/5] Checking Rust toolchain..." -ForegroundColor Yellow
    $cargoCmd = Get-Command cargo -ErrorAction SilentlyContinue
    if (-not $cargoCmd) {
        $cargoBin = Join-Path $realUserHome ".cargo\bin"
        if (Test-Path $cargoBin) {
            Write-Host "  -> Found Cargo at $cargoBin (adding to PATH)" -ForegroundColor Green
            Add-ToUserPath $cargoBin
        } else {
            Write-Host "  -> Cargo not found. Installing Rustup via winget..." -ForegroundColor Yellow
            winget install Rustlang.Rustup --accept-source-agreements --accept-package-agreements
            Refresh-Environment
            if (Test-Path $cargoBin) { Add-ToUserPath $cargoBin }
        }
    } else {
        Write-Host "  -> Cargo available: $($cargoCmd.Source)" -ForegroundColor Green
    }

    # -------------------------------------------------------------------------
    # STEP 3: CUDA Toolkit 12.x
    # -------------------------------------------------------------------------
    Write-Host "`n[3/5] Checking CUDA Toolkit 12.x..." -ForegroundColor Yellow
    $cudaBin = $null
    if (Test-Path "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA") {
        $cudaRoot = Get-ChildItem "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA" -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '^v1[02]\.' } |
            Where-Object { Test-Path (Join-Path $_.FullName 'bin\cublasLt64_12.dll') } |
            Sort-Object { $_.Name } -Descending |
            Select-Object -First 1 -ExpandProperty FullName
        if ($cudaRoot) {
            $cudaBin = Join-Path $cudaRoot 'bin'
        }
    }

    if ($cudaBin) {
        Write-Host "  -> CUDA 12.x Toolkit found: $cudaBin" -ForegroundColor Green
        Add-ToUserPath $cudaBin
    } else {
        Write-Warning "  -> CUDA 12.x Toolkit with cublasLt64_12.dll not found."
        Write-Host "  -> Download & install CUDA Toolkit 12.x from NVIDIA if not installed." -ForegroundColor Yellow
    }

    # -------------------------------------------------------------------------
    # STEP 4: Package Manager (`uv`)
    # -------------------------------------------------------------------------
    Write-Host "`n[4/5] Checking 'uv' package installer..." -ForegroundColor Yellow
    $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uvCmd) {
        $cargoUv = Join-Path $realUserHome ".cargo\bin\uv.exe"
        $localBinUv = Join-Path $realUserHome "AppData\Local\bin\uv.exe"
        
        if (Test-Path $cargoUv) {
            Add-ToUserPath (Split-Path $cargoUv)
            $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
        } elseif (Test-Path $localBinUv) {
            Add-ToUserPath (Split-Path $localBinUv)
            $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
        } else {
            Write-Host "  -> Installing uv via pip..." -ForegroundColor Yellow
            & $pythonExePath -m pip install uv
            Refresh-Environment
            $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
        }
    }

    if ($uvCmd) {
        Write-Host "  -> uv package installer ready: $($uvCmd.Source)" -ForegroundColor Green
    }

    # -------------------------------------------------------------------------
    # STEP 5: Python Dependencies & cuDNN
    # -------------------------------------------------------------------------
    Write-Host "`n[5/5] Installing Python dependencies (PyTorch CUDA, ONNX, cuDNN...)" -ForegroundColor Yellow
    $reqFile = Join-Path $PSScriptRoot "src_python\requirements.txt"
    if (Test-Path $reqFile) {
        $maxRetries = 3
        $success = $false
        for ($i = 1; $i -le $maxRetries; $i++) {
            Write-Host "  -> Running uv pip install (Attempt $i/$maxRetries)..." -ForegroundColor Cyan
            
            # Temporary set ErrorActionPreference to Continue for native command execution
            $oldEA = $global:ErrorActionPreference
            $global:ErrorActionPreference = "Continue"
            
            if ($uvCmd) {
                if (Test-Path $venvPy) {
                    & uv pip install --python $pythonExePath -r $reqFile
                } else {
                    & uv pip install --system -r $reqFile
                }
            } else {
                & $pythonExePath -m pip install -r $reqFile
            }
            $exitCode = $LASTEXITCODE
            $global:ErrorActionPreference = $oldEA
            
            if ($exitCode -eq 0) {
                $success = $true
                break
            } else {
                Write-Warning "  -> Download/install failed (exit code $exitCode). Retrying in 3 seconds..."
                Start-Sleep -Seconds 3
            }
        }

        if (-not $success) {
            Write-Error "[setup] Failed to install Python dependencies after $maxRetries attempts."
            exit 1
        }
        Write-Host "  -> Dependencies installed successfully!" -ForegroundColor Green
    }

    # Locate and register cuDNN bin path
    Write-Host "`n[cuDNN] Registering cuDNN 9.x runtime DLLs into PATH..." -ForegroundColor Yellow
    $cudnnLoc = & $pythonExePath -c "import importlib.metadata, os; dist = importlib.metadata.distribution('nvidia-cudnn-cu12'); print(os.path.join(os.path.dirname(dist._path), 'nvidia', 'cudnn', 'bin'))" 2>$null
    $cudnnLoc = ($cudnnLoc | Select-Object -Last 1).Trim()

    if ($cudnnLoc -and (Test-Path $cudnnLoc)) {
        Write-Host "  -> Located cuDNN DLL folder: $cudnnLoc" -ForegroundColor Green
        Add-ToUserPath $cudnnLoc
    } else {
        Write-Warning "  -> Could not locate nvidia-cudnn-cu12 bin folder. Make sure nvidia-cudnn-cu12 is installed."
    }

    # -------------------------------------------------------------------------
    # VERIFICATION SUMMARY
    # -------------------------------------------------------------------------
    Write-Host "`n=================================================================" -ForegroundColor Cyan
    Write-Host "  SETUP VERIFICATION SUMMARY" -ForegroundColor Cyan
    Write-Host "=================================================================" -ForegroundColor Cyan
    Refresh-Environment

    $pyVer = & $pythonExePath --version 2>&1
    Write-Host "  * Python:   $pyVer ($pythonExePath)" -ForegroundColor Green
    if (Get-Command cargo -ErrorAction SilentlyContinue) {
        $cargoVer = & cargo --version 2>&1
        Write-Host "  * Rust:     $cargoVer" -ForegroundColor Green
    }
    if ($cudaBin) {
        Write-Host "  * CUDA:     $cudaBin" -ForegroundColor Green
    }
    if ($cudnnLoc -and (Test-Path $cudnnLoc)) {
        Write-Host "  * cuDNN:    $cudnnLoc" -ForegroundColor Green
    }

    Write-Host "=================================================================" -ForegroundColor Cyan
    Write-Host "[setup] Complete! You can now run .\bench_cycle.ps1 or .\run_pipeline.ps1" -ForegroundColor Green
    Write-Host "=================================================================" -ForegroundColor Cyan

} catch {
    Write-Error "[setup] An error occurred during setup: $_"
    exit 1
}
