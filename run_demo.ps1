$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Get-PythonCommand {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return @("py", "-3")
    }

    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @("python")
    }

    throw "Python 3 was not found in PATH. Please install Python first."
}

function Invoke-Python {
    param(
        [string[]]$PythonCmd,
        [string[]]$Args
    )

    $pythonExe = $PythonCmd[0]
    $pythonPrefix = @()
    if ($PythonCmd.Length -gt 1) {
        $pythonPrefix = $PythonCmd[1..($PythonCmd.Length - 1)]
    }

    & $pythonExe @pythonPrefix @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $($PythonCmd -join ' ') $($Args -join ' ')"
    }
}

function Ensure-EnvFile {
    $envPath = Join-Path $PSScriptRoot ".env"
    $examplePath = Join-Path $PSScriptRoot ".env.example"

    if (-not (Test-Path $envPath) -and (Test-Path $examplePath)) {
        Copy-Item -LiteralPath $examplePath -Destination $envPath
        Write-Host "Created .env from .env.example" -ForegroundColor Yellow
    }

    if (-not (Test-Path $envPath)) {
        Set-Content -LiteralPath $envPath -Value "DASHSCOPE_API_KEY="
    }

    $content = Get-Content -LiteralPath $envPath -Raw
    $match = [regex]::Match($content, "(?m)^\s*DASHSCOPE_API_KEY\s*=\s*(.+?)\s*$")
    $currentKey = if ($match.Success) { $match.Groups[1].Value.Trim() } else { "" }

    if ([string]::IsNullOrWhiteSpace($currentKey) -or $currentKey -eq "sk-your-api-key-here") {
        Write-Host ""
        Write-Host "DASHSCOPE_API_KEY is missing." -ForegroundColor Yellow
        $inputKey = Read-Host "Enter your DashScope API key (it will be saved to .env)"

        if ([string]::IsNullOrWhiteSpace($inputKey)) {
            throw "No DASHSCOPE_API_KEY was provided. Startup cancelled."
        }

        $newLine = "DASHSCOPE_API_KEY=$inputKey"
        if ($match.Success) {
            $updated = [regex]::Replace($content, "(?m)^\s*DASHSCOPE_API_KEY\s*=.*$", $newLine)
        }
        else {
            $updated = $content.TrimEnd() + [Environment]::NewLine + $newLine + [Environment]::NewLine
        }

        Set-Content -LiteralPath $envPath -Value $updated -Encoding UTF8
        Write-Host ".env updated." -ForegroundColor Green
    }
}

function Ensure-Vectorstore {
    $indexFile = Join-Path $PSScriptRoot "faiss_index\index.faiss"
    $metaFile = Join-Path $PSScriptRoot "faiss_index\index.pkl"

    if ((Test-Path $indexFile) -and (Test-Path $metaFile)) {
        Write-Host "Existing vector index found. Skipping rebuild." -ForegroundColor Green
        return
    }

    Write-Step "Vector index not found. Building now"
    Invoke-Python -PythonCmd $script:PythonCmd -Args @("build_vectorstore.py")
}


$script:PythonCmd = Get-PythonCommand

Write-Step "Checking Python"
Invoke-Python -PythonCmd $PythonCmd -Args @("--version")

Write-Step "Checking API key"
Ensure-EnvFile

Write-Step "Installing dependencies"
Invoke-Python -PythonCmd $PythonCmd -Args @("-m", "pip", "install", "-r", "requirements.txt")

Write-Step "Checking vector index"
Ensure-Vectorstore

Write-Step "Starting Streamlit demo"
Write-Host "The browser should open shortly." -ForegroundColor Green
Invoke-Python -PythonCmd $PythonCmd -Args @("-m", "streamlit", "run", "app.py")
