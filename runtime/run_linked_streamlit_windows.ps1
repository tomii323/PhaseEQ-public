param(
    [Parameter(Mandatory = $true)][string]$PythonExe,
    [Parameter(Mandatory = $true)][string]$MainFile,
    [Parameter(Mandatory = $true)][string]$CompositeFile,
    [Parameter(Mandatory = $true)][int]$MainPort,
    [Parameter(Mandatory = $true)][int]$CompositePort,
    [int]$StartupRetryCount = 60,
    [int]$StartupRetryDelayMilliseconds = 300,
    [int]$StartupRequestTimeoutMilliseconds = 1000
)

$ErrorActionPreference = "Stop"
$composite = $null
$main = $null

function Test-ServiceReady {
    param([Parameter(Mandatory = $true)][string]$Url)

    $request = $null
    $response = $null
    try {
        $request = [System.Net.HttpWebRequest]::CreateHttp($Url)
        $request.Method = "GET"
        $request.Proxy = $null
        $request.Timeout = $StartupRequestTimeoutMilliseconds
        $request.ReadWriteTimeout = $StartupRequestTimeoutMilliseconds
        $response = $request.GetResponse()
        $statusCode = [int]$response.StatusCode
        return $statusCode -ge 200 -and $statusCode -lt 400
    }
    catch {
        return $false
    }
    finally {
        if ($null -ne $response) {
            $response.Dispose()
        }
    }
}

function Wait-ForServices {
    $mainHealthUrl = "http://127.0.0.1:$MainPort/_stcore/health"
    $compositeHealthUrl = "http://127.0.0.1:$CompositePort/_stcore/health"

    Write-Host "Waiting for PhaseEQ and Composite Engine to become ready..."
    for ($attempt = 1; $attempt -le $StartupRetryCount; $attempt++) {
        if ($main.HasExited) {
            throw "PhaseEQ stopped before becoming ready (exit code $($main.ExitCode))."
        }
        if ($composite.HasExited) {
            throw "Composite Engine stopped before becoming ready (exit code $($composite.ExitCode))."
        }

        $mainReady = Test-ServiceReady -Url $mainHealthUrl
        $compositeReady = Test-ServiceReady -Url $compositeHealthUrl
        if ($mainReady -and $compositeReady) {
            Write-Host "PhaseEQ and Composite Engine are ready."
            return
        }

        if ($attempt -lt $StartupRetryCount) {
            Start-Sleep -Milliseconds $StartupRetryDelayMilliseconds
        }
    }

    throw "Startup timed out after $StartupRetryCount attempts. PhaseEQ health: $mainHealthUrl; Composite Engine health: $compositeHealthUrl"
}

try {
    $compositeArguments = @(
        "-m", "streamlit", "run", $CompositeFile,
        "--server.port", [string]$CompositePort,
        "--server.address", "127.0.0.1",
        "--server.headless", "true"
    )
    $composite = Start-Process `
        -FilePath $PythonExe `
        -ArgumentList $compositeArguments `
        -NoNewWindow `
        -PassThru

    $mainArguments = @(
        "-m", "streamlit", "run", $MainFile,
        "--server.port", [string]$MainPort,
        "--server.address", "127.0.0.1",
        "--server.headless", "true"
    )
    $main = Start-Process `
        -FilePath $PythonExe `
        -ArgumentList $mainArguments `
        -NoNewWindow `
        -PassThru

    Wait-ForServices
    Start-Process "http://localhost:$MainPort"
    Start-Process "http://localhost:$CompositePort"

    while (-not $main.WaitForExit(300)) {
        if ($composite.HasExited) {
            throw "Composite Engine stopped unexpectedly (exit code $($composite.ExitCode))."
        }
    }
    exit $main.ExitCode
}
finally {
    if ($null -ne $main -and -not $main.HasExited) {
        Stop-Process -Id $main.Id -Force -ErrorAction SilentlyContinue
        $main.WaitForExit(5000) | Out-Null
    }
    if ($null -ne $composite -and -not $composite.HasExited) {
        Stop-Process -Id $composite.Id -Force -ErrorAction SilentlyContinue
        $composite.WaitForExit(5000) | Out-Null
    }
}
