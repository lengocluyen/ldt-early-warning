[CmdletBinding()]
param(
    [ValidateSet("smoke", "core", "tab_transformer", "modern", "sklearn_modern", "all_modules", "publication", "full_run", "publication_full")]
    [string]$Preset = "full_run",
    [switch]$SkipInstall,
    [switch]$NoVenv,
    [switch]$Help
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Show-Usage {
    @"
Usage: .\run_pipeline.ps1 [-Preset NAME] [-SkipInstall] [-NoVenv]

Presets:
  smoke             DDD Week 4 only; fastest end-to-end sanity check
  full_run          all modules with LR/XGB/XGB-LDT (default)
  publication_full  expensive all-modules modern benchmark with XGB grid search
  core              LR-raw, XGB-raw, XGB-LDT
  tab_transformer   core + TABTX
  sklearn_modern    core + RF, ET, HGB, TABTX
  modern            core + RF, ET, HGB, LightGBM, CatBoost, TABTX
  all_modules       alias of full_run
"@
}

function Find-Python {
    param([bool]$UseVenv)

    $venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if ($UseVenv -and (Test-Path $venvPython)) {
        return $venvPython
    }

    foreach ($candidate in @("python", "py")) {
        $command = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($null -ne $command) {
            return $command.Source
        }
    }

    throw "Python was not found. Install Python 3.10+ and rerun this command."
}

if ($Help) {
    Show-Usage
    exit 0
}

$presetConfig = @{
    smoke              = @{ Config = "config/experiment_smoke.yaml"; Requirements = "requirements-core.txt" }
    core               = @{ Config = "config/experiment.yaml"; Requirements = "requirements-core.txt" }
    tab_transformer    = @{ Config = "config/experiment_tab_transformer.yaml"; Requirements = "requirements-tab-transformer.txt" }
    modern             = @{ Config = "config/experiment_modern_models.yaml"; Requirements = "requirements.txt" }
    sklearn_modern     = @{ Config = "config/experiment_sklearn_modern.yaml"; Requirements = "requirements-tab-transformer.txt" }
    all_modules        = @{ Config = "config/experiment_all_modules.yaml"; Requirements = "requirements-core.txt" }
    publication        = @{ Config = "config/experiment_all_modules.yaml"; Requirements = "requirements-core.txt" }
    full_run           = @{ Config = "config/experiment_all_modules.yaml"; Requirements = "requirements-core.txt" }
    publication_full   = @{ Config = "config/experiment_publication_full.yaml"; Requirements = "requirements.txt" }
}

if ($Preset -eq "publication_full" -and $env:CONFIRM_LONG_RUN -ne "1") {
    [Console]::Error.WriteLine(@"
Preset 'publication_full' is intentionally expensive. It can take days because it runs all modules with modern models and XGBoost grid search.

To run it anyway:
  `$env:CONFIRM_LONG_RUN = "1"
  .\run_pipeline.ps1 -Preset publication_full

For the normal paper tables, use:
  .\run_pipeline.ps1
"@)
    exit 2
}

Set-Location $PSScriptRoot
$selection = $presetConfig[$Preset]
$config = $selection.Config
$requirements = $selection.Requirements

New-Item -ItemType Directory -Force -Path "results/logs" | Out-Null
$runId = "{0:yyyyMMddTHHmmssZ}_{1}" -f (Get-Date).ToUniversalTime(), $Preset
$logFile = Join-Path $PSScriptRoot "results/logs/$runId.log"

Start-Transcript -Path $logFile -Append | Out-Null
try {
    Write-Output "Run ID: $runId"
    Write-Output "Log file: $logFile"
    Write-Output "Started at UTC: $((Get-Date).ToUniversalTime().ToString('o'))"
    Write-Output "Host: $env:COMPUTERNAME"
    Write-Output "Working directory: $(Get-Location)"
    Write-Output "Preset: $Preset"
    Write-Output "Config: $config"
    Write-Output "Requirements: $requirements"

    if (-not $NoVenv -and -not (Test-Path ".venv\Scripts\python.exe")) {
        $systemPython = Find-Python -UseVenv:$false
        Write-Output "Creating virtual environment in .venv ..."
        & $systemPython -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed." }
    }

    $python = Find-Python -UseVenv:(-not $NoVenv)

    if (-not $SkipInstall) {
        Write-Output "Installing requirements ..."
        & $python -m pip install --upgrade pip
        if ($LASTEXITCODE -ne 0) { throw "Pip upgrade failed." }
        & $python -m pip install -r $requirements
        if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
    }

    Write-Output "Running preset '$Preset' with $config ..."
    & $python experiments/run_all.py --config $config
    if ($LASTEXITCODE -ne 0) { throw "Experiment run failed." }

    Write-Output "Aggregating result tables ..."
    & $python experiments/aggregate_results.py --results results --output results/tables
    if ($LASTEXITCODE -ne 0) { throw "Result aggregation failed." }

    Write-Output "Generating result figures ..."
    & $python experiments/visualize_results.py --results results --output results/figures
    if ($LASTEXITCODE -ne 0) { throw "Figure generation failed." }

    Write-Output "Finished at UTC: $((Get-Date).ToUniversalTime().ToString('o'))"
    Write-Output ""
    Write-Output "Done. Main outputs:"
    Write-Output "  results/tables/dataset_distribution.csv"
    Write-Output "  results/tables/incremental_construction.csv"
    Write-Output "  results/tables/predictive_performance.csv"
    Write-Output "  results/tables/explanation_grounding.csv"
    Write-Output "  results/tables/feature_parity.csv"
    Write-Output "  results/tables/experimental_process.csv"
    Write-Output "  results/tables/model_timing.csv"
    Write-Output "  results/tables/training_history.csv"
    Write-Output "  results/tables/publication_readiness.json"
    Write-Output "  results/figures/"
    Write-Output "  results/logs/"
}
finally {
    Stop-Transcript | Out-Null
}
