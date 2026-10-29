[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$VideoPath,
    [string]$OutputDirectory,
    [ValidateSet('automatic', 'review')]
    [string]$Workflow = 'automatic'
)

$repository = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $repository 'engine\.venv\Scripts\python.exe'
$harness = Join-Path $PSScriptRoot 'test_local_e2e.py'

if (-not (Test-Path -LiteralPath $VideoPath -PathType Leaf)) {
    throw 'VideoPath must identify an existing local video file.'
}
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'The engine virtual environment is missing. Create engine\.venv and install engine\requirements.lock first.'
}

$arguments = @($harness, '--video', (Resolve-Path -LiteralPath $VideoPath).Path, '--workflow', $Workflow)
if ($OutputDirectory) {
    $arguments += @('--output', $OutputDirectory)
}

& $python @arguments
exit $LASTEXITCODE
