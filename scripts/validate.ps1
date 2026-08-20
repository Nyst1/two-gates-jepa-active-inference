$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$command = Join-Path $projectRoot ".venv\Scripts\two-gates.exe"
& $command validate --samples 5000

