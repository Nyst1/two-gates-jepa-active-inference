$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$command = Join-Path $projectRoot ".venv\Scripts\two-gates.exe"
& $command train --transitions 50000 --epochs 5 --batch-size 512 --seeds 11 29 47

