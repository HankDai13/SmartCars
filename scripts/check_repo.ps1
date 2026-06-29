$ErrorActionPreference = "Stop"

$required = @(
    "README.md",
    "configs/models.yaml",
    "configs/topics.yaml",
    "configs/vehicle.yaml",
    "ml/yolov5/README.md",
    "ml/lfnet/train.py",
    "ml/lanenet_hnet/train.py",
    "ros2_ws/src/smartcar_bringup/launch/smartcar_1_15.launch.py",
    "docs/DEPLOYMENT_ATLAS_ROS2.md",
    "docs/TRAINING_PIPELINES.md"
)

$missing = @()
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path)) {
        $missing += $path
    }
}

if ($missing.Count -gt 0) {
    Write-Error "Missing required files:`n$($missing -join "`n")"
}

$pythonFiles = Get-ChildItem -Recurse -Include *.py -File |
    Where-Object { $_.FullName -notmatch "\\external\\" -and $_.FullName -notmatch "\\.venv\\" }

foreach ($file in $pythonFiles) {
    python -m py_compile $file.FullName
}

Write-Host "Repository scaffold check passed."

