param(
    [string]$Version = "1.1.0",
    [string]$OutDir = "data\ms-cxr\1.1.0"
)

$ErrorActionPreference = "Stop"

$files = @(
    "MS_CXR_Local_Alignment_v$Version.json",
    "MS_CXR_Local_Alignment_v$Version.csv",
    "convert_coco_json_to_csv.py"
)

New-Item -ItemType Directory -Force $OutDir | Out-Null

Write-Host "MS-CXR is a PhysioNet credentialed-access dataset."
Write-Host "Use your PhysioNet account that has completed credentialing and signed the DUA."
Write-Host ""

$username = Read-Host "PhysioNet username"
$password = Read-Host "PhysioNet password" -AsSecureString
$credential = New-Object System.Management.Automation.PSCredential($username, $password)

$baseUrl = "https://physionet.org/files/ms-cxr/$Version"

foreach ($file in $files) {
    $url = "$baseUrl/$file"
    $out = Join-Path $OutDir $file
    Write-Host "Downloading $url"
    Invoke-WebRequest -Uri $url -OutFile $out -Credential $credential -MaximumRedirection 5
    Write-Host "Saved $out"
}

Write-Host ""
Write-Host "Done. Downloaded files:"
Get-ChildItem $OutDir | Select-Object Name, Length

Write-Host ""
Write-Host "Note: MS-CXR only provides phrase-bounding-box annotations."
Write-Host "The actual CXR images must be downloaded separately from MIMIC-CXR-JPG."
