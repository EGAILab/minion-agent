param(
    [Parameter(Mandatory=$true)][string]$Logs,
    [string]$GateTarget = 'E:/AI/Projects/OpenMinds/Minions/Minion-Agent/.tmp/cache/cargo-target/linux',
    [string]$ControlTarget = 'E:/AI/Projects/OpenMinds/Minions/Minion-Agent/.tmp/cache/cargo-target/linux-controls',
    [ValidateSet('all','gates','controls')][string]$Phase = 'all'
)
$ErrorActionPreference = 'Stop'
$root = 'E:/AI/Projects/OpenMinds/Minions/Minion-Agent'
. ($root + '/.agents/project-environment.ps1')
. ($root + '/.agents/gate-guard.ps1')
$preflight = & ($root + '/.agents/gate-preflight.ps1') -MaxWaitMinutes 0
$preflightExit = $LASTEXITCODE
$preflight | Write-Output
if ($preflightExit -ne 0) { throw 'preflight refused' }
$source = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
Assert-Inside $GateTarget | Out-Null
Assert-Inside $ControlTarget | Out-Null
# Resolve existing cache identities on the HOST, including 8.3 aliases; two
# container bind names do not by themselves prove two different host caches.
$resolved = & ($root + '/.tmp/kit-venv/Scripts/python.exe') -c 'import json,sys; from pathlib import Path; print(json.dumps([str(Path(v).resolve(strict=True)) for v in sys.argv[1:]]))' $GateTarget $ControlTarget
if ($LASTEXITCODE -ne 0) { throw 'cache identity inspection failed' }
$identities = $resolved | ConvertFrom-Json
if ($identities.Count -ne 2) { throw 'two cache identities required' }
$GateTarget = $identities[0]
$ControlTarget = $identities[1]
# NEW evidence only. No target is cleaned, moved or truncated.
Assert-Inside $Logs | Out-Null
try { Get-Item -LiteralPath $Logs -Force -ErrorAction Stop | Out-Null; throw 'existing logs' }
catch [System.Management.Automation.ItemNotFoundException] { Guard-Op mkdir $Logs | Out-Null }
[IO.File]::WriteAllText((Assert-Inside (Join-Path $Logs 'preflight.log')), (($preflight -join "`n") + "`nexit $preflightExit`n"))
$mounts = @{
    '/source' = $source
    '/cargo-home' = $root + '/.tmp/cache/cargo-home-linux'
    '/target' = $GateTarget
    '/control-target' = $ControlTarget
    '/build' = $root + '/.toolchain/icu-78.3-linux'
    '/node' = $root + '/.tmp/cache/node-22.15.1-linux/node'
    '/search-artifacts' = $root + '/.toolchain/search-engines/dl'
    '/logs' = $Logs
}
$argsList = @('run','--rm','--read-only','--tmpfs','/tmp:exec,size=8g')
foreach ($entry in $mounts.GetEnumerator()) {
    $actual = (Resolve-Path -LiteralPath $entry.Value).Path
    Assert-Inside $actual | Out-Null
    $suffix = if ($entry.Key -in @('/cargo-home','/target','/control-target','/logs')) { '' } else { ',readonly' }
    $argsList += @('--mount', "type=bind,source=$actual,target=$($entry.Key)$suffix")
}
$gateReal = (Resolve-Path -LiteralPath $GateTarget).Path.TrimEnd('\')
$controlReal = (Resolve-Path -LiteralPath $ControlTarget).Path.TrimEnd('\')
if ($gateReal -eq $controlReal -or $gateReal.StartsWith($controlReal+'\',[StringComparison]::OrdinalIgnoreCase) -or $controlReal.StartsWith($gateReal+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'disjoint gate/control targets required' }
foreach ($platform in @('windows','linux')) {
    $shared = [IO.Path]::GetFullPath($root + '/.tmp/cache/cargo-target/' + $platform).TrimEnd('\')
    if ($controlReal -eq $shared -or $controlReal.StartsWith($shared+'\',[StringComparison]::OrdinalIgnoreCase) -or $shared.StartsWith($controlReal+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'shared gate cache cannot be a controls target' }
}
$argsList += @('-e','TMP=/tmp','-e','TEMP=/tmp','-e','TMPDIR=/tmp','-e','CARGO_BUILD_JOBS=4',
    'sha256:b1b3c9c0d921d7fa0a6d1f9ec7e4eab87f8c8ec97644c3d791450f131dec813f',
    'python3','-B','/source/minion-agent-rust/scripts/path-resolution-gates.py',
    '--logs','/logs/run','--phase',$Phase)
# PowerShell 5.1 must not turn Docker's ordinary stderr into a launcher abort.
$previous = $ErrorActionPreference
try {
    $ErrorActionPreference = 'Continue'
    & docker @argsList
    $result = $LASTEXITCODE
} finally { $ErrorActionPreference = $previous }
if ($result -ne 0) { throw "Linux gate exit $result; inspect evidence before any rerun" }
