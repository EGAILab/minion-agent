param(
    [Parameter(Mandatory=$true)][string]$EvidenceDirectory
)
$ErrorActionPreference = 'Stop'
# Run against a committed snapshot in a separate detached worktree. Never edit
# the candidate worktree. The normal pinned ICU/Cargo environment is inherited.
$repository = (git -C $PSScriptRoot rev-parse --show-toplevel).Trim()
$candidate = (git -C $repository rev-parse HEAD).Trim()
$evidence = [IO.Path]::GetFullPath($EvidenceDirectory)
New-Item -ItemType Directory -Path $evidence -Force | Out-Null
$scratch = Join-Path $evidence ('scratch-' + [guid]::NewGuid().ToString('N'))
git -C $repository worktree add --detach $scratch $candidate
if ($LASTEXITCODE -ne 0) { throw 'Cannot create isolated mutant worktree' }
$rust = Join-Path $scratch 'minion-agent-rust'
$src = Join-Path $rust 'crates/minion-agent/src'
$files = @{}
foreach ($name in @('argument_object.rs','tools/prepared.rs','tools/execution.rs','llm/raw.rs')) {
    $files[$name] = [IO.File]::ReadAllText((Join-Path $src $name)).Replace("`r`n","`n")
}
function Change([string]$name,[string]$old,[string]$new) {
    $path = Join-Path $src $name
    $text = [IO.File]::ReadAllText($path).Replace("`r`n","`n")
    if (-not $text.Contains($old)) { throw "Mutant anchor absent: $name / $old" }
    [IO.File]::WriteAllText($path,$text.Replace($old,$new),[Text.UTF8Encoding]::new($false))
}
$sort = [regex]::Match($files['argument_object.rs'],'(?s)        // Stable sorting.*?        previous').Value.Replace('        previous','')
if (-not $sort) { throw 'Order mutation anchor absent' }
$mutants = [ordered]@{
    'insertion-without-index-first' = { Change 'argument_object.rs' $sort '' }
    'sorted-map' = { Change 'argument_object.rs' '(None, None) => std::cmp::Ordering::Equal' '(None, None) => a.code_units().cmp(b.code_units())' }
    'noncanonical-leading-zero' = { Change 'argument_object.rs' ' || (units.len() > 1 && units[0] == 48)' '' }
    'noncanonical-u32-max' = { Change 'argument_object.rs' 'number < u64::from(u32::MAX)' 'number <= u64::from(u32::MAX)' }
    'long-decimal-conversion' = { Change 'argument_object.rs' ' || units.len() > 10' '' }
    'construction-only' = {
        Change 'argument_object.rs' $sort ''
        Change 'argument_object.rs' "        object`n    }`n}" ($sort.Replace('self.0','object.0') + "        object`n    }`n}")
    }
    'sorted-prepared-value' = {
        Change 'tools/prepared.rs' 'for (key, child) in values {' "let mut entries = values.iter().collect::<Vec<_>>();`n                    entries.sort_by(|a,b| a.0.code_units().cmp(b.0.code_units()));`n                    for (key, child) in entries {"
    }
    'sorted-hook-replacement' = {
        Change 'tools/execution.rs' 'arguments: arguments.unwrap_or(current.call.arguments),' 'arguments: arguments.map(|v| { let mut json = v.try_to_json().unwrap(); json.sort_all_objects(); PreparedValue::from(json) }).unwrap_or(current.call.arguments),'
    }
    'event-argument-projection' = {
        Change 'tools/execution.rs' "    arguments.clone()`n}" "    let mut json = arguments.try_to_json().unwrap();`n    json.sort_all_objects();`n    crate::llm::RawValue::from(json)`n}"
    }
    'session-serialization-sorted' = {
        Change 'llm/raw.rs' "        self.try_to_json()`n            .map_err(serde::ser::Error::custom)?`n            .serialize(serializer)" "        let mut json = self.try_to_json().map_err(serde::ser::Error::custom)?;`n        json.sort_all_objects();`n        json.serialize(serializer)"
    }
}
$results = @()
try {
    Push-Location $rust
    try {
        foreach ($entry in $mutants.GetEnumerator()) {
            foreach ($name in $files.Keys) { [IO.File]::WriteAllText((Join-Path $src $name),$files[$name],[Text.UTF8Encoding]::new($false)) }
            & $entry.Value
            $ErrorActionPreference = 'Continue'
            $output = & cargo test -p minion-agent --all-features --offline --lib --test argument_object_order --test argument_graph_identity --test key_order_conformance --no-fail-fast --quiet 2>&1
            $exit = $LASTEXITCODE
            $ErrorActionPreference = 'Stop'
            $log = $output -join "`n"
            [IO.File]::WriteAllText((Join-Path $evidence ($entry.Key+'.log')),$log,[Text.UTF8Encoding]::new($false))
            # A compile failure is not a killed semantic mutant.
            $failed = ([regex]::Matches($log,'(?m)^.+ --- FAILED$')).Count
            $killed = $exit -ne 0 -and $failed -gt 0 -and -not $log.Contains('could not compile')
            $results += [pscustomobject]@{ mutant=$entry.Key; exit=$exit; failed_tests=$failed; killed=$killed; candidate=$candidate }
            Write-Host ($entry.Key + ': ' + $failed + ' failed tests; killed=' + $killed)
        }
    } finally { Pop-Location }
} finally {
    foreach ($name in $files.Keys) { [IO.File]::WriteAllText((Join-Path $src $name),$files[$name],[Text.UTF8Encoding]::new($false)) }
    $results | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $evidence 'results.json') -Encoding UTF8
}
# Keep the restored scratch worktree and logs for independent inspection.
if (($results | Where-Object { -not $_.killed }).Count -gt 0) { throw 'A K1 semantic mutant survived or failed to compile' }
