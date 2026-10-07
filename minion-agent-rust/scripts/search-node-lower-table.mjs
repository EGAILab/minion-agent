// Reproduce search_node_lower.json with pinned Node; stdout is the artifact.
if (process.version !== 'v22.15.1' || process.versions.unicode !== '16.0') {
  throw new Error(`Wrong authority runtime: ${process.version}/${process.versions.unicode}`);
}
const lower = [], cased = [], ignorable = [];
for (let n = 0; n <= 0x10ffff; n++) {
  if (n >= 0xd800 && n <= 0xdfff) continue;
  const value = String.fromCodePoint(n), mapped = value.toLowerCase();
  if (mapped !== value) lower.push([n, Array.from(mapped, c => c.codePointAt(0))]);
  if (/\p{Cased}/u.test(value)) cased.push(n);
  if (/\p{Case_Ignorable}/u.test(value)) ignorable.push(n);
}
function ranges(values) {
  const result = [];
  for (const n of values) {
    if (result.length && result[result.length - 1][1] + 1 === n) result[result.length - 1][1] = n;
    else result.push([n, n]);
  }
  return result;
}
process.stdout.write(JSON.stringify({ node: process.version, unicode: process.versions.unicode,
  lower, cased: ranges(cased), ignorable: ranges(ignorable) }) + '\n');
