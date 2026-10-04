// Real inherited-pipe witness. Parent exit is gated on the descendant's readiness.
const fs = require('node:fs');
const { spawn } = require('node:child_process');
const child = spawn(process.execPath, ['-e', 'console.log("descendant-ready");process.send("ready");setTimeout(()=>{},30000)'], {
  detached: true, stdio: ['ignore', 1, 2, 'ipc']
});
fs.writeFileSync('descendant.pid', String(child.pid));
child.on('message', () => process.exit(0));
