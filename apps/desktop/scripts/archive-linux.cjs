const { spawnSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');

if (process.platform !== 'linux') {
  console.error('The FUSE-free Linux archive must be produced on Linux.');
  process.exit(1);
}

const desktopRoot = path.resolve(__dirname, '..');
const distDirectory = path.join(desktopRoot, 'dist');
const unpackedDirectory = path.join(distDirectory, 'linux-unpacked');
const output = path.join(distDirectory, 'arbrellis-0.1.0-linux-x64.tar.gz');

if (!fs.existsSync(path.join(unpackedDirectory, 'arbrellis'))) {
  console.error('The unpacked Linux application is missing. Run electron-builder first.');
  process.exit(1);
}
if (fs.existsSync(output)) fs.rmSync(output);

const result = spawnSync(
  'tar',
  [
    '-C', distDirectory,
    '--use-compress-program=gzip -1',
    '--transform', 's,^linux-unpacked,Arbrellis-0.1.0,',
    '-cf', output,
    'linux-unpacked',
  ],
  { stdio: 'inherit' },
);
if (result.error) {
  console.error(result.error.message);
  process.exit(1);
}
process.exit(result.status ?? 1);
