const { spawnSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');

const projectRoot = path.resolve(__dirname, '..', '..', '..');
const sourceRoot = path.join(projectRoot, 'src', 'server');
const platformNames = { linux: 'linux', darwin: 'mac', win32: 'win' };
const platformName = platformNames[process.platform];

if (!platformName) {
  console.error(`Unsupported build platform: ${process.platform}`);
  process.exit(1);
}

const pythonCandidates = [
  process.env.ARBRELLIS_PYTHON,
  process.platform === 'win32'
    ? path.join(projectRoot, '.venv', 'Scripts', 'python.exe')
    : path.join(projectRoot, '.venv', 'bin', 'python'),
].filter(Boolean);
const python = pythonCandidates.find(candidate => fs.existsSync(candidate));

if (!python) {
  console.error('Python was not found. Set ARBRELLIS_PYTHON to the project interpreter.');
  process.exit(1);
}

const target = `${platformName}-${process.arch}`;
const distPath = path.join(projectRoot, 'dist-server', target);
const workPath = path.join(projectRoot, 'build-server', target);
const addDataSeparator = path.delimiter;
const addData = (source, destination) => `${source}${addDataSeparator}${destination}`;
const args = [
  '-m', 'PyInstaller',
  '--noconfirm',
  '--clean',
  '--onedir',
  '--contents-directory', 'lib',
  '--name', 'arbrellis-server',
  '--distpath', distPath,
  '--workpath', workPath,
  '--specpath', workPath,
  '--paths', sourceRoot,
  '--collect-all', 'fastembed',
  '--add-data', addData(path.join(sourceRoot, 'graph_view.html'), '.'),
  '--add-data', addData(path.join(sourceRoot, 'document_capture.html'), '.'),
  '--add-data', addData(path.join(projectRoot, 'node_modules', 'katex', 'dist'), 'katex'),
  path.join(sourceRoot, 'document_capture_server.py'),
];

console.log(`Building ${target} service with ${python}`);
const result = spawnSync(python, args, {
  cwd: projectRoot,
  env: process.env,
  stdio: 'inherit',
});
if (result.error) {
  console.error(result.error.message);
  process.exit(1);
}
process.exit(result.status ?? 1);
