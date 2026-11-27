const { app, BrowserWindow, dialog, shell } = require('electron');
const { spawn } = require('node:child_process');
const crypto = require('node:crypto');
const http = require('node:http');
const net = require('node:net');
const path = require('node:path');
const fs = require('node:fs');

const SERVER_HOST = '127.0.0.1';
const STARTUP_TIMEOUT_MS = 90_000;

let mainWindow = null;
let serverProcess = null;
let isQuitting = false;
let serverPort = null;
let serverOrigin = null;
const sessionToken = crypto.randomBytes(32).toString('hex');

function repositoryRoot() {
  return path.resolve(__dirname, '..', '..');
}

function existingFile(candidates) {
  return candidates.find(candidate => candidate && fs.existsSync(candidate));
}

function developmentCommand() {
  const root = repositoryRoot();
  const configuredPython = process.env.ARBRELLIS_PYTHON;
  const python = existingFile([
    configuredPython,
    process.platform === 'win32'
      ? path.join(root, '.venv', 'Scripts', 'python.exe')
      : path.join(root, '.venv', 'bin', 'python'),
  ]);
  if (!python) {
    throw new Error(
      'Python was not found. Set ARBRELLIS_PYTHON to the project environment interpreter.',
    );
  }
  return {
    executable: python,
    args: [
      path.join(root, 'src', 'server', 'document_capture_server.py'),
      '--host', SERVER_HOST,
      '--port', String(serverPort),
      '--auth-token', sessionToken,
    ],
    cwd: root,
  };
}

function packagedCommand() {
  const configuredServer = process.env.ARBRELLIS_SERVER;
  const executableName = process.platform === 'win32' ? 'arbrellis-server.exe' : 'arbrellis-server';
  const executable = existingFile([
    configuredServer,
    path.join(process.resourcesPath, 'server', executableName),
  ]);
  if (!executable) {
    throw new Error(
      `The packaged local service was not found. Expected ${path.join(process.resourcesPath, 'server', executableName)}.`,
    );
  }
  const dataDirectory = app.getPath('userData');
  fs.mkdirSync(dataDirectory, { recursive: true });
  return {
    executable,
    args: [
      '--host', SERVER_HOST,
      '--port', String(serverPort),
      '--auth-token', sessionToken,
      '--db', path.join(dataDirectory, 'knowledge-captures.sqlite3'),
    ],
    cwd: dataDirectory,
  };
}

function chooseServerPort() {
  const configured = process.env.ARBRELLIS_PORT;
  if (configured) {
    const value = Number.parseInt(configured, 10);
    if (!Number.isInteger(value) || value < 1024 || value > 65535) {
      throw new Error('ARBRELLIS_PORT must be an integer between 1024 and 65535.');
    }
    return Promise.resolve(value);
  }
  return new Promise((resolve, reject) => {
    const probe = net.createServer();
    probe.unref();
    probe.once('error', reject);
    probe.listen(0, SERVER_HOST, () => {
      const address = probe.address();
      const port = typeof address === 'object' && address ? address.port : null;
      probe.close(error => error ? reject(error) : resolve(port));
    });
  });
}

function healthCheck(timeoutMs = 600) {
  return new Promise(resolve => {
    const request = http.get(`${serverOrigin}/api/health`, {
      timeout: timeoutMs,
      headers: { Authorization: `Bearer ${sessionToken}` },
    }, response => {
      let body = '';
      response.setEncoding('utf8');
      response.on('data', chunk => { body += chunk; });
      response.on('end', () => {
        try {
          const payload = JSON.parse(body);
          resolve(response.statusCode === 200 && payload.status === 'ok');
        } catch {
          resolve(false);
        }
      });
    });
    request.on('timeout', () => request.destroy());
    request.on('error', () => resolve(false));
  });
}

async function waitForServer() {
  const deadline = Date.now() + STARTUP_TIMEOUT_MS;
  while (Date.now() < deadline) {
    if (await healthCheck()) return;
    if (serverProcess && serverProcess.exitCode !== null) {
      throw new Error(`The local service stopped with exit code ${serverProcess.exitCode}.`);
    }
    await new Promise(resolve => setTimeout(resolve, 180));
  }
  throw new Error(`The local service did not become ready within ${STARTUP_TIMEOUT_MS / 1000} seconds.`);
}

async function ensureServer() {
  if (await healthCheck()) return;
  const command = app.isPackaged ? packagedCommand() : developmentCommand();
  serverProcess = spawn(command.executable, command.args, {
    cwd: command.cwd,
    env: { ...process.env, PYTHONUNBUFFERED: '1' },
    stdio: ['ignore', 'inherit', 'inherit'],
    windowsHide: true,
  });
  serverProcess.once('error', error => {
    console.error('Could not start the local service:', error);
  });
  await waitForServer();
}

function graphUrl() {
  const target = new URL(serverOrigin);
  target.searchParams.set('session', sessionToken);
  const rawQuery = process.env.ARBRELLIS_QUERY;
  if (rawQuery) {
    const query = new URLSearchParams(rawQuery.replace(/^\?/, ''));
    for (const [key, value] of query) target.searchParams.set(key, value);
  }
  return target.toString();
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 960,
    minHeight: 640,
    show: false,
    backgroundColor: '#071f19',
    title: 'Arbrellis',
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      backgroundThrottling: false,
    },
  });

  mainWindow.removeMenu();
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith('https://') || url.startsWith('http://')) shell.openExternal(url);
    return { action: 'deny' };
  });
  mainWindow.webContents.on('will-navigate', (event, url) => {
    let origin = null;
    try { origin = new URL(url).origin; } catch { /* blocked below */ }
    if (origin !== serverOrigin) {
      event.preventDefault();
      if (url.startsWith('https://') || url.startsWith('http://')) shell.openExternal(url);
    }
  });
  mainWindow.once('ready-to-show', () => mainWindow?.show());
  mainWindow.on('closed', () => { mainWindow = null; });
  mainWindow.loadFile(path.join(__dirname, 'loading.html'));
  return mainWindow;
}

async function startApplication() {
  if (serverPort === null) {
    serverPort = await chooseServerPort();
    serverOrigin = `http://${SERVER_HOST}:${serverPort}`;
  }
  const window = createWindow();
  try {
    await ensureServer();
    if (!window.isDestroyed()) await window.loadURL(graphUrl());
  } catch (error) {
    console.error(error);
    if (!window.isDestroyed()) window.destroy();
    await dialog.showMessageBox({
      type: 'error',
      title: 'Arbrellis could not start',
      message: 'The local knowledge service could not be started.',
      detail: error instanceof Error ? error.message : String(error),
    });
    app.quit();
  }
}

function stopOwnedServer() {
  if (!serverProcess || serverProcess.killed) return;
  serverProcess.kill();
  serverProcess = null;
}

const hasSingleInstanceLock = app.requestSingleInstanceLock();
if (!hasSingleInstanceLock) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (!mainWindow) return;
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.focus();
  });

  app.whenReady().then(startApplication);

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) startApplication();
  });

  app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') app.quit();
  });

  app.on('before-quit', () => {
    isQuitting = true;
    stopOwnedServer();
  });

  app.on('will-quit', () => {
    if (!isQuitting) stopOwnedServer();
  });
}
