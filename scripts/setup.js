/**
 * First-time project setup script.
 *
 * What it does:
 *   1. Locates or creates a Python virtual environment at ./srenv
 *      - Uses `uv` if available (fast), otherwise falls back to `python -m venv`
 *   2. Installs Python dependencies from requirements.txt into the venv
 *      - Uses `uv pip install` if available, otherwise `pip install`
 *   3. Installs frontend Node dependencies (cd frontend && npm install)
 *   4. Validates that a .env file exists (copies .env.example if not)
 *
 * Usage:
 *   node scripts/setup.js
 *   npm run setup
 */

const { spawnSync } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');

const ROOT = path.resolve(__dirname, '..');
const isWin = os.platform() === 'win32';

// ── Helpers ───────────────────────────────────────────────────────────────────
function log(msg)  { console.log(`\n  ${msg}`); }
function ok(msg)   { console.log(`  ✓ ${msg}`); }
function warn(msg) { console.warn(`  ⚠  ${msg}`); }
function die(msg)  { console.error(`  ✗ ${msg}`); process.exit(1); }

function run(cmd, opts = {}) {
  const result = spawnSync(cmd, { shell: true, stdio: 'inherit', cwd: ROOT, ...opts });
  if (result.status !== 0) die(`Command failed: ${cmd}`);
}

function probe(cmd) {
  // Returns true if the command exits 0 (i.e. the tool exists and is runnable)
  const r = spawnSync(cmd, { shell: true, stdio: 'ignore', cwd: ROOT });
  return r.status === 0;
}

// ── 1. Detect uv ──────────────────────────────────────────────────────────────
log('Checking for uv...');
const hasUv = probe('uv --version');
if (hasUv) {
  const uvVer = spawnSync('uv --version', { shell: true, encoding: 'utf8' });
  ok(`Found uv: ${(uvVer.stdout || '').trim()} — will use uv for venv + installs`);
} else {
  warn('uv not found — falling back to python venv + pip.');
  warn('Install uv for faster setup: https://docs.astral.sh/uv/getting-started/installation/');
}

// ── 2. Find system Python (only needed when uv is absent) ─────────────────────
const sysPy = isWin ? 'python' : 'python3';
if (!hasUv) {
  log('Checking for Python...');
  const verResult = spawnSync(sysPy, ['--version'], { encoding: 'utf8' });
  if (verResult.status !== 0) {
    die(`Could not find "${sysPy}". Install Python 3.9+ and make sure it is on your PATH.`);
  }
  const pyVersion = (verResult.stdout || verResult.stderr).trim();
  ok(`Found: ${pyVersion}  (${sysPy})`);
}

// ── 3. Create venv if needed ──────────────────────────────────────────────────
const venvDir = path.join(ROOT, 'srenv');
const venvPy = isWin
  ? path.join(venvDir, 'Scripts', 'python.exe')
  : path.join(venvDir, 'bin', 'python');

if (fs.existsSync(venvPy)) {
  ok('Virtual environment already exists at ./srenv');
} else {
  log('Creating virtual environment at ./srenv ...');
  if (hasUv) {
    run('uv venv srenv');
  } else {
    run(`${sysPy} -m venv srenv`);
  }
  ok('Virtual environment created');
}

// ── 4. Install Python requirements ───────────────────────────────────────────
log('Installing Python requirements...');
const req = path.join(ROOT, 'requirements.txt');
if (!fs.existsSync(req)) die('requirements.txt not found.');

if (hasUv) {
  // uv pip install resolves and installs significantly faster than pip
  run(`uv pip install --python "${venvPy}" -r requirements.txt`);
} else {
  // Upgrade pip first, then install
  run(`"${venvPy}" -m pip install --upgrade pip --quiet`);
  run(`"${venvPy}" -m pip install -r requirements.txt`);
}
ok('Python dependencies installed');

// ── 5. Install frontend Node dependencies ────────────────────────────────────
const frontendDir = path.join(ROOT, 'frontend');
if (fs.existsSync(frontendDir)) {
  log('Installing frontend Node dependencies...');
  run('npm install', { cwd: frontendDir });
  ok('Frontend dependencies installed');
} else {
  warn('frontend/ directory not found — skipping frontend install.');
}

// ── 6. Ensure .env exists ─────────────────────────────────────────────────────
const envFile = path.join(ROOT, '.env');
const envExample = path.join(ROOT, '.env.example');
if (!fs.existsSync(envFile)) {
  if (fs.existsSync(envExample)) {
    fs.copyFileSync(envExample, envFile);
    warn('.env did not exist — copied from .env.example.');
    warn('Open .env and set your GROQ_API_KEY before starting.');
  } else {
    warn('.env not found. Create one with GROQ_API_KEY=your_key before starting.');
  }
} else {
  ok('.env file exists');
}

// ── Done ──────────────────────────────────────────────────────────────────────
console.log(`
  ──────────────────────────────────────────
  Setup complete!

  Next steps:
    1. Set GROQ_API_KEY in .env (if not already set)
    2. Run:  npm start
  ──────────────────────────────────────────
`);
