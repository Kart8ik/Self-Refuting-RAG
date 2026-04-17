/**
 * Cross-platform backend launcher.
 *
 * Python resolution order:
 *   1. PYTHON_BIN environment variable (explicit override)
 *   2. ./srenv/Scripts/python  (Windows venv)
 *   3. ./srenv/bin/python      (macOS/Linux venv)
 *   4. python3 / python        (system fallback)
 *
 * Passes through any extra CLI flags to uvicorn (e.g. --reload).
 *
 * Flags handled by this script:
 *   --log <file>   Tee stdout+stderr to <file> as well as the console.
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');

const ROOT = path.resolve(__dirname, '..');
const isWin = os.platform() === 'win32';

// ── Python resolution ──────────────────────────────────────────────────────────
function findPython() {
  if (process.env.PYTHON_BIN) {
    const p = process.env.PYTHON_BIN;
    if (fs.existsSync(p)) return p;
    console.warn(`PYTHON_BIN="${p}" does not exist, falling through to auto-detect.`);
  }

  const candidates = isWin
    ? [
        path.join(ROOT, 'srenv', 'Scripts', 'python.exe'),
        path.join(ROOT, '.venv', 'Scripts', 'python.exe'),
        'python',
      ]
    : [
        path.join(ROOT, 'srenv', 'bin', 'python'),
        path.join(ROOT, '.venv', 'bin', 'python'),
        'python3',
        'python',
      ];

  for (const c of candidates) {
    if (path.isAbsolute(c)) {
      if (fs.existsSync(c)) return c;
    } else {
      return c; // system command — let the OS resolve it
    }
  }
  return 'python';
}

// ── Argument parsing ──────────────────────────────────────────────────────────
let logFile = null;
const uvicornArgs = [];

const argv = process.argv.slice(2);
for (let i = 0; i < argv.length; i++) {
  if (argv[i] === '--log' && argv[i + 1]) {
    logFile = path.resolve(ROOT, argv[++i]);
  } else {
    uvicornArgs.push(argv[i]);
  }
}

// ── Environment ───────────────────────────────────────────────────────────────
const env = {
  ...process.env,
  // Prevent OpenMP / BLAS / MKL from spawning many threads (avoids FAISS crashes)
  OMP_NUM_THREADS: '1',
  OPENBLAS_NUM_THREADS: '1',
  MKL_NUM_THREADS: '1',
  VECLIB_MAXIMUM_THREADS: '1',
  NUMEXPR_NUM_THREADS: '1',
  TOKENIZERS_PARALLELISM: 'false',
  FAISS_THREADS: '1',
  SR_RAG_FAISS_INDEX: process.env.SR_RAG_FAISS_INDEX || 'flat',
  // Suppress duplicate OpenMP library abort on some Windows + Intel MKL setups
  KMP_DUPLICATE_LIB_OK: 'TRUE',
};

// ── Launch ────────────────────────────────────────────────────────────────────
const python = findPython();
const cmd = ['-m', 'uvicorn', 'chat_api:app', '--host', '0.0.0.0', '--port', '8000', ...uvicornArgs];

console.log(`Backend  : ${python} ${cmd.join(' ')}`);
if (logFile) {
  fs.mkdirSync(path.dirname(logFile), { recursive: true });
  console.log(`Log file : ${logFile}`);
}

const proc = spawn(python, cmd, { cwd: ROOT, env, stdio: logFile ? 'pipe' : 'inherit' });

if (logFile) {
  const stream = fs.createWriteStream(logFile, { flags: 'a' });
  proc.stdout.pipe(process.stdout);
  proc.stderr.pipe(process.stderr);
  proc.stdout.pipe(stream);
  proc.stderr.pipe(stream);
}

proc.on('exit', (code) => process.exit(code ?? 0));
process.on('SIGINT', () => proc.kill('SIGINT'));
process.on('SIGTERM', () => proc.kill('SIGTERM'));
