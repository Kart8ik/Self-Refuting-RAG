/**
 * Cross-platform port killer.
 * Stops any process occupying ports 8000 (backend) or 5173 (frontend).
 * Works on Windows (taskkill), macOS, and Linux (kill).
 */

const { execSync } = require('child_process');
const os = require('os');

const PORTS = [8000, 5173];
const isWin = os.platform() === 'win32';

function killPort(port) {
  try {
    if (isWin) {
      const output = execSync(`netstat -ano`, { encoding: 'utf8', stdio: ['pipe', 'pipe', 'ignore'] });
      const pids = new Set();

      for (const line of output.split('\n')) {
        // Match lines that have our port as the local address (e.g. 0.0.0.0:8000 or [::]:8000)
        if (new RegExp(`[:\\s]${port}\\s`).test(line) && line.includes('LISTENING')) {
          const parts = line.trim().split(/\s+/);
          const pid = parts[parts.length - 1];
          if (pid && /^\d+$/.test(pid) && pid !== '0') {
            pids.add(pid);
          }
        }
      }

      if (pids.size === 0) {
        return; // nothing on this port
      }

      for (const pid of pids) {
        try {
          execSync(`taskkill /F /PID ${pid}`, { stdio: 'ignore' });
          console.log(`  Stopped PID ${pid} on port ${port}`);
        } catch {
          // process may have already exited
        }
      }
    } else {
      const raw = execSync(`lsof -ti tcp:${port} 2>/dev/null || true`, { encoding: 'utf8' }).trim();
      if (!raw) return;
      for (const pid of raw.split('\n').filter(Boolean)) {
        try {
          execSync(`kill -9 ${pid}`);
          console.log(`  Stopped PID ${pid} on port ${port}`);
        } catch {
          // process may have already exited
        }
      }
    }
  } catch {
    // Port was not in use — that's fine
  }
}

console.log('Releasing ports 8000 and 5173...');
for (const port of PORTS) {
  killPort(port);
}
console.log('Done.');
