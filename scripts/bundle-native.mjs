import { execFileSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, readdirSync } from 'node:fs';
import { basename, delimiter, join, resolve } from 'node:path';

const sourceArg = process.argv[2];
const destArg = process.argv[3];
const cross = flagValue('--cross');
const cuda = process.argv.includes('--cuda');

if (!sourceArg || !destArg) {
    console.error('usage: node scripts/bundle-native.mjs <tsgrad_native.node> <dest> [--cuda] [--cross triple]');
    process.exit(1);
}

const nodeFile = resolve(sourceArg);
const dest = resolve(destArg);
if (!existsSync(nodeFile)) {
    console.error(`native addon not found: ${nodeFile}`);
    process.exit(1);
}

mkdirSync(dest, { recursive: true });
const shipped = join(dest, 'tsgrad_native.node');
copyFileSync(nodeFile, shipped);

if (process.platform === 'win32') bundleWindows(shipped, dest);
else if (process.platform === 'darwin') bundleMac(shipped, dest);
else if (cross) bundleCross(shipped, cross);
else bundleLinux(shipped, dest, cuda);

function flagValue(name) {
    const index = process.argv.indexOf(name);
    return index === -1 ? '' : process.argv[index + 1] ?? '';
}

function run(command, args, env = process.env) {
    return execFileSync(command, args, { encoding: 'utf8', env, stdio: ['ignore', 'pipe', 'pipe'] });
}

function copyRuntime(source, directory, name = basename(source)) {
    const target = join(directory, name);
    copyFileSync(source, target);
    console.log(`bundled ${name}`);
    return target;
}

const WINDOWS_SYSTEM = /^(kernel32|kernelbase|user32|advapi32|shell32|ole32|oleaut32|ws2_32|bcrypt|ntdll|dbghelp|psapi|shlwapi|gdi32|rpcrt4|combase|sechost|msvcrt|ucrtbase|vcruntime\d+(?:_\d+)?|msvcp\d+|concrt\d+|api-ms-win-.+)\.dll$/i;

function bundleWindows(binary, directory) {
    const dumpbin = findDumpbin();
    const dirs = windowsSearchDirs();
    const pending = [binary];
    const seen = new Set();
    while (pending.length) {
        const file = pending.pop();
        const key = basename(file).toLowerCase();
        if (seen.has(key)) continue;
        seen.add(key);
        const output = run(dumpbin, ['/nologo', '/dependents', file]);
        for (const name of windowsDependents(output)) {
            if (WINDOWS_SYSTEM.test(name) || seen.has(name.toLowerCase())) continue;
            const found = findFile(name, dirs);
            if (!found) throw new Error(`${name} is required by ${basename(file)} and was not found beside LLVM or the CUDA toolkit`);
            pending.push(copyRuntime(found, directory, name));
        }
    }
}

function findDumpbin() {
    const vswhere = join(process.env['ProgramFiles(x86)'] ?? 'C:\\Program Files (x86)', 'Microsoft Visual Studio', 'Installer', 'vswhere.exe');
    if (!existsSync(vswhere)) throw new Error('vswhere.exe was not found');
    const found = run(vswhere, ['-latest', '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-find', 'VC\\Tools\\MSVC\\**\\bin\\Hostx64\\x64\\dumpbin.exe'])
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter(Boolean);
    if (!found.length) throw new Error('dumpbin.exe was not found');
    return found[found.length - 1];
}

function windowsSearchDirs() {
    const dirs = new Set((process.env.PATH ?? '').split(delimiter).filter(Boolean));
    if (process.env.CUDA_PATH) dirs.add(join(process.env.CUDA_PATH, 'bin'));
    const programFiles = process.env.ProgramFiles ?? 'C:\\Program Files';
    dirs.add(join(programFiles, 'LLVM', 'bin'));
    const toolkit = join(programFiles, 'NVIDIA GPU Computing Toolkit', 'CUDA');
    if (existsSync(toolkit)) {
        for (const version of readdirSync(toolkit)) dirs.add(join(toolkit, version, 'bin'));
    }
    return [...dirs];
}

function windowsDependents(output) {
    return [...output.matchAll(/^\s+([\w.+-]+\.dll)\s*$/gim)].map((match) => match[1]);
}

function findFile(name, dirs) {
    for (const dir of dirs) {
        const candidate = join(dir, name);
        if (existsSync(candidate)) return candidate;
    }
    return null;
}

const LINUX_VENDOR = /^(libcudart|libcublas|libcublasLt|libnvJitLink|libnvrtc|libnvToolsExt)\.so/;
const LINUX_SYSTEM = /^(libc|libm|libdl|librt|libpthread|libgcc_s|libstdc\+\+|ld-linux|libgomp|linux-vdso)\.so/;

function bundleLinux(binary, directory, shipCuda) {
    const search = shipCuda ? cudaLibDirs() : [];
    for (const entry of ldd(binary, search.join(':'))) {
        if (entry.loader) continue;
        const vendor = LINUX_VENDOR.test(entry.name);
        if (!vendor && LINUX_SYSTEM.test(entry.name) && entry.path && isSystemLinux(entry.path)) continue;
        const source = entry.path && existsSync(entry.path) ? entry.path : findFile(entry.name, search);
        if (!source) throw new Error(`${entry.name} resolves to ${entry.path ?? 'not found'} and was not bundled`);
        if (!vendor && isSystemLinux(source)) continue;
        copyRuntime(source, directory, entry.name);
    }
    assertLinux(join(directory, basename(binary)), directory);
}

function ldd(binary, extraPath) {
    const env = { ...process.env };
    if (extraPath) env.LD_LIBRARY_PATH = [extraPath, env.LD_LIBRARY_PATH].filter(Boolean).join(':');
    return parseLdd(run('ldd', [binary], env));
}

function parseLdd(text) {
    const entries = [];
    for (const line of text.split('\n')) {
        const missing = line.match(/^\s*(\S+)\s+=>\s+not found/);
        if (missing) {
            entries.push({ name: missing[1], path: null, loader: false });
            continue;
        }
        const found = line.match(/^\s*(\S+)\s+=>\s+(\S+)/);
        if (found) {
            entries.push({ name: found[1], path: found[2], loader: false });
            continue;
        }
        const loader = line.match(/^\s*(\/\S*ld-linux\S*)/);
        if (loader) entries.push({ name: basename(loader[1]), path: loader[1], loader: true });
    }
    return entries;
}

function isSystemLinux(path) {
    return path.startsWith('/lib/') || path.startsWith('/lib64/') || path.startsWith('/usr/lib/') || path.startsWith('/usr/lib64/');
}

function cudaLibDirs() {
    const roots = [process.env.CUDA_HOME, process.env.CUDA_PATH, '/usr/local/cuda'].filter(Boolean);
    const dirs = [];
    for (const root of roots) {
        dirs.push(join(root, 'lib64'), join(root, 'lib'), join(root, 'targets', 'x86_64-linux', 'lib'));
    }
    return dirs.filter((dir) => existsSync(dir));
}

function assertLinux(binary, directory) {
    const env = { ...process.env, LD_LIBRARY_PATH: directory };
    for (const entry of parseLdd(run('ldd', [binary], env))) {
        if (entry.loader) continue;
        if (!entry.path) throw new Error(`${entry.name} was not found next to ${basename(binary)}`);
        if (entry.path.startsWith(directory)) continue;
        if (isSystemLinux(entry.path) && !LINUX_VENDOR.test(entry.name)) continue;
        throw new Error(`${entry.name} resolves to ${entry.path}. Bundle it beside the addon.`);
    }
}

function bundleCross(binary, triple) {
    const text = run(`${triple}-readelf`, ['-d', binary]);
    for (const match of text.matchAll(/\((?:RPATH|RUNPATH)\).*\[([^\]]*)\]/g)) {
        if (match[1] !== '$ORIGIN') throw new Error(`${triple} ${match[0]} should be $ORIGIN`);
    }
    for (const match of text.matchAll(/\(NEEDED\).*\[([^\]]*)\]/g)) {
        const name = match[1];
        if (LINUX_SYSTEM.test(name)) continue;
        throw new Error(`Cross build depends on ${name}, which is not a system library`);
    }
}

function bundleMac(binary, directory) {
    run('install_name_tool', ['-id', '@loader_path/tsgrad_native.node', binary]);
    const pending = [binary];
    const seen = new Set();
    while (pending.length) {
        const file = pending.pop();
        if (seen.has(file)) continue;
        seen.add(file);
        const deps = parseOtool(run('otool', ['-L', file])).slice(1);
        for (const dep of deps) {
            if (dep.startsWith('@loader_path/')) {
                const sibling = join(directory, dep.slice('@loader_path/'.length));
                if (!existsSync(sibling)) throw new Error(`${basename(file)} loads ${dep}, which is missing from ${directory}`);
                continue;
            }
            if (isMacSystem(dep)) continue;
            const source = dep.startsWith('@rpath/') ? findMacLib(dep.slice('@rpath/'.length)) : dep;
            const name = basename(source);
            const target = join(directory, name);
            if (!existsSync(target)) copyRuntime(source, directory, name);
            run('install_name_tool', ['-id', `@loader_path/${name}`, target]);
            run('install_name_tool', ['-change', dep, `@loader_path/${name}`, file]);
            pending.push(target);
        }
    }
    for (const file of seen) run('codesign', ['--force', '--sign', '-', file]);
}

function parseOtool(text) {
    return [...text.matchAll(/^\s+(\S+)\s+\(compatibility/gm)].map((match) => match[1]);
}

function isMacSystem(path) {
    return path.startsWith('/usr/lib/') || path.startsWith('/System/');
}

function findMacLib(name) {
    const roots = [];
    if (process.env.OpenMP_ROOT) roots.push(join(process.env.OpenMP_ROOT, 'lib'));
    try {
        roots.push(join(run('brew', ['--prefix', 'libomp']).trim(), 'lib'));
    } catch {
        // Homebrew is absent when OpenMP_ROOT already points at the library.
    }
    const found = findFile(name, roots);
    if (!found) throw new Error(`Could not find ${name}. Set OpenMP_ROOT to the libomp prefix.`);
    return found;
}
