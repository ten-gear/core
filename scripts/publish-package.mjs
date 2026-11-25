import { execFileSync } from 'node:child_process';

const directory = process.argv[2];
if (!directory) {
    console.error('usage: node scripts/publish-package.mjs <package directory>');
    process.exit(1);
}

try {
    const output = execFileSync('npm', ['publish', '--provenance', '--access', 'public'], {
        cwd: directory,
        encoding: 'utf8',
        stdio: ['ignore', 'pipe', 'pipe'],
    });
    process.stdout.write(output);
} catch (error) {
    const output = `${error.stdout ?? ''}\n${error.stderr ?? ''}`;
    process.stdout.write(error.stdout ?? '');
    process.stderr.write(error.stderr ?? '');
    if (/EPUBLISHCONFLICT|previously published|cannot publish over/i.test(output)) {
        console.log(`Already published: ${directory}`);
        process.exit(0);
    }
    process.exit(error.status || 1);
}
