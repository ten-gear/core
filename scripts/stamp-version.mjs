import { readFileSync, writeFileSync } from 'node:fs';

const file = process.argv[2];
const version = process.argv[3];
if (!file || !version) {
    console.error('usage: node scripts/stamp-version.mjs <package.json> <version>');
    process.exit(1);
}

const pkg = JSON.parse(readFileSync(file, 'utf8'));
pkg.version = version;
for (const dep of Object.keys(pkg.optionalDependencies ?? {})) {
    if (dep.startsWith('tengear-')) pkg.optionalDependencies[dep] = version;
}
writeFileSync(file, JSON.stringify(pkg, null, 2) + '\n');
