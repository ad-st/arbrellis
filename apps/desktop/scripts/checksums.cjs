const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');

const distDirectory = path.resolve(__dirname, '..', 'dist');
const artifacts = fs.readdirSync(distDirectory)
  .filter(name => /\.(?:AppImage|deb|tar\.gz)$/.test(name))
  .sort();

if (!artifacts.length) {
  console.error('No Linux release artifacts were found.');
  process.exit(1);
}

const lines = artifacts.map(name => {
  const digest = crypto.createHash('sha256')
    .update(fs.readFileSync(path.join(distDirectory, name)))
    .digest('hex');
  return `${digest}  ${name}`;
});
fs.writeFileSync(path.join(distDirectory, 'SHA256SUMS'), `${lines.join('\n')}\n`);
