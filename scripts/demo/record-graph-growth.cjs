const fs = require('node:fs/promises');
const path = require('node:path');
const { chromium } = require('playwright');

const configPath = path.resolve(process.argv[2] || path.join(__dirname, 'graph-growth.json'));
const outputDirectory = path.resolve(process.argv[3] || path.join(process.cwd(), 'demo-recording'));
const sleep = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));

async function installDemoPresentation(page, config) {
  if (config.solidBackground) {
    await page.addStyleTag({
      content: `html, body { background: ${config.solidBackground} !important; }`,
    });
  }
  if (!config.showClicks) return;
  await page.evaluate(clickColor => {
    const showClick = (x, y) => {
      const marker = document.createElement('span');
      marker.setAttribute('aria-hidden', 'true');
      marker.setAttribute('popover', 'manual');
      Object.assign(marker.style, {
        position: 'fixed',
        zIndex: '2147483647',
        left: `${x}px`,
        top: `${y}px`,
        width: '12px',
        height: '12px',
        margin: '0',
        padding: '0',
        background: 'transparent',
        border: `2px solid ${clickColor}`,
        borderRadius: '50%',
        pointerEvents: 'none',
        transform: 'translate(-50%, -50%) scale(.35)',
        boxShadow: `0 0 10px ${clickColor}`,
      });
      document.body.append(marker);
      marker.showPopover();
      marker.animate(
        [
          { opacity: 1, transform: 'translate(-50%, -50%) scale(.35)' },
          { opacity: 0, transform: 'translate(-50%, -50%) scale(2.4)' },
        ],
        { duration: 520, easing: 'ease-out' },
      ).finished.finally(() => {
        marker.hidePopover();
        marker.remove();
      });
    };
    window.__arbrellisDemoShowClick = showClick;
    document.addEventListener('pointerdown', event => {
      if (event.button !== 0) return;
      showClick(event.clientX, event.clientY);
    }, true);
    // A modal dialog enters the top layer after the two pointer events. Add a
    // final pulse after it opens so the double-click remains visible.
    document.addEventListener('dblclick', event => {
      if (!event.isTrusted) return;
      setTimeout(() => showClick(event.clientX, event.clientY), 0);
    }, true);
  }, config.clickColor || '#f1bd74');
}

async function main() {
  const config = JSON.parse(await fs.readFile(configPath, 'utf8'));
  await fs.mkdir(outputDirectory, { recursive: true });
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: config.viewport,
    deviceScaleFactor: 1,
    recordVideo: { dir: outputDirectory, size: config.viewport },
  });
  const pageCreatedAt = Date.now();
  const page = await context.newPage();
  const video = page.video();
  await page.goto(config.url, { waitUntil: 'networkidle' });
  await page.waitForFunction(() => document.querySelectorAll('#nodes > g').length === 4);
  await installDemoPresentation(page, config);
  await page.locator('#reset').click();
  await sleep(1200);

  const sequenceStartedAt = Date.now();
  const timeline = {
    sequenceOffsetMs: sequenceStartedAt - pageCreatedAt,
    submittedMs: null,
    afterIntegratePauseMs: null,
    modelStartedMs: {},
    nodeAppearedMs: null,
    endedMs: null,
  };
  await sleep(config.initialWaitMs);
  await page.evaluate(() => window.__arbrellisDemoShowClick(110, 230));
  await sleep(140);
  await page.evaluate(() => window.__arbrellisDemoShowClick(110, 230));
  await sleep(Math.max(0, config.popupDelayMs - 140));
  await page.locator('#graph').dispatchEvent('dblclick', {
    button: 0,
    clientX: 110,
    clientY: 230,
  });
  const input = page.locator('#fact-text');
  await input.waitFor({ state: 'visible' });
  await input.focus();
  const words = config.passage.split(' ');
  for (let index = 0; index < words.length; index += 1) {
    const suffix = index === words.length - 1 ? '' : ' ';
    await input.pressSequentially(words[index] + suffix, { delay: config.characterDelayMs });
    await sleep(config.wordPauseMs);
  }
  await sleep(config.afterTypingPauseMs);

  const modelStart = model => page
    .locator(`.model-box[data-model="${model}"]`)
    .evaluate(element => new Promise(resolve => {
      if (element.classList.contains('working')) return resolve();
      const observer = new MutationObserver(() => {
        if (!element.classList.contains('working')) return;
        observer.disconnect();
        resolve();
      });
      observer.observe(element, { attributes: true, attributeFilter: ['class'] });
    }))
    .then(() => { timeline.modelStartedMs[model] = Date.now() - sequenceStartedAt; });
  const modelStarts = ['t5gemma', 'qwen', 'bge'].map(modelStart);
  await page.locator('#fact-submit').click();
  timeline.submittedMs = Date.now() - sequenceStartedAt;
  await sleep(config.afterIntegratePauseMs);
  timeline.afterIntegratePauseMs = Date.now() - sequenceStartedAt;
  await page.waitForFunction(() => document.querySelectorAll('#nodes > g').length === 5, null, { timeout: 300000 });
  await Promise.all(modelStarts);
  await sleep(config.newNodeSettleMs);
  await page.evaluate(offsetY => {
    const node = graph.nodes.find(item => item.label === 'pagedattention');
    if (!node) throw new Error('The PagedAttention node did not appear.');
    node.y = Math.min(innerHeight - 50, node.y + offsetY);
    node.fixed = true;
    drawLayout();
    persistPositions([node]);
  }, config.newNodeOffsetY);
  timeline.nodeAppearedMs = Date.now() - sequenceStartedAt;
  await sleep(config.graphAnimationMs);
  await page.locator('#graph').click({ position: { x: 110, y: 230 } });
  await sleep(config.finalWaitMs);
  timeline.endedMs = Date.now() - sequenceStartedAt;

  await page.close();
  await video.saveAs(path.join(outputDirectory, 'graph-growth-raw.webm'));
  await fs.writeFile(
    path.join(outputDirectory, 'graph-growth-timeline.json'),
    `${JSON.stringify({ config, timeline }, null, 2)}\n`,
  );
  await context.close();
  await browser.close();
  console.log(JSON.stringify(timeline));
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
