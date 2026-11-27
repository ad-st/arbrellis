const { test, expect } = require('@playwright/test');

async function openWorkspace(page, modelSetupStatus = null) {
  await page.route('**/api/model-setup', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify(modelSetupStatus || {
      profile: 'standard', ready: true, standard_ready: true, enhanced_ready: false,
      models: {
        qwen: { cached: true }, bge: { cached: true }, t5gemma: { cached: false },
      },
    }),
  }));
  await page.goto('/');
  await page.waitForFunction(() => window.__ARBRELLIS_READY__ === true);
}

test('loads the fixture graph and entity totals', async ({ page }) => {
  await openWorkspace(page);
  await expect(page.locator('#nodes .node')).toHaveCount(3);
  await expect(page.locator('[data-entity="facts"] [data-entity-count]')).toHaveText('2');
  await expect(page.locator('[data-entity="docs"] [data-entity-count]')).toHaveText('1');
  await expect(page.locator('[data-entity="chats"] [data-entity-count]')).toHaveText('0');
});

test('opens, resizes, and closes the facts drawer', async ({ page }) => {
  await openWorkspace(page);
  const facts = page.locator('[data-entity="facts"]');
  await facts.click();
  await expect(page.locator('#entity-panel')).toHaveAttribute('data-width-tier', '1');
  await expect(page.locator('.fact-list-button')).toHaveCount(2);
  await facts.click();
  await expect(page.locator('#entity-panel')).toHaveAttribute('data-width-tier', '2');
  await page.locator('#entity-panel').click({ button: 'right' });
  await expect(page.locator('#entity-panel')).toBeHidden();
});

test('opens a document with extracted passages', async ({ page }) => {
  await openWorkspace(page);
  await page.locator('[data-entity="docs"]').click();
  await page.getByRole('button', { name: /transformer-notes\.md/ }).click();
  await expect(page.locator('.document-detail-title')).toHaveText('transformer-notes.md');
  await expect(page.locator('.document-preview')).toContainText('Causal attention');
  await expect(page.locator('.document-fact-extract')).toHaveCount(2);
});

test('opens a single-fact graph node in the fact editor', async ({ page }) => {
  await openWorkspace(page);
  await page.locator('#nodes .node').filter({ hasText: 'residual streams' }).click();
  await expect(page.locator('.fact-detail-label')).toHaveText('residual streams');
  await expect(page.locator('.fact-editor')).toHaveValue(/preserve information/);
});

test('opens and cancels the add-fact dialog without invoking models', async ({ page }) => {
  await openWorkspace(page);
  await page.locator('#graph').dblclick({ position: { x: 60, y: 420 } });
  await expect(page.locator('#fact-dialog')).toBeVisible();
  await page.locator('#fact-text').fill('A disposable UI test fact.');
  await page.keyboard.press('Escape');
  await expect(page.locator('#fact-dialog')).toBeHidden();
});

test('explains model downloads before the first integration and preserves the fact', async ({ page }) => {
  await openWorkspace(page, {
    profile: 'standard', ready: false, standard_ready: false, enhanced_ready: false,
    models: {
      qwen: { cached: false }, bge: { cached: false }, t5gemma: { cached: false },
    },
  });
  await expect(page.locator('#model-setup-dialog')).toBeVisible();
  await page.locator('#model-setup-cancel').click();
  await page.locator('#graph').dblclick({ position: { x: 60, y: 420 } });
  await page.locator('#fact-text').fill('KV caching reuses earlier attention states.');
  await page.locator('#fact-submit').click();

  await expect(page.locator('#model-setup-dialog')).toBeVisible();
  await expect(page.locator('#model-setup-dialog')).toContainText('About 3 GB');
  await page.locator('#model-setup-cancel').click();
  await expect(page.locator('#fact-dialog')).toBeVisible();
  await expect(page.locator('#fact-text')).toHaveValue('KV caching reuses earlier attention states.');
});

test('keeps empty graph gestures available beneath the empty-state message', async ({ page }) => {
  let initialGraph = true;
  await page.route('**/api/graph', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify(initialGraph ? {
      nodes: [], edges: [], neighbours: 2, baseline: 0,
      cache: { mode: 'full', pairs: 0 },
    } : {
      nodes: [{
        id: 99, label: 'first concept', facts: [{ id: 99 }], fact_edges: [],
        fact_count: 1, density: 0, degree: 0, position: null,
      }],
      edges: [], neighbours: 2, baseline: 0, cache: { mode: 'full', pairs: 0 },
    }),
  }));
  await openWorkspace(page);
  await expect(page.locator('.empty')).toHaveText('No stored passages yet.');

  await page.locator('[data-entity="facts"]').click();
  await expect(page.locator('#entity-panel')).toBeVisible();
  await page.locator('#graph').click({ button: 'right', position: { x: 500, y: 400 } });
  await expect(page.locator('#entity-panel')).toBeHidden();

  await page.locator('#graph').dblclick({ position: { x: 500, y: 400 } });
  await expect(page.locator('#fact-dialog')).toBeVisible();
  await page.keyboard.press('Escape');

  initialGraph = false;
  await page.evaluate(() => refreshGraphFromServer());
  await expect(page.locator('.empty')).toHaveCount(0);
  await expect(page.locator('#nodes .node')).toHaveCount(1);
});
