const { test, expect } = require('@playwright/test');

async function openWorkspace(page) {
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

test('keeps empty graph gestures available beneath the empty-state message', async ({ page }) => {
  await page.route('**/api/graph', route => route.fulfill({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify({
      nodes: [], edges: [], neighbours: 2, baseline: 0,
      cache: { mode: 'full', pairs: 0 },
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
});
