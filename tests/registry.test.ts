import assert from "node:assert/strict";
import test from "node:test";
import { families, getModel, models } from "../src/content/models";

test("model slugs are unique and links resolve", () => {
  assert.equal(new Set(models.map((model)=>model.slug)).size,models.length);
  for (const model of models) {
    assert.ok(families.includes(model.family));
    assert.ok(model.paper.url.startsWith("https://"));
    assert.ok(model.architecture.length >= 3);
    for (const slug of model.related) assert.ok(getModel(slug),`${model.slug} links to missing ${slug}`);
  }
});

test("the first live lab has a registered execution mode", () => {
  assert.equal(getModel("linear-regression")?.executionMode,"live-training");
  assert.equal(models.filter((model)=>model.executionMode==="live-training").length,1);
});
