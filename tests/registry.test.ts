import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import test from "node:test";
import { countLabel, families, featuredModels, getModel, isLiveMode, labPages, labPath, liveLabs, models, modelsInFamily, pad2, plannedLabs } from "../src/content/models";

test("model slugs are unique and links resolve", () => {
  assert.equal(new Set(models.map((model)=>model.slug)).size,models.length);
  for (const model of models) {
    assert.ok(families.includes(model.family));
    assert.ok(model.paper.url.startsWith("https://"));
    assert.ok(model.architecture.length >= 3);
    for (const slug of model.related) assert.ok(getModel(slug),`${model.slug} links to missing ${slug}`);
  }
});

test("live labs have a registered execution mode", () => {
  assert.equal(getModel("linear-regression")?.executionMode,"live-training");
  assert.equal(getModel("cnn")?.executionMode,"live-training");
  assert.deepEqual(labPages.map((lab)=>lab.href),["/lab/linear-regression","/lab/mlp","/lab/cnn","/lab/attention","/lab/regression"]);
  assert.deepEqual(labPages.find((lab)=>lab.href==="/lab/regression")!.models.map((m)=>m.slug),["adaboost","gradient-boosting","random-forest","catboost","rbf-network"]);
  for (const model of liveLabs) if (labPath(model.lab!.href) === "/lab/regression") assert.equal(model.lab!.href, `/lab/regression?model=${model.slug}`);
});

test("execution mode and lab metadata agree", () => {
  for (const model of models) {
    if (isLiveMode(model.executionMode) || model.executionMode === "trace") {
      assert.ok(model.lab, `${model.slug} is ${model.executionMode} but has no lab entry`);
    } else {
      assert.equal(model.lab, undefined, `${model.slug} is planned but advertises a lab`);
    }
    assert.ok(!(model.lab && model.labPlan), `${model.slug} has both a shipped lab and a roadmap entry`);
  }
});

test("every shipped lab has a route", () => {
  for (const model of liveLabs) {
    const route = `src/app${labPath(model.lab!.href)}/page.tsx`;
    assert.ok(existsSync(route), `${model.slug} links to ${model.lab!.href}, but ${route} does not exist`);
  }
});

test("home page collections derive from the registry", () => {
  assert.equal(featuredModels.length, 3, "the featured grid is laid out for three cards");
  assert.equal(families.reduce((sum, family)=>sum + modelsInFamily(family).length, 0), models.length);
  for (const family of families) assert.ok(modelsInFamily(family).length > 0, `${family} has no models`);
  assert.ok(plannedLabs.every((model)=>model.executionMode === "planned"));
});

test("count formatting", () => {
  assert.equal(pad2(8), "08");
  assert.equal(pad2(12), "12");
  assert.equal(countLabel(1, "model"), "01 model");
  assert.equal(countLabel(3, "model"), "03 models");
});

test("every formula is valid KaTeX", async () => {
  const { renderLatex } = await import("../src/lib/math");
  const { cnnLessons } = await import("../src/content/cnn-theory");
  const { mlpPhaseText } = await import("../src/content/mlp-phases");
  const { attentionPhaseText } = await import("../src/content/attention-phases");
  const formulas = [...models.map((model) => [model.slug, model.latex]), ...cnnLessons.map((lesson) => [lesson.title, lesson.latex]), ...Object.entries(mlpPhaseText).map(([id, phase]) => [id, phase.latex]), ...Object.entries(attentionPhaseText).map(([id, phase]) => [id, phase.latex])];
  assert.equal(formulas.length, models.length + 12 + 12 + 5);
  for (const [name, latex] of formulas) {
    assert.doesNotThrow(() => renderLatex(latex), `${name}: ${latex}`);
    assert.match(renderLatex(latex), /<math/, `${name} has a MathML rendering for screen readers`);
  }
});
