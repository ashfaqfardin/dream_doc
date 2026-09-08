# FinalTex consistency review

## Review scope

This review covers every TeX fragment and supplied figure under `FinalTex`. It checks document structure, paragraph style, cross references, terminology, experimental claims, metric comparability, figure usage, and agreement with the available implementations and result artifacts.

The findings below are ordered by priority. Each finding is followed immediately by a recommended fix.

## Critical issues

### 1. Full compilation cannot be validated from FinalTex alone

The folder contains chapter fragments and images, while the main TeX file, bibliography database, and document class configuration are maintained elsewhere. This is not a manuscript inconsistency by itself, but it limits this review: citation resolution, chapter inclusion order, package availability, and full compilation could not be verified from `FinalTex` alone.

Suggested fix: no relocation is required. Add a short README identifying the authoritative main file and bibliography, then run the final reference and compilation checks from the complete document root.

### 2. References point to an appendix that is absent from FinalTex

The appendix was intentionally removed from the dissertation. The two surviving references are therefore stale:

* `discussions.tex`, line 45: `app:project-experiment-failures`
* `technical2.tex`, line 42: `app:project-experiment-failures`

Suggested fix: remove both references and summarize the corresponding failure evidence in the discussion or implementation chapter. Do not restore the appendix unless its inclusion becomes an explicit manuscript decision.

### 3. The hard composition equation references do not match the methodology label
Anwer: 

The following files reference `eq:hard-composition`, but no such label exists inside FinalTex:

* `discussions.tex`, line 8
* `results.tex`, line 32
* `technical2.tex`, line 42

The corresponding equation in `technical.tex` is labelled `eq:final_compositing` at line 237.

Suggested fix: choose one canonical label. The clearest option is to rename the label in `technical.tex` to `eq:hard-composition`, then retain all three existing references. Alternatively, update all three references to `eq:final_compositing`.

### 4. The pipeline figure reference is unresolved

`technical2.tex`, line 8 refers to `fig:pipeline`, but the full methodology chapter defines `fig:methodology-overview` at line 80 of `technical.tex`.

Suggested fix: change the reference in `technical2.tex` to `fig:methodology-overview`, or give the figure the canonical label `fig:pipeline` and update every reference consistently.

### 5. Two empty chapter references occur in the literature review

`theory.tex`, lines 46 and 56 each contain two empty references in the form `\ref{}`. These will compile as unresolved references and leave the reader without chapter destinations.

Suggested fix: replace them with explicit references to the implementation and results chapters, most likely `chapter:implementation_two_backbone` and `chapter:results`.

### 6. The opening sentence of the results chapter is incomplete

`results.tex`, line 4 says that evaluation is performed “under the of Chapter”. A noun phrase is missing.

Suggested fix: write “under the shared local collage harmonization framework described in Chapter~\ref{chapter:implementation_two_backbone}.”

### 7. Reference image provenance conflicts with the implementation

`technical.tex`, lines 86 to 99 state that object references were generated from Canny edge maps and that source photographs were not used directly. The implementation in `e3_prompt_suite.py` instead prioritizes images found in `ExperimentQwen/references`; Canny based generation is only a fallback when a provided reference is unavailable. The text also states that either FLUX.1 Kontext or Qwen generated the reference, whereas the fallback code uses the Qwen pipeline inherited from `e1_baseline.py`.

Suggested fix: inspect the reference manifest used by the reported final runs. If all references have source `provided_reference`, describe them as supplied object photographs and treat the Canny generation procedure as a separate preparation experiment. If some references have source `generated_from_canny`, report the exact count, generator, seed 1337, and source for every object. Do not claim a uniformly Canny generated dataset unless the artifacts prove it.

### 8. The dissertation claims four stages and three stages without defining their levels

`introduction.tex`, line 37 describes a four stage pipeline consisting of base generation, object generation, compositing, and drift mitigation. `technical.tex`, lines 54 to 61 describes each insertion as three stages: deterministic preparation, local harmonization, and hard preservation. The figure caption uses the three stage interpretation.

Suggested fix: distinguish the full experimental workflow from the per turn insertion operator. State that the complete workflow has four phases, while each insertion turn has three internal stages. Use “phase” for the full workflow and “stage” for the insertion operator.

### 9. Methodology and implementation chapters substantially duplicate one another

`technical.tex` already presents the complete method, equations, pipeline figure, and sequential procedure. `technical2.tex` repeats the task definition, backbone configuration, project variants, evaluation, and summary. It also depends on labels defined only in `technical.tex` and the absent appendix.

Suggested fix: keep `technical.tex` as the methodology chapter and convert `technical2.tex` into a focused implementation details chapter containing only checkpoint versions, inference parameters, hardware, software, reproducibility, and backbone specific differences. Remove repeated method summaries and equations.

### 10. The introduction maps chapters to the wrong content

`introduction.tex`, line 48 says that Chapter `chapter:implementation_two_backbone` defines evaluation methodology, presents qualitative and quantitative results, reports ablations, and analyses tradeoffs. In FinalTex, that chapter is an implementation chapter; the actual results are in `chapter:results`. Furthermore, no completed ablation section appears in the results chapter.

Suggested fix: describe `chapter:implementation_two_backbone` as implementation and reproducibility, then describe `chapter:results` as evaluation and results. Remove the ablation claim unless those experiments and tables are added.

### 11. A methodology chapter incorrectly refers to itself as a later chapter

`technical.tex`, lines 346 onward states that Chapter `chapter:methodology` describes how the methodology is instantiated. That label belongs to the current chapter.

Suggested fix: refer to `chapter:implementation_two_backbone` for concrete backbone instantiation, or rewrite the sentence to summarize what the current chapter has established.

### 12. The native baseline evidence is only partly compatible with the proposed method metrics

The strict baseline intentionally uses no placement mask. Therefore it cannot directly produce object support, placement compliance, exterior preservation, or localized prior object stability metrics without a separate localization procedure. `results.tex` appropriately uses unavailable entries for these fields, but claims that the method “strongly improves spatial control” over the native baseline at line 136 without a mask free quantitative placement measurement.

Suggested fix: phrase this as a design difference rather than a measured improvement. For example, state that the proposed method enforces an explicit user supplied placement constraint, whereas the native baseline offers no explicit placement interface. Keep the full image changed fraction comparison, which is genuinely mask free.

### 13. Baseline version provenance is unclear

The paper reports mask free statistics derived from `NativeTwoImageBaseline_01`, while the corrected strict implementation writes to `NativeTwoImageBaseline_02`. The first result folder also contains rectangle based diagnostic fields from an older evaluator even though its generated images were not mask conditioned.

Suggested fix: rerun the corrected baseline, cite only `NativeTwoImageBaseline_02`, and regenerate all aggregate tables from its `metrics_summary.json`. Archive or clearly mark the first folder as superseded. Do not mix its rectangle based fields into the strict baseline analysis.

### 14. The native baseline is called complete before the corrected run is documented

`discussions.tex`, line 73 states that the native baseline is complete. This is premature until the corrected mask free run and its exported metric summary are fixed as the authoritative result set.

Suggested fix: either rerun and document the corrected baseline before retaining this claim, or change the sentence to say that the native baseline protocol has been implemented and awaits final rerun verification.

## Cross reference and structure issues

### 15. No duplicate labels were found

This is currently consistent. Preserve uniqueness when the appendix and main file are added.

### 16. Several labels are currently unused

Unused labels are not compilation errors, but they indicate either incomplete navigation or unnecessary markup. The most important unused labels are:

* `tab:aggregate-results`
* `tab:depth-results`
* `tab:per-case`
* `fig:challenging-results`
* `tab:baseline-reserved`
* `tab:backbone-comparison`
* `sec:pipeline_overview`
* most equation labels in `technical.tex`
* `tab:gap-matrix`

Suggested fix: refer to every substantive table and figure in the surrounding prose. Retain equation labels only when an equation is referenced later; otherwise an unlabelled equation is sufficient.

### 17. Equation reference formatting is inconsistent

`technical.tex`, lines 240 and 243 use `Equation~(\ref{...})`, while other chapters use plain `\ref`. Manual parentheses are fragile.

Suggested fix: use `Equation~\eqref{...}` everywhere for equations, `Figure~\ref{...}` for figures, `Table~\ref{...}` for tables, and `Chapter~\ref{...}` for chapters.

### 18. Section reference formatting is inconsistent

Some references use a nonbreaking space, such as `Section~\ref`, while others use `Section \ref`. The same inconsistency occurs for chapters, tables, and figures.

Suggested fix: standardize all named references with a nonbreaking space between the name and reference command.

### 19. The main pipeline figure appears only in technical.tex

`technical2.tex` expects a pipeline figure but contains no figure environment of its own. This dependency is easy to break if the chapter order changes.

Suggested fix: keep the figure and its discussion in exactly one chapter. The implementation chapter should refer back to the methodology figure using its correct canonical label.

## Paragraph indentation audit

The project style requires `\noindent` at the start of every prose paragraph. LaTeX normally suppresses indentation automatically after headings, but that does not satisfy the stated source style rule. The following probable prose paragraphs do not start with `\noindent`.

### abstract.tex

Line 3.

### conclusions.tex

Lines 4, 8, 18, and 24.

### discussions.tex

Lines 4, 8, 10, 12, 14, 20, 22, 26, 28, 32, 34, 38, 40, 45, 55, 59, 61, 63, 67, 69, 73, 87, and 91.

### introduction.tex

Lines 4, 10, and 44.

### results.tex

Lines 4, 8, 32, 49, 53, 69, 99, 132, and 136.

### technical.tex

Lines 4, 12, 41, 104, 170, 173, 185, 200, 206, 219, 229, 240, 252, 259, 267, 282, 290, 297, 306, 318, 328, and 336.

### technical2.tex

Lines 4, 8, 12, 14, 40, 42, 46, 48, and 52.

### theory.tex

Lines 4, 14, 22, 46, 63, 75, 80, 85, 94, 99, 104, 110, 118, 128, 142, 157, 176, 179, 185, 188, 191, 194, 197, 200, 203, 206, 237, 242, 247, 252, 257, and 279.

The entries in the table areas of `theory.tex`, particularly lines 176 onward and 237 onward, should be inspected manually before adding `\noindent`, because some detected lines may be wrapped table cells rather than independent prose paragraphs.

### wordcount.tex

Line 4. This may be a form field rather than prose, so `\noindent` is optional unless the rule is literal.

Suggested fix: add `\noindent` to genuine prose paragraphs in the lists above. Do not add it to table rows, captions, displayed equations, list items, or continuation lines. Then rerun the audit manually because a blank line inside a deliberately wrapped paragraph can create a false paragraph boundary.

## Typography and language consistency

### 20. Prohibited consecutive hyphen sequences remain in the TeX sources

They occur in `introduction.tex`, `results.tex`, `technical2.tex`, and `theory.tex`. Some represent TeX dash syntax, some represent unavailable table entries, and some occur only in editorial comments.

Suggested fix: use `\textemdash{}` for an em dash, `\textendash{}` for ranges, and `N/A` or `Not measured` for unavailable table values. Remove editorial comments before submission.

### 21. Several single hyphens are incorrectly used as punctuation

Examples include “variables pose” and “scene depth are” at `discussions.tex`, line 14, where the source currently joins clauses with raw hyphens. Similar compounds include “identity naturalness gap” and “quality runtime curves”.

Suggested fix: use commas or `\textemdash{}` for sentence punctuation, and use consistent compounds such as “identity–naturalness gap” and “quality–runtime curves”.

### 22. British and American spelling are mixed

The dissertation mostly uses British forms such as “colour”, “harmonisation”, “localised”, and “artefact”, but several files use “harmonization”, “localized”, “color”, and “artifacts”. The project title currently uses “Harmonization”.

Suggested fix: choose one convention. For a UK dissertation, standardize prose to “harmonisation”, “localised”, “colour”, “normalisation”, and “artefact”. Preserve official model names, code identifiers, filenames, and quoted titles unchanged.

### 23. Multi step, multi turn, and sequential terminology is inconsistent

The files alternate among “multi-step”, “multi-turn”, “incremental”, and “sequential” without defining whether they are synonyms.

Suggested fix: define “sequential editing” as the task, “turn” as one edit, and “multi-step pipeline” only for internal processing stages. Use those terms consistently.

### 24. Model naming is inconsistent

The text alternates among “Qwen-Image-Edit”, “Qwen-Image-Edit-2509”, “Qwen”, “FLUX.1 Kontext Dev”, “Kontext”, and “Kontext-dev”.

Suggested fix: introduce the full checkpoint names once, then define “Qwen” and “Kontext” as abbreviations. Use the exact checkpoint spelling in implementation tables and the abbreviations elsewhere.

### 25. Citation commands are unnecessarily separated

`introduction.tex`, line 4 uses several adjacent citation commands. Similar patterns appear throughout the literature review.

Suggested fix: combine adjacent citations into a single command, for example `\cite{key1,key2,key3}`, unless the chosen bibliography style requires another command.

### 26. Capitalization error in the introduction

`introduction.tex`, line 4 begins a mid sentence clause with “However, Repeated edits”.

Suggested fix: change “Repeated” to “repeated”.

### 27. Subject agreement and chapter linking are incorrect

`technical.tex`, line 6 says “Chapter A and Chapter B describes”. It also uses an escaped ampersand in prose.

Suggested fix: write “Chapters A and B describe” and use the word “and”.

### 28. The literature review opening is grammatically awkward

`theory.tex`, line 4 includes “Sections ... to ..., then review” and “a gap matrix Section”.

Suggested fix: remove the comma before “then review” and write “a gap matrix in Section~\ref{sec:comp_analysis_gap}”.

### 29. The results figure caption overstates measured placement failure for the strict baseline

`results.tex`, line 75 says native editing does not respect withheld placement regions. The strict baseline is intentionally given no placement constraint, and the final mask free evaluation does not measure compliance with those regions.

Suggested fix: say that native editing provides no explicit placement control. Reserve claims about compliance for a separately annotated qualitative assessment or detector based evaluation.

### 30. The word count file is incomplete

`wordcount.tex` contains empty word and page counts.

Suggested fix: populate both values from the final compiled document immediately before submission, using the institution’s required counting convention.

## Metric and reporting issues

### 31. Deterministic metrics and empirical model metrics are mixed in one ranking table

Object support containment and zero exterior change are guaranteed by deterministic placement and hard composition. DINO, colour, and edge scores measure empirical outputs. Boldface ranking makes deterministic guarantees look like backbone wins.

Suggested fix: separate the aggregate table into “model dependent fidelity” and “pipeline enforced constraints”. Do not bold deterministic ties as performance victories.

### 32. Changed image fraction has no universal preferred direction

The aggregate table marks lower changed fraction as better. A value of zero could also indicate failure to insert anything.

Suggested fix: remove the direction arrow for this metric and interpret it jointly with successful insertion and localisation.

### 33. Exact zeros need a numerical precision statement

The manuscript states zero pixel change outside the mask. Readers need to know whether this means exact integer equality after saving, equality before encoding, or a rounded aggregate.

Suggested fix: state that final RGB pixels outside the binary hard support are copied directly from the preceding image and verified after image construction using maximum absolute error equal to zero. If verification occurred after saving and reloading, state that explicitly.

### 34. The DINO metric description risks overstating instance identity

DINO similarity can remain high for category, shape, or unchanged pasted texture without establishing exact instance identity. The discussion acknowledges this, but the aggregate table calls it simply “DINO identity”.

Suggested fix: rename it “localised DINO similarity” throughout and reserve “identity fidelity” for the combined interpretation of DINO, colour, edge, and human judgment.

### 35. Full image baseline MAE is not yet tied to an authoritative exported summary

The value 10.1894 appears in results, discussion, and conclusion, but the corrected baseline has not yet been identified as the authoritative exported result folder in FinalTex.

Suggested fix: rerun the strict code, ensure `metrics_summary.json` contains this value, and cite the exact result directory in the appendix. If the rerun differs, update every occurrence from one generated table rather than manual transcription.

### 36. Statistical uncertainty is absent

The evaluation uses one seed and one fixed object order. Means over 30 turns are not independent replicates because turns are nested within scenes and sequences.

Suggested fix: present the current analysis as descriptive. For inferential claims, run multiple seeds and preferably multiple insertion orders, then report scene clustered or hierarchical confidence intervals rather than treating 30 turns as independent samples.

### 37. The Qwen and Kontext comparison is not compute matched

Qwen uses eight steps with a Lightning adapter, while Kontext uses 28 steps. The discussion acknowledges this, but phrases such as “Kontext improves” can still be read as architectural superiority.

Suggested fix: consistently say “the evaluated Kontext configuration obtains higher scores”. Add either a matched runtime comparison or a quality against compute curve before making architecture level claims.

### 38. No human plausibility results are available

The most important unresolved quality is natural placement, yet the corresponding table entries remain “To collect”. Automatic metrics do not measure contact, support, perspective, or realism.

Suggested fix: complete a blinded rating study with separate questions for reference resemblance, placement and physical contact, and overall realism. Report participant count, randomisation, rating scale, agreement, and confidence intervals.

## Figure issues

### 39. The all cases figure source must be synchronized with its caption

The caption states that four rows are shown, including the native baseline. Confirm that the copied PDF in `FinalTex/images/MyWork/all_cases_comparison.pdf` is the regenerated four row version rather than the older three row figure.

Suggested fix: rebuild the figure from the authoritative result folders, copy the generated PDF into FinalTex, and visually verify all row labels before submission.

### 40. The failure figures are present but unused

`developmental_failures.pdf` and `project_experiment_failures.pdf` exist in `images/MyWork`, but no FinalTex chapter includes either one. This leaves the E1 to E15 failure discussion without visual evidence.

Suggested fix: include only `project_experiment_failures.pdf` in the appendix or discussion, with a precise caption mapping each panel to its implementation. Remove the older developmental figure if it is superseded.

### 41. Figure accessibility and interpretation are incomplete

The qualitative figures lack textual descriptions of what a reader should inspect beyond the captions. The all cases montage is dense at dissertation page width.

Suggested fix: add explicit callouts in prose for two or three representative cases and move the full montage to a landscape page or appendix. Ensure labels remain legible in the final printed PDF.

## Recommended correction order

1. Add the main file, bibliography, and appendix to FinalTex.
2. Resolve every missing and empty reference.
3. Establish the true provenance of all reference object images.
4. Rerun and freeze the corrected strict native baseline result folder.
5. Consolidate `technical.tex` and `technical2.tex` into methodology and implementation chapters with no duplication.
6. Correct chapter descriptions in the introduction and methodology summary.
7. Regenerate tables directly from authoritative JSON files.
8. Rebuild and verify every figure against its caption.
9. Add `\noindent` to genuine prose paragraphs according to the line audit.
10. Standardize spelling, terminology, model names, reference formatting, and dash typography.
11. Complete the human evaluation or clearly limit all placement conclusions to qualitative observations.
12. Remove editorial comments and populate the final word and page counts.

## Overall assessment

The central technical narrative is defensible: explicit pixel space initialization supplies reference evidence, local generation limits model exposure, and hard RGB composition provides a genuine non disturbance guarantee. The strongest current weakness is not the core idea but documentary consistency. The reference provenance, missing cross references, duplicated methodology, baseline versioning, and absence of quantitative physical plausibility evaluation must be resolved before the manuscript can support its strongest claims.
