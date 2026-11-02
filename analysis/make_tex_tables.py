#!/usr/bin/env python3
"""Generate paper-ready LaTeX tables from the audit artifacts.

Every table is followed by an \insight block and a \failuremode block, as required: numbers
alone are not a deliverable. American English throughout the prose. Code identifiers, variant
labels and bibliography titles are left exactly as they appear in the codebase.

Writes tables/rebuttal_tables.tex
"""
import csv, json, os, sys

REPO = "/nas/home/jahin/causal-lama"
V = os.path.join(REPO, "results", "verification")
N = os.path.join(REPO, "results", "permutation_null")
OUT = os.path.join(REPO, "tables", "rebuttal_tables.tex")

def rd(p):
    with open(p) as f: return list(csv.DictReader(f))

parts = []
parts.append(r"""% ============================================================================
% Rebuttal tables -- AAAI-27 submission 43976
% Generated from the analysis/, results/ and docs/ trees/. Every number traces to a CSV in
% results/verification/ or results/permutation_null/.
% Requires: booktabs. Define once in the preamble if not already present:
%   \newcommand{\insight}[1]{\par\smallskip\noindent\textit{Insight.} #1}
%   \newcommand{\failuremode}[1]{\par\smallskip\noindent\textit{Failure mode.} #1}
% ============================================================================
""")

# ---------------------------------------------------------------- Table N1: permutation null
rows = rd(os.path.join(N, "summary_corrected.csv"))
body = "\n".join(
    r"%s & %s & %s & %s$\times$ & %s & %s & %s & %s \\" % (
        r["model"], r["L"], r["obs_rcs_mean"], r["rcs_ratio_obsmean_over_null"],
        r["obs_gini"], r["null_gini_mean"], r["obs_n80_over_L"], r["null_n80_over_L_mean"])
    for r in rows)
parts.append(r"""
\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrrr}
\toprule
& & \multicolumn{2}{c}{RCS magnitude} & \multicolumn{2}{c}{Gini} & \multicolumn{2}{c}{$n_{80}/L$} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
Model & $L$ & obs.\ mean & vs.\ null & obs. & null & obs. & null \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\caption{Observed RCS profiles against a permuted-label null. The null draws 200 relabelings
of the pooled representations, preserving class sizes, applied identically at every layer.
Under relabeling the per-layer scale cancels between the numerator and the denominator of
Eq.~(2), so the expected score reduces to $\sqrt{1/n_{\mathrm{hf}}+1/n_{\mathrm{hl}}}=0.0634$
for the $260/6000$ split actually used, independent of layer, model, and representation
geometry. For Qwen2.5-7B this closed form is confirmed against a permutation null measured on
the model's own activations, which gives $0.0640$, a relative agreement of $0.95\%$. The
empirical tail probability is $0.000$ for every model on both concentration statistics.}
\label{tab:permutation-null}
\end{table}

\insight{The null is flat by construction, which fixes the reference points a reader needs:
a chance profile has Gini near zero and $n_{80}/L$ near $0.8$. Measured against it, every model
separates on two independent axes. Magnitudes sit 12--45$\times$ above chance, so RCS carrying
mass at every layer reflects a real harmful--harmless signal rather than the statistic being
nonnegative. And every profile is \emph{more} concentrated than chance, not
indistinguishable from flat. The honest reading is therefore ``distributed but measurably
non-uniform'', which is precisely the regime in which per-layer adaptive weighting has
something to exploit; a genuinely flat profile would make adaptive weighting pointless.}

\failuremode{A label permutation destroys class structure outright, so it tests whether
\emph{any} harmful-versus-harmless signal exists, never whether that signal is refusal
specific. Because the two classes are drawn from different corpora, SORRY-Bench and Alpaca,
topic, style, and source remain perfectly collinear with the label, and this control cannot
separate them. A matched-provenance contrast or a base-versus-aligned comparison would be
needed, and neither is available here. Note also that the ordering of models by Gini does not
track the ordering by LAMA benefit: Mistral-7B-RR has the highest observed Gini and the largest
null spread, yet is excluded from the alignment analysis entirely.}
""")

# ---------------------------------------------------------------- Table N2: interval procedure
rows = rd(os.path.join(V, "ci_candidateC_audit.csv"))
rows = [r for r in rows if int(r["n_seeds"]) > 1][:8]
body = "\n".join(
    r"%s & %s & [%s, %s] & [%s, %s] & %s \\" % (
        r["model"], r["method"].replace("LAMA-", "L-"),
        r["pub_lo"], r["pub_hi"], r["C_pooled_lo"], r["C_pooled_hi"], r["C_max_abs_dev_pp"])
    for r in rows)
parts.append(r"""
\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llccr}
\toprule
Model & Variant & Published CI & Pooled-independent & dev.\ (pp) \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\caption{Reconstructing the published confidence intervals. Resampling
$(\mathrm{behavior},\mathrm{seed})$ pairs as independent units reproduces them to a median
deviation of $0.28$ percentage points, whereas the hierarchical scheme described in the
supplement deviates by $2.48$ and a seed-averaged behavior bootstrap by $2.34$.}
\label{tab:ci-procedure}
\end{table}

\insight{The reconstruction identifies which estimator actually produced the reported
intervals, and it is not the one described. Pooling behavior--seed pairs multiplies the
effective sample size by the number of seeds, narrowing intervals by roughly $\sqrt{3}$.
Correcting this is cheap and strengthens rather than weakens the paper: the two starred
comparisons survive a behavior-level paired test, so the headline claims never depended on the
inflation, and saying so preempts the objection.}

\failuremode{The inflation is not benign, because the seeds are close to degenerate. On
Qwen2.5-7B and Gemma2-9B all three seeds yield identical per-behavior judgments, so the extra
$2n$ observations carry no information at all. This follows mechanically from greedy decoding
combined with a direction collection held fixed across configurations, and it means the
reported seed-to-seed stability is a property of the decoding rule rather than evidence of
robustness. Any future claim of multi-seed robustness needs a source of variation that
actually varies.}
""")

# ---------------------------------------------------------------- Table N3: corrected significance
rows = rd(os.path.join(V, "significance_multiseed_audit.csv"))
rows = [r for r in rows if r["model"] in ("Qwen2.5-7B", "Qwen2.5-3B", "Gemma2-9B")]
mark = {"star": r"$^{*}$", "dagger": r"$^{\dagger}$", "none": ""}
body = "\n".join(
    r"%s & %s & %s%s & %s & [%s, %s] & %s & %s \\" % (
        r["model"], r["variant"].replace("LAMA-", "L-"),
        r["published_delta_pp"], mark.get(r["paper_mark"], ""),
        r["recomputed_delta_pp"], r["ci_lo_pp"], r["ci_hi_pp"], r["paired_boot_p"],
        "holds" if r["agreement_with_paper"] == "AGREES" else r"\textbf{does not hold}")
    for r in rows)
parts.append(r"""
\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrrcrl}
\toprule
Model & Variant & Pub.\ $\Delta$ & Recomp. & 95\% CI & $p$ & Verdict \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\caption{Table~4 deltas re-tested with the behavior as the statistical unit, using a paired
bootstrap over seed-averaged per-behavior outcomes ($n=159$). Point estimates reproduce to
$0.05$ percentage points.}
\label{tab:corrected-significance}
\end{table}

\insight{Both starred claims survive, and all three Gemma2-9B negative results survive
decisively at $p<10^{-4}$, so the paper's substantive conclusions do not rest on the interval
defect. The correction also sharpens RQ5. Depth-specific direction construction alone, the
LAMA-uniform row, is not distinguishable from the control at $p=0.31$, while the full
LAMA-RCS configuration is. Read together with the benign-cost decomposition, where
LAMA-uniform at $\alpha_l=1$ already captures the entire perplexity reduction, the two design
axes divide cleanly: grouped directions buy the utility, RCS weighting buys the attack
success. That is a more specific claim than ``both components contribute'' and each half is
separately supported.}

\failuremode{The two marginal daggers do not survive and should be removed. More broadly,
with $159$ behaviors a paired bootstrap resolves differences of roughly $6$ percentage points;
the $+3.8$ and $+3.4$ effects are simply below what this evaluation can detect, and no
reanalysis of the same judgments will change that. Distinguishing them needs more behaviors,
not more seeds.}
""")

# ---------------------------------------------------------------- Table N4: depth trend
rows = rd(os.path.join(V, "depth_trend_audit.csv"))
body = "\n".join(
    r"%s & %s & %s & %s & %s$\times$ & %s$\times$ & %s$\times$ \\" % (
        r["model"], r["spearman_gap_vs_depth"], r["spearman_harmfulspread_vs_depth"],
        r["spearman_rcs_vs_depth"], r["gap_dynamic_range"], r["spread_dynamic_range"],
        r["rcs_dynamic_range"])
    for r in rows)
parts.append(r"""
\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lrrrrrr}
\toprule
& \multicolumn{3}{c}{Spearman $\rho$ with depth} & \multicolumn{3}{c}{Dynamic range} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Model & gap & spread & RCS & gap & spread & RCS \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\caption{What the denominator of Eq.~(2) removes. The raw centroid gap and the harmful
intra-class spread each grow almost monotonically with depth across one to four orders of
magnitude; their ratio does not.}
\label{tab:depth-trend}
\end{table}

\insight{This answers directly what information RCS retains beyond the raw centroid norm: the
normalization removes a depth trend that the numerator and denominator share. Both grow
together by up to four orders of magnitude, while RCS varies only $1.7$--$7.4\times$ and is far
less tied to depth. Llama-2-7B is the clean demonstration, where the raw gap correlates
$-0.07$ with the causal profile while RCS reaches $0.466$, rising to $0.625$ once the gap is
controlled for. A statistic dominated by residual-stream scale cannot rank layers; removing
that scale is what makes a ranking possible at all.}

\failuremode{The residual is a predictive association on some models, not proof of refusal
specificity and not a general causal ranking. On Gemma2-9B the partial correlation is $-0.219$,
so after removing the shared component nothing causal remains, which is the same model where
LAMA-RCS fails. Depth-detrending is therefore necessary for a usable ranking but plainly not
sufficient for a reliable one. The normalizations a reader may ask about, by the harmless
spread and by the pooled spread among others, are now computed directly from the regenerated
activations and reported in Table~\ref{tab:alt-normalizations}.}
""")

# ---------------------------------------------------------------- Table N5: normalizations
rows = rd(os.path.join(V, "alt_normalizations_qwen2-7b.csv"))
def esc(x): return x.replace("_", r"\_")
body = "\n".join(
    r"\texttt{%s} & %s & %s & %s \\" % (
        esc(r["normalization"]), r["spearman_vs_causal"],
        r["delta_vs_rcs"] if r["delta_vs_rcs"] else "---",
        r["partial_given_raw_gap"] if r["partial_given_raw_gap"] else "---")
    for r in rows)
parts.append(r"""
\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lrrr}
\toprule
Normalization & $\rho$ (causal) & $\Delta$ vs.\ RCS & partial $\mid$ gap \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\caption{Sensitivity of the layer ranking to the denominator of Eq.~(2), computed on
regenerated Qwen2.5-7B activations that reproduce the published profile to
$2{\times}10^{-4}$ (Spearman $1.0000$). Column~1 is the rank correlation between each
normalized profile and the measured causal profile over $L=28$ layers; column~2 its
difference from RCS; column~3 the partial correlation after controlling for the raw
centroid gap.}
\label{tab:alt-normalizations}
\end{table}

\insight{The conclusion does not depend on which spread is used in the denominator, but it
does depend on dividing by one. All seven normalized variants land between $0.867$ and
$0.873$, a spread of $0.006$, while the unnormalized gap reaches only $0.744$. Normalizing is
worth about $0.13$ in rank correlation; choosing among the normalizers is worth almost nothing.
The partial correlations tell the same story from the other side: every normalized variant
retains $0.67$--$0.68$ after the raw gap is controlled for, so they all carry the same
additional information, and the alternatives carry essentially none that RCS lacks, with
differences of $-0.0005$ to $-0.006$. This is the strongest available answer to the
normalization question: RCS is a sufficient choice, and the layer ranking is robust to the
reviewer's substitutions.}

\failuremode{It also means RCS is not a \emph{distinguished} choice, and the paper should not
imply that it is. With $L=28$ layers the standard error of a Spearman correlation is roughly
$1/\sqrt{L-1}\approx0.19$, which is thirty times the largest gap between variants. The ordering
among the seven is therefore noise, and two of them, \texttt{pooled\_std\_trace} and
\texttt{snr\_fisher}, are indistinguishable from RCS at $-0.0005$. Any claim that the harmful
intra-class spread is the principled denominator is unsupported by this evidence; the defensible
claim is that it is one of several equivalent ones. Two further limits: this is a single model,
and $\rho$ against the causal profile is an association, so a normalizer that ranks layers well
here is not thereby shown to identify refusal mechanism.}
""")

# ------------------------------------------------- Table N6: refusal-specificity controls
MC = os.path.join(REPO, "results", "matched_controls")
mrows = rd(os.path.join(MC, "matched_controls_qwen2-7b.csv"))
nrows = rd(os.path.join(MC, "nonrefusal_controls_qwen2-7b.csv"))
def esc2(x): return x.replace("_", r"\_")
def mline(name, nf, nl, mean, peak, gini, rho, ratio, bold=False):
    f = (lambda t: r"\textbf{%s}" % t) if bold else (lambda t: t)
    return r"%s & %s & %s & %s & %s & %s & %s & %s$\times$ \\" % (
        f(r"\texttt{%s}" % esc2(name)), f(nf), f(nl), f(mean), f("L%s" % peak),
        f(gini), f(rho), f(ratio))
pa = [mline(r["control"], r["n_f"], r["n_l"], r["mean_rcs"], r["peak_layer"], r["gini"],
            r["spearman_vs_published"], r["ratio_obs_over_null"],
            bold=r["control"] == "published_equivalent") for r in mrows]
pb = [mline(r["contrast"], r["n_a"], r["n_b"], r["mean_rcs"], r["peak_layer"], r["gini"],
            r["spearman_vs_published"], r["ratio_obs_over_null"],
            bold=r["contrast"] == "published_harmful_vs_harmless") for r in nrows]
parts.append(r"""
\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrrrrr}
\toprule
Contrast & $n_f$ & $n_l$ & mean & peak & Gini & $\rho_{\mathrm{pub}}$ & vs.\ null \\
\midrule
\multicolumn{8}{l}{\emph{(a) the refusal contrast, with length and register matched away}} \\
""" + "\n".join(pa) + r"""
\midrule
\multicolumn{8}{l}{\emph{(b) non-refusal contrasts, formed inside a held-out harmless slice}} \\
""" + "\n".join(pb) + r"""
\bottomrule
\end{tabular}
\caption{Refusal-specificity controls on Qwen2.5-7B. Panel~(a) resamples the harmless class to
match the harmful token-length histogram stratum by stratum, restricts to the imperative
register, and does both, on one side and on both sides. Panel~(b) builds each contrast entirely
inside harmless prompts $6000$--$12999$, held out from the published set, so neither class
carries refusal content; class sizes are pinned at $260/6000$ so the null is identical.
$\rho_{\mathrm{pub}}$ is the rank correlation with the published profile.}
\label{tab:refusal-specificity}
\end{table}

\insight{Depth, not magnitude, is what carries refusal-specific information. Panel~(a) shows
the profile is untouched by the obvious confounds: through every matching the peak holds at
layer $19$ or $20$, Gini moves only from $0.188$ to $0.167$, and $\rho_{\mathrm{pub}}$ never
falls below $0.974$. Panel~(b) then supplies the discriminator. Every non-refusal contrast is
\emph{anti}-correlated with the published profile, $\rho_{\mathrm{pub}}$ between $-0.56$ and
$-0.84$, and every one peaks in layers $4$ to $5$, whereas the refusal contrast peaks at layer
$19$. Generic semantic and syntactic separability is an early-layer phenomenon; what the
harmful-versus-harmless comparison tracks is a late-layer one. Length is eliminated outright: a
length contrast tuned to the same Cliff's delta as the real one, $0.145$ against a target of
$0.148$, reaches only $1.3\times$ the null against the real contrast's $18.7\times$. The random
split lands at $1.0\times$, confirming the null by a third independent route after the closed
form and the permutation test.}

\failuremode{RCS is not exclusively a refusal detector, and the paper should not be read as
claiming it is. A code-versus-prose topic contrast reaches $9.0\times$ the null and a
question-versus-imperative register contrast $11.2\times$; any strong semantic contrast produces
a substantial score. Worse for the concentration argument, the register contrast's Gini of
$0.1908$ slightly \emph{exceeds} the published $0.1881$, so concentration alone separates
nothing. The defensible claim is therefore narrower than refusal measurement in general: score
magnitude and Gini are not refusal specific, and only the location of the mass is. Three limits
remain. These are one model, the only cached checkpoint. The topic split is keyword based,
because the \texttt{category} field is uniformly \texttt{"None"} in both training splits. The
control that settles the magnitude half of the question, an aligned-versus-base comparison on
identical prompts, has since been run at the full published class sizes of 260 and 6000: a
checkpoint never trained to refuse reaches $60.2\%$ of the aligned mean RCS, so magnitude
claims must be dropped rather than softened, while the late concentration is alignment's own and
survives.}
""")

# ---------------------------------------------- Tables N7/N8: joint causal ablation (PKr6)
JC = os.path.join(REPO, "results", "joint_causal")
jc_path = os.path.join(JC, "joint_causal_qwen2-7b.json")
if os.path.exists(jc_path):
    jc = json.load(open(jc_path))
    dc, ds, sm = jc["delta_cumulative_rcs"], jc["delta_single"], jc["summary"]
    add, order, g = sm["additivity"], jc["rcs_order"], sm["published_probe_guard"]
    base_p = sm["baseline_p_refuse"]
    dmax = max(dc); kmax = dc.index(dmax) + 1

    ks = [k for k in (1, 2, 3, 4, 5, 8, 12, 14, 20, 28) if k <= len(dc)]
    body7 = "\n".join(
        str(k) + " & " + str(order[k - 1]) + " & " + f"{dc[k-1]:.4f}" + " & "
        + f"{100.0*dc[k-1]/dmax:.1f}" + " & "
        + ("--" if add.get(str(k), {}).get("ratio") is None
           else f"{add[str(k)]['ratio']:.3f}") + r" \\"
        for k in ks)

    parts.append(r"""
\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{rrrrr}
\toprule
$k$ & layer added & $\Delta P$ & \% of best & joint\,/\,$\Sigma$\,singles \\
\midrule
""" + body7 + r"""
\bottomrule
\end{tabular}
\caption{Joint multi-layer ablation of Qwen2.5-7B-Instruct, cumulative over layers ranked by
RCS. $\Delta P$ is the drop in refusal-token probability mass at the first generated position
over 159 \texttt{harmbench\_test} prompts, so no text is generated and no judge is needed. The
last column divides the joint effect by the sum of those same layers' individual effects.
Baseline $P_{\text{refuse}} = """ + f"{base_p:.4f}" + r"""$; the largest attainable drop is
$""" + f"{dmax:.4f}" + r"""$ at $k = """ + str(kmax) + r"""$.}
\label{tab:joint-causal-curve}
\end{table}

\insight{This is the causal form of the paper's central claim, and it both supports and narrows
it. The best \emph{single} layer, L""" + str(sm["best_single_layer"]) + r""", removes only
$""" + f"{100.0*sm['best_single_delta']/dmax:.1f}" + r"""\%$ of what joint ablation removes, so
no layer is the site of refusal and single-layer localization is refuted causally rather than
by inference. At $k=2$ and $k=3$ the joint effect exceeds the sum of its parts by about
$30\%$, while still far from the ceiling that saturation would impose, so the layers are
complementary rather than redundant: refusal is \emph{jointly} encoded, not duplicated in each
layer. The harness earns this reading by reproducing the paper's own single-layer probe to
$\max|\mathrm{diff}| = """ + f"{g['delta_single_max_abs_dev']:.1e}" + r"""$ with Spearman
$""" + f"{g['delta_single_spearman_vs_published']:.5f}" + r"""$, and by returning to the
baseline exactly once every ablation is undone.}

\failuremode{Three layers reach $""" + f"{100.0*dc[2]/dmax:.1f}" + r"""\%$ of the attainable
effect and four reach $""" + f"{100.0*dc[3]/dmax:.1f}" + r"""\%$, against the representational
$n_{80}/L = 0.679$, which would imply 19 of 28 layers. Causal concentration is roughly six
times tighter than representational spread, so the title is defensible as ``no single layer
suffices and the layers act jointly'' and \emph{not} as ``refusal is distributed across the
network''. The curve is also non-monotone: past $k = """ + str(kmax) + r"""$ it declines, from
$""" + f"{dmax:.4f}" + r"""$ to $""" + f"{dc[-1]:.4f}" + r"""$ at $k = 28$, so a uniform
all-layer intervention is strictly worse than ablating the band. That is a causal argument in
favor of a layer-selective method which the paper does not currently make, and it also means
``more ablation is better'' is false.}
""")

    rdp = os.path.join(JC, "random_direction_qwen2-7b.json")
    rdj = json.load(open(rdp)) if os.path.exists(rdp) else None
    sel = jc["configs"]["selection"]
    body8 = "\n".join(
        str(k) + r" & \textbf{" + f"{sel['rcs_top'][str(k)]['delta']:.4f}" + "} & "
        + f"{sel['rcs_bottom'][str(k)]['delta']:.4f}" + " & "
        + f"{sel['norm_top'][str(k)]['delta']:.4f}" + " & "
        + f"{sel['random'][str(k)]['mean']:.4f}" + r"\," + "("
        + f"{sel['random'][str(k)]['max']:.4f}" + ") & "
        + ("--" if not rdj or str(k) not in rdj["by_k"]
           else "$" + f"{rdj['by_k'][str(k)]['random_mean']:+.4f}" + "$") + r" \\"
        for k in jc["k_grid"])

    parts.append(r"""
\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{rrrrrr}
\toprule
& \multicolumn{4}{c}{choice of layers (refusal direction)} & random dir. \\
\cmidrule(lr){2-5}\cmidrule(lr){6-6}
$k$ & top-$k$ RCS & bottom-$k$ & top-$k$ norm & random, mean (best of 5) & at top-$k$ RCS \\
\midrule
""" + body8 + r"""
\bottomrule
\end{tabular}
\caption{Matched-budget controls for the joint ablation. Every arm ablates exactly $k$ layers,
so the intervention budget is fixed and only the choice of layers, or of direction, varies. The
final column holds the layer set at the top-$k$ by RCS and substitutes a random unit direction;
the ablation normalizes its input, so the magnitude of the weight edit is matched by
construction.}
\label{tab:joint-causal-controls}
\end{table}

\insight{This is what makes the claim causal rather than correlational. At $k = 3$, layers
chosen by RCS remove $""" + f"{sel['rcs_top']['3']['delta']:.4f}" + r"""$ of the refusal mass,
against $""" + f"{sel['norm_top']['3']['delta']:.4f}" + r"""$ for the layers with the largest
raw diff-in-means norm, a factor of
$""" + f"{sel['rcs_top']['3']['delta']/sel['norm_top']['3']['delta']:.1f}" + r"""$; against
$""" + f"{sel['random']['3']['mean']:.4f}" + r"""$ for random layers and
$""" + f"{sel['random']['3']['max']:.4f}" + r"""$ for the luckiest of five draws; and against
$""" + f"{sel['rcs_bottom']['3']['delta']:.4f}" + r"""$ for the lowest-RCS layers. The raw-norm
column is the non-circularity control, since RCS shares its numerator with the ablated
direction, and the normalization earns its place at small $k$. The final column settles the
remaining objection: a random rank-1 edit at the same layers moves refusal by less than
$0.001$ at every $k$, so the effect belongs to the direction and not to the perturbation.}

\failuremode{The advantage is confined to small $k$. By $k = 8$ the raw-norm rule has
essentially caught up, $""" + f"{sel['norm_top']['8']['delta']:.4f}" + r"""$ against
$""" + f"{sel['rcs_top']['8']['delta']:.4f}" + r"""$, and by $k = 14$ the two are identical,
because every rule saturates once the late band is covered. Any claim that RCS is the better
layer ranking must therefore be made in the small-$k$ regime an adaptive method actually
operates in, not as a general property. Ratios against the random-direction column should also
be avoided: its mean sits within $0.001$ of zero, so the ratios stored in the CSV are
numerically unstable, and the defensible statement is simply that a random direction does
nothing.}
""")


open(OUT, "w").write("\n".join(parts))
print("wrote", OUT)
print("tables:", sum(1 for p in parts if r"\begin{table}" in p),
      "| insight blocks:", sum(p.count(r"\insight{") for p in parts),
      "| failure-mode blocks:", sum(p.count(r"\failuremode{") for p in parts))
