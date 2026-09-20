# Spending-behaviour clustering

Code: `backend/app/ml/clustering/` (model, evaluate, train),
features in `backend/app/features/monthly_features.py` (`monthly_share_vectors`),
serving in `backend/app/services/profile.py`, endpoint `GET /analytics/spending-profile`.
Results: `docs/metrics/clustering.json`.

## The problem

Group users by *how* they spend, without being told the groups. This is
**unsupervised learning (clustering)**: no labels during training. Output: a named
profile such as "Higher-spend, Housing / Entertainment-heavy", used by the
dashboard and the LLM to say "you spend like a ...".

## Features

One point per (user, window): **share of spend per category** (11 numbers that sum
to 1) plus **log total monthly spend**. Shares make a Rs 30k earner and a Rs 150k
earner comparable; the log total keeps overall level as a separate signal.
All features are standardized (mean 0, std 1) so no feature dominates by scale.

## KMeans, in plain English

Pick k centres; assign every point to its nearest centre; move each centre to the
mean of its points; repeat until stable. `n_init=20` reruns from 20 random starts
and keeps the best, because KMeans can get stuck in poor local solutions.

## Choosing k without labels

- **Silhouette score** (-1..1): for each point, compare its average distance to its
  own cluster (a) with the nearest other cluster (b): `(b - a) / max(a, b)`.
  Higher = tighter, better-separated clusters.
- **Davies-Bouldin** (lower better), **stability** (rerun with different seeds; do
  we get the same groups? measured with ARI).
- Rule I used: best silhouette with 3 <= k <= n_users/2. Why the cap: silhouette
  keeps rising as k grows because with 12 users extra clusters just split users
  apart; requiring >= 2 users per cluster on average avoids that.

## Grading against ground truth (only possible because data is synthetic)

- **ARI (Adjusted Rand Index)**: agreement between two groupings, 0 = random,
  1 = identical. Corrected for chance.
- **NMI**: how much knowing the cluster tells you about the persona (0..1).

## The most important story: the noise experiment

First version clustered single months. Result was poor (ARI 0.34; a held-out user
matched their persona only 17-25% of the time; a trivial "bin by total spend"
baseline got ARI 0.31). Hypothesis: one month is a noisy sample of a stable
behaviour (travel is 0% one month, 40% the next). Test: average spend over a
trailing window before computing shares.

| Window (months) | Points | Best silhouette | ARI at that k |
|---|---|---|---|
| 1 | 288 | 0.257 | 0.338 |
| 3 | 264 | 0.306 | 0.272 |
| 6 | 228 | 0.329 | 0.159 |
| **12** | 156 | **0.448** | **0.771** |

The window was chosen by silhouette (no labels). This is a classic
"signal-to-noise via aggregation" result and a great interview story.

## Final results (window 12, k = 6)

| Method | Silhouette | ARI | NMI |
|---|---|---|---|
| KMeans | 0.448 | 0.771 | 0.902 |
| Ward hierarchical | 0.448 | 0.771 | 0.902 |
| Gaussian mixture | 0.448 | 0.771 | 0.902 |
| Baseline: bins of total spend | 0.269 | 0.725 | 0.808 |

- All three algorithms agree: the data has real cluster structure.
- Cluster x persona table: 4 personas map to their own cluster; young
  professionals and big spenders merge (similar shares at different scales after
  standardizing); the students split into two clusters.

## Honest limitations (say these first)

1. **The spend-level baseline is close** (ARI 0.725 vs 0.771). In my synthetic data
   personas differ mostly by how much they spend, so category mix adds only a
   little. Real data may differ.
2. **Overlapping windows from one user are near-duplicates**, which inflates
   silhouette and ARI. The fair test is leave-one-user-out.
3. **Leave-one-user-out user-level persona accuracy is only 50%** (6 of 12 users).
   With just 2 users per persona, holding one out leaves a single example of that
   persona. More users would help.
4. The persona labels are synthetic, and 12 users is a small population.
5. Profiles need history: the model was trained on 12-month windows; with fewer
   months the API still answers but sets `is_full_window: false`, and with fewer
   than 3 months it returns null.

## How profiles are named (no LLM involved)

From each centroid: spend level by rank ("Lower-spend / Mid-spend / Higher-spend",
*relative to the other profiles*, and the actual typical monthly spend is shown),
plus the categories whose share stands out from the average by more than 3
percentage points.

## Concepts to be able to explain

- Unsupervised learning; what "no labels" means for evaluation.
- KMeans algorithm and its limits (needs k, assumes round clusters, sensitive to scale).
- Standardization before distance-based methods.
- Silhouette, Davies-Bouldin, stability; internal vs external metrics; ARI/NMI.
- Bias in choosing k; why silhouette can keep rising.
- Feature aggregation to reduce noise (the window experiment).

## Interview questions

**Q: How did you choose k?**
A: Silhouette score, with k between 3 and n_users/2. Personas were only used
afterwards for grading, never for choosing.

**Q: How do you know clusters are meaningful, not random?**
A: Internal metrics (silhouette 0.45), stability across seeds, agreement between
three different algorithms, and agreement with ground-truth personas (ARI 0.77,
NMI 0.90), plus leave-one-user-out.

**Q: Your first result was poor. What did you do?**
A: I hypothesized single-month vectors were noise, tested trailing windows of 1,
3, 6 and 12 months, chose the window by silhouette, and ARI rose from 0.34 to 0.77.

**Q: Why is the simple baseline so close?**
A: Because in the synthetic data spending level is the main persona difference; I
report it rather than hide it.

**Q: What are the weaknesses?**
A: 12 users, synthetic personas, near-duplicate overlapping windows, and only 50%
user-level recovery on held-out users. With real data I would collect more users.

**Q: Why not use the personas to train a classifier instead?**
A: Real users have no persona labels; clustering discovers structure without them.

## Reading

- scikit-learn: Clustering (KMeans, hierarchical, GMM): https://scikit-learn.org/stable/modules/clustering.html
- scikit-learn: Silhouette analysis example: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
- scikit-learn: Adjusted Rand index: https://scikit-learn.org/stable/modules/clustering.html#adjusted-rand-index
- scikit-learn: Preprocessing / scaling: https://scikit-learn.org/stable/modules/preprocessing.html
- StatQuest (YouTube): "K-means clustering", "Hierarchical clustering", "Silhouette" (search the channel)
