# Lecture 06 · Level 10 — Clustering and Dimensionality Reduction

> An unsupervised-learning exercise: group customers into similar segments with no answer key (k-means), and squeeze several variables into a 2-D map you can inspect by eye (PCA).
**Difficulty** ⭐⭐⭐⭐ / **Prerequisites** level01, level04 / **Estimated time** 50 min

## 1. Why Learn This — The Business View

"What kinds of customers make up our base?" has no answer labels. Unlike churn prediction, where past answers (cancellation records) exist, this is a problem where **the structure itself must be discovered**. Customer segmentation is marketing's old homework — a company that texts every customer the same message performs differently from one that sends "discount coupons to the price-sensitive, exclusive perks to VIPs, reminders to the dormancy-risk group."

Clustering is the tool that lets the data do that segmentation, and dimensionality reduction is the tool that presses data with dozens of variables into a 2-D map the human eye can see. The two are often used as a set: build groups with clustering, then color them onto a dimensionality-reduction map to visually verify "are the groups really separated?"

## 2. Understanding by Analogy

**k-means is a food-court desk-placement problem.** Customers (data points) sit scattered across a wide hall, and you must decide where to put k movable help desks (centroids). A good placement is one where every customer's walk to their nearest desk is short overall. The k-means algorithm is simple repetition: (1) put the desks anywhere → (2) assign each customer to the nearest desk → (3) move each desk to the center of its own customers → repeat (2) and (3) until nothing moves. That simple back-and-forth divides the customers into k natural crowds.

**PCA is choosing a shadow angle.** Suppose you hold a complex wire sculpture (high-dimensional data) and project its shadow (2-D) onto a wall. Choose a bad angle and the sculpture flattens into a single line; choose well and the shadow keeps the structure alive. PCA (principal component analysis) is the tool that mathematically finds **the angle at which the data appears most spread out** — the projection direction with the least information loss. The "first principal component" is the axis of widest spread; the "second" is the next.

## 3. Core Concepts

### 3.1 What to Know When Using k-means

- **A human sets k**: the data won't tell you what number of groups is "correct." As a reference tool there's the elbow method — grow k while plotting total within-cluster distance (inertia) and look for where the decrease bends sharply — but the final call is a business judgment: "does this number of groups split into distinct marketing actions?"
- **Standardization is mandatory**: it's distance-based, so a large-unit variable (annual spend, in the millions of KRW) overwhelms a small-unit one (visit count).
- **Starting-position luck**: results can vary with the initial placement, so it's run multiple times and the best is kept (`n_init`).
- **Limitations**: it assumes round crowds, so it struggles with elongated or donut-shaped ones. Every point must be assigned somewhere, so outliers get squeezed into some group too.

### 3.2 Interpreting Clusters — the Half the Machine Can't Do

All k-means gives you is numbers: "group 0, 1, 2, 3." **Looking at each group's average profile and naming it — "thrifty newcomers," "VIP regulars" — is human work**, and only named groups produce actions. This is precisely the "interpretation cost of unsupervised learning" from level01. If a profile won't interpret, change k or the variable set and go again — that round trip is the normal process.

### 3.3 What to Know When Using PCA

- **Explained variance ratio**: each principal component reports what % of the original information (variance) it carries. If PC1+PC2 = 70%, "the 2-D shadow holds 70% of the original" — your basis for how much to trust the map.
- **Interpreting the axes**: each principal component is a weighted sum of the original variables. Reading the weights (loadings) lets you say things like "PC1 is roughly a 'spending scale' axis, PC2 a 'discount sensitivity' axis."
- **Uses**: beyond visualization, it serves as preprocessing to compress model inputs when there are too many variables (with a noise-removal side effect).
- **Standardization mandatory here too**: it's variance-based, so large-unit variables monopolize the axes.

### 3.4 Why Evaluating Unsupervised Learning Is Fuzzy

With no answers there's no accuracy-style metric. Internal measures exist, like the silhouette score (how close to your own cluster and far from others), but they're auxiliary; the ultimate evaluation is downstream: **"did a campaign run on this segmentation actually perform?"**

## 4. Hands-On — main.py

Run it:

```bash
python3 main.py
```

We perform the full segmentation workflow on synthetic customer data.

- **[1]** Use `make_blobs` to create 800 customers with "4 hidden types," converting each axis to realistic scales: annual spend, visits, discount-usage rate, tenure months. (We never tell the model the types — recreating the unsupervised situation.)
- **[2]** After standardizing, run k-means for k=2–8, print the elbow table (inertia) and silhouette scores, and choose k=4.
- **[3]** Print the k=4 **profile table** (per-cluster averages of spend, visits, discount rate, tenure, plus headcount), and auto-suggest a name for each cluster ("VIP regulars" and the like) — read the naming-rule code and think about how you would name them.
- **[4]** Press the 4 variables into 2-D with PCA and print the explained variance ratios and each axis's loadings.
- **[5]** Save a scatter plot with clusters colored on the PCA 2-D map to `outputs/segments_pca.png`. Check in the figure whether the clusters really look separated.

The heart of the code: `KMeans(n_clusters=k, n_init=10)` and `PCA(n_components=2)`. And [3]'s profile aggregation (`groupby("cluster").mean()`) is the table you'll stare at longest in real work.

## 5. Try It Yourself

1. **(Easy)** In [2], do the final clustering with k=3 and k=6 instead of k=4. How does the profile table change? From the standpoint of "how many groups can the marketing team actually manage," what number looks right?
2. **(Medium)** Increase `cluster_std` in [1] (spreading the crowds wider) and experiment. Watch the elbow go blunt and the silhouette drop — the signal of "data with weak cluster structure." Real-world data usually lives closer to this end.
3. **(Challenge)** Remove standardization (drop the scaler) and rerun k-means, comparing the profiles. Confirm that annual spend (the large-unit variable) monopolizes the clustering, and explain why 3.1 called standardization mandatory.

## 6. Common Mistakes

- **Skipping standardization**: both clustering and PCA get dominated by large-unit variables. The single biggest trap in this level.
- **Elevating k-means output to "discovered truth"**: k-means will split any data into k groups, **no matter what** — even with no structure present. Cross-check the structure's existence with silhouette, elbow, and visualization.
- **Putting raw cluster numbers in a report**: "cluster 2 has high churn" communicates nothing. Give profile-based names. Also, cluster numbers are arbitrary and can swap on a rerun.
- **Interpreting PCA axes as physical realities**: principal components are mathematical directions, not "the customer's nature." Name them cautiously from the loadings, and if explained variance is low, don't over-trust the map itself.
- **Skipping action design after segmentation**: countless segmentation projects end at drawing groups. Level02's spec principle (analysis without action is a decoration) applies to unsupervised learning verbatim.

## Next Level Preview

The final level. We return to supervised learning and push the churn model's performance with gradient boosting — the method that conquered both data competitions and industry — and then learn the techniques that make the stronger-but-more-opaque model transparent again: permutation importance and per-prediction explanations.
