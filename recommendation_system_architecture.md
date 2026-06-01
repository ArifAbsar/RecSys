# Architecture & Mathematical Foundation of the Adaptive Business-Aware Recommendation System

## 1. Introduction and Problem Statement
Modern e-commerce recommendation engines typically optimize purely for user relevance (predicting the highest likelihood of conversion or click). However, in real-world retail environments, optimizing *only* for relevance ignores critical business constraints and objectives, such as:
1. **Inventory & Campaign Commitments:** Over-indexing on popularity leaves strategic items (promotions, clearances, brand campaigns) undiscovered.
2. **Catalog Monopolization & Fatigue:** Pure AI scores often lead to a "rich get richer" loop where the same top 1% of products are repeatedly recommended, causing user fatigue and starving the long-tail catalog.
3. **Cold-Start Sparsity:** New users lack enough history for personalization, and new items lack the interaction data required to surface organically.

This system solves these issues through a **Business-Aware Adaptive Pipeline**. It injects deterministic business logic (quotas, promotions) and probabilistic diversity (Thompson Sampling) into the raw AI scoring mechanism. The result is an engine that maximizes relevance while strictly adhering to commercial strategies and catalog exploration.

---

## 2. System Architecture Overview
The system is divided into two distinct components:
1. **The Mapping Engine (Data-to-Intent):** An autonomous data pipeline that maps raw, heterogeneous database schemas (e.g., Shopify, custom ERPs) into a unified semantic schema (`universal_ecommerce_schema`). This ensures the downstream model understands the fundamental business intent behind the data (e.g., recognizing "clearance" flags or "margin").
2. **The Model Engine (Adaptive Inference):** The scoring and ranking pipeline that evaluates the catalog, applies strategic boosts, and samples final recommendations using a multi-armed bandit approach.

---

## 3. Mathematical Formulation of the Base Score
The core of the Model Engine calculates a *Base Score* for every candidate item $i$ for a given user $u$. The score is a weighted linear combination of four primary signals, dynamically adjusted based on the user's data density.

### 3.1 The Base Score Equation
For a user $u$ and item $i$, the base score $S_{u,i}$ is defined as:

$$ S_{u,i} = (W_{cur} \cdot AI_{i}) + (W_{strat} \cdot B_{i}) + (W_{per} \cdot P_{u,i}) + (W_{pop} \cdot Pop_{i}) + (0.40 \cdot C_{i}) + J_{i} $$

Where:
- $AI_{i} \in [0, 1]$ is the normalized **AI Relevance Score** (e.g., from a DeepFM model).
- $B_{i} \in [0, 1]$ is the **Global Strategic Score** (e.g., high margin, high inventory).
- $P_{u,i} \in [0, 1]$ is the **Personalization Match Score**, computed by multiplying the user's affinity for item $i$'s themes by a multiplier (default: $1.5$).
- $Pop_{i} \in [0, 1]$ is the **Popularity Score** of the item.
- $C_{i} \in \{0, 1\}$ is a **Campaign Flag** (1 if promoted/clearance, 0 otherwise).
- $J_{i} \sim U(0, \epsilon)$ is a **Discovery Jitter** applied to cold-start users to break ties and introduce micro-exploration (default $\epsilon = 0.05$).

### 3.2 Adaptive Weight Shifting
The weights vector $\vec{W} = \langle W_{cur}, W_{strat}, W_{per}, W_{pop} \rangle$ is not static. It shifts dynamically based on the user's interaction history (data density).

Let $N_u$ be the number of historical interactions for user $u$:
1. **Cold-Start User** ($N_u < 5$): 
   $\vec{W} = \langle 0.65, 0.10, 0.15, 0.10 \rangle$ 
   *Focus on broad curation and discovery.*
2. **Moderate User** ($5 \leq N_u < 20$):
   $\vec{W} = \langle 0.30, 0.10, 0.50, 0.10 \rangle$ 
   *Balanced approach as intent solidifies.*
3. **Dense / Power User** ($N_u \geq 20$):
   $\vec{W} = \langle 0.15, 0.10, 0.70, 0.05 \rangle$ 
   *Highly personalized, relevance-driven.*

---

## 4. Solving Exposure Bias: Multi-Armed Bandit (Thompson Sampling)
Traditional recommendation systems use hardcoded heuristic caps (e.g., "do not show item $i$ to more than 10% of users") to prevent popular items from monopolizing the feed. These hard limits are brittle and non-mathematical.

This system replaces heuristic bounds with **Thompson Sampling**, a multi-armed bandit algorithm that uses the **Beta Distribution** to naturally balance Exploration (showing new items) vs. Exploitation (showing proven popular items).

### 4.1 Beta Distribution Formulation
For every item $i$ in the candidate pool, we sample a random multiplier $M_i$ from a Beta distribution:

$$ M_i \sim \text{Beta}(\alpha_i, \beta_i) $$

Where:
- $\alpha_i$ represents our **confidence** in the item's quality.
- $\beta_i$ represents the **fatigue** or exposure the item has already received in the current inference batch.

### 4.2 Deriving Alpha and Beta
We parameterize the Beta distribution as follows:

$$ \alpha_i = 1.0 + K \cdot (AI_i + Pop_i) $$
$$ \beta_i = 1.0 + E_i $$

- $K$ is the `thompson_alpha_scale` (default $50.0$), which controls the strictness of the exploit behavior.
- $E_i$ is the exact number of times item $i$ has been selected for recommendation across all users in the current batch.

### 4.3 The Mathematical Elegance of the Approach
The final score for the item becomes:
$$ \text{FinalScore}_{u,i} = S_{u,i} \cdot M_i $$

**Why this works beautifully:**
1. **High Quality, Low Exposure (Exploit):** If an item has high AI/Popularity scores ($\alpha$ is large) and hasn't been shown much ($\beta \approx 1$), the Beta curve skews heavily to the right. $M_i$ will consistently sample near $1.0$, preserving its high base score.
2. **Over-Exposure Penalty (Fatigue):** As an item gets repeatedly recommended, $E_i$ grows, causing $\beta_i$ to increase rapidly. The Beta curve collapses toward $0$. $M_i$ will sample near $0$, organically crushing the item's final score without a hard cap, allowing other items to surface.
3. **Cold-Start Exploration:** A brand new item has low $\alpha$ and $\beta=1$. Its Beta curve is wide and uncertain. Occasionally, it will randomly sample a high $M_i$, giving it an organic chance to be recommended and "prove" itself, solving the cold-start item problem.

---

## 5. Quota-Based Greedy Selector
Once the final scores are computed, the system must translate them into a concrete list of top $N$ recommendations (e.g., $N=10$) that respects business quotas.

The system uses a **Greedy Quota-Based Selection** algorithm. Rather than just taking the top 10 items, it categorizes each item into a strategic bucket and fills them according to business needs.

### 5.1 Categorization Hierarchy
Each candidate item is evaluated and assigned exactly one label, strictly in this order:
1. **Promotion:** If the item is flagged as a campaign/clearance item and campaigns are enabled.
2. **Personalization:** If the item matches the user's top themes OR its Personalization Score $> \tau_{per}$ (dynamically calculated as the 80th percentile of the batch).
3. **Curation:** If the item's AI Score $> \tau_{cur}$ (dynamically calculated as the 60th percentile).
4. **Fallback:** Defaults to Curation.

### 5.2 Quota Fulfillment
The business defines limits, for example: `perso_quota = 7`, `curation_quota = 2`, `promo_quota = 1`.
The greedy selector iterates through the sorted items (by Final Score) and selects them if:
- The quota for their assigned category is not yet full.
- OR every other category's quota is already completely exhausted (graceful fallback).

This guarantees that the final payload sent to the frontend has the exact strategic mix the business requires, complete with human-readable "explainability" strings (e.g., *"Matched your interest in Outdoor Gear"* vs *"Featured brand selected for you"*).

---

## 6. Conclusion
This architecture successfully bridges the gap between state-of-the-art machine learning and practical business operations. By abstracting the heavy lifting of exploration/exploitation into a mathematically pure Thompson Sampling layer, and enforcing business logic via a deterministic quota selector, the system achieves maximum relevance without sacrificing catalog diversity or strategic objectives.
