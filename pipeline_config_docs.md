# Adaptive Semantic Recommendation Pipeline: Exhaustive Configuration Architecture

This document provides a highly exhaustive, variable-by-variable technical breakdown of every single parameter in the `PipelineConfig` class. This architecture relies on dynamic percentiles, temporal decay algorithms, adaptive weighting vectors, and stochastic sampling (Thompson Sampling) to create a self-healing, scale-invariant recommendation system.

---

## 1. Batch & Output Parameters
These define the core throughput and volume of the inference pipeline.

### `batch_size = 32`
* **The Problem:** Processing inference for 10,000 users one at a time on a GPU leaves 99% of the parallel processing cores idle, resulting in massive latency.
* **The Solution & Math:** Groups users into chunks of 32. The RecBole inference engine mathematically evaluates all 32 users against the entire catalog simultaneously via a single massive tensor multiplication step.
* **The Analogy:** Instead of moving one passenger at a time in a sports car, you put 32 passengers on a bus. It takes slightly longer to load the bus, but the total time to move everyone is drastically reduced.

### `n_recs = 10`
* **The Problem:** The engine calculates scores for every item in the catalog. Returning thousands of scores per user to the front-end application would crash the client's device and waste bandwidth.
* **The Solution & Math:** This is the absolute truncation limit for the final API response. After all sorting, diversity, and quota logic is complete, only the top 10 items are yielded.
* **The Analogy:** The chef cooks a massive buffet of 1,000 dishes, but only serves you a final plate with the 10 best bites.

---

## 2. Candidate Gating Mechanisms
These run *before* complex scoring to optimize performance and prevent AI hallucination.

### `relevance_percentile = 10`
* **The Problem:** When an AI scores a catalog, some items will inevitably score near zero, representing a high mathematical certainty of user disinterest. Allowing these into the diversity loops wastes CPU and risks random hallucinations.
* **The Solution & Math:** The system drops the bottom 10% of the active catalog globally based on raw AI prediction scores. 
* **The Analogy:** A university throwing out the worst 10% of applications (those with zero qualifications) before passing the rest to human reviewers.

### `candidate_limit = None`
* **The Problem:** Hard caps (like `200`) cause "cold-start starvation" because new/niche items naturally score low initially and are truncated before the stochastic explorer can ever see them.
* **The Solution & Math:** By setting this to `None`, the slicing logic `[:None]` allows the *entire* eligible catalog to pass through to the Thompson Sampling matrix.
* **The Analogy:** Rather than only interviewing the top 200 candidates and missing a hidden genius at rank 201, you allow the entire remaining pool (after the 10% bottom-drop) to be explored.

---

## 3. Promotion Labels
These handle operator-injected business campaigns.

### `enable_promotion = False`
* **The Problem:** Sometimes the business wants pure, organic AI recommendations with zero corporate interference. Other times, marketing demands specific items be pushed.
* **The Solution:** A master kill-switch for all campaign logic. If false, the Promotion quota is re-routed to Personalization.

### `promo_threshold_percentile = 93`
* **The Problem:** Even if marketing demands an item be promoted, you don't want to show it to a user if it's completely irrelevant to them.
* **The Solution & Math:** For an item to qualify for a Promotion slot for a specific user, its AI score must still sit in the 93rd percentile (the top 7%) of that user's specific strategic scoring distribution.
* **The Analogy:** Marketing wants to push a new dog food. The system will push it, but *only* to the top 7% of users who have actually shown some interest in dogs, preventing cat-owners from being spammed.

---

## 4. Personalization & Curation Labels
These parameters dynamically label items to fill the 70/20/10 quotas.

### `perso_label_percentile = 80`
* **The Problem:** Static rules like *"If score > 0.8, it's Personalization"* break when a catalog grows or model confidence drifts.
* **The Solution & Math:** The system dynamically calculates the 80th percentile of the current user's scores. Only items above this dynamic line can fill a Personalization slot.
* **The Analogy:** Grading on a curve. No matter how hard the test is, the top 20% of the class always gets an "A".

### `curation_discovery_percentile = 60`
* **The Problem:** Discovery/Curation slots shouldn't just be random garbage.
* **The Solution & Math:** Items failing the Personalization test must still pass the 60th percentile (top 40%) of general AI scores to be labeled Curation.
* **The Analogy:** Even if an item isn't perfectly personalized to you, it must still be a generally high-quality, popular item to make it into your discovery feed.

### `perso_min_threshold = 0.20`
* **The Problem:** If a user has a highly erratic history, their top 20% might still be mathematically terrible (e.g., scores of 0.15).
* **The Solution & Math:** An absolute mathematical floor. An item cannot be labeled Personalization if its raw score is < 0.20.
* **The Analogy:** Even when grading on a curve, you can't get an "A" if you only answered 1 out of 100 questions correctly.

### `perso_match_multiplier = 1.5`
* **The Problem:** Recommenders often feel "laggy" and don't respond to what the user is clicking on right *now*.
* **The Solution & Math:** If a candidate item shares a semantic theme with the user's active profile, its final score is instantly multiplied by 1.5x (capped at 1.0).
* **The Analogy:** A serendipity booster. If you start clicking Action movies, the system instantly aggressively floods your feed with Action movies in the very next batch.

### `past_items_window = 10`
* **The Problem:** The API JSON response needs to explain *why* an item was recommended without returning a list of 5,000 past clicks.
* **The Solution & Math:** The mathematical theme extraction uses the *full history*, but the text explanation string truncates to naming only the last 10 items.
* **The Analogy:** Summarizing a 500-page biography into a 1-paragraph blurb for the back cover.

---

## 5. Temporal User Profiling
These parameters control how the AI understands the user's shifting interests over time.

### `top_themes = 5`
* **The Problem:** Users accidentally click on things. Tracking every single accidental theme dilutes their profile.
* **The Solution & Math:** The system ranks all extracted themes but only retains the top 5 highest-weighted themes for real-time personalization matching.
* **The Analogy:** Noise reduction. Ignoring the one time you clicked on "Polka Music" by accident and focusing on the 5 genres you actually listen to daily.

### `recency_decay = 0.1`
* **The Problem:** Traditional systems weigh an interaction from 5 years ago the exact same as an interaction from 5 seconds ago.
* **The Solution & Math:** Applies an exponential decay function $e^{-0.1 \times \text{age}}$ to the user's history array. Recent items approach a weight of 1.0; old items mathematically vanish toward 0.
* **The Analogy:** Modeling human memory. You vividly remember what you ate yesterday, but have forgotten what you ate 3 years ago.

---

## 6. Adaptive Inference Weighting (The Continuum)
The recommendation pipeline alters its mathematical equation based on data density.

### `cold_start_cutoff = 5.0`
* **The Problem:** New users have no data. Trying to personalize their feed results in random guessing.
* **The Solution & Math:** Users with <5 interactions are bucketed into `cold_weights`: `(65% Curation, 15% Personalization)`. We rely heavily on general quality rather than guessing their specific tastes.
* **The Analogy:** Serving the most universally loved dishes (pizza, burgers) to a stranger at a restaurant until you learn what they actually like.

### `moderate_cutoff = 20.0`
* **The Problem:** Power users get annoyed by generic, popular recommendations. They want niche content.
* **The Solution & Math:** Users with >20 interactions are bucketed into `dense_weights`: `(15% Curation, 70% Personalization)`. We confidently blast them with hyper-niche personalized content and turn generic Curation off.
* **The Analogy:** A bartender who knows exactly what you drink the moment you walk in the door.

---

## 7. Stochastic Exposure Regulation (Thompson Sampling)
The Multi-Armed Bandit architecture that replaces hard exposure caps.

### `thompson_alpha_scale = 50.0`
* **The Problem:** Hard exposure caps (e.g. "stop showing after 100 views") are arbitrary, don't scale, and don't help invisible "cold" items get discovered.
* **The Solution & Math:** Tunes the Explore vs. Exploit balance of the Beta distribution $Beta(\alpha, \beta)$. The AI quality score drives $\alpha$ (exploitation), while recommendation counts drive $\beta$ (suppression). A scale of 50.0 heavily trusts the AI, but allows enough variance for new items ($\beta=1$) to randomly roll a high multiplier and get explored.
* **The Analogy:** A casino slot machine. The system mostly plays the machines it *knows* pay out (Exploitation), but occasionally drops a coin into a brand-new, unplayed machine (Exploration) just to see if it's the next big winner.

### `campaign_exposure_cap = 0.60`
* **The Problem:** Human marketing operators can artificially inject bad items into the feed.
* **The Solution & Math:** A hard, deterministic override. No operator-promoted item can ever exceed 60% of the total user feeds in a given batch.
* **The Analogy:** A business safety net preventing a human error from spamming 100% of the userbase.

### `cold_start_min_users = 50`
* **The Problem:** In a tiny batch of 5 users, showing an item to 3 users triggers the 60% cap, but 5 users is a statistically meaningless sample size.
* **The Solution & Math:** The 60% cap is completely ignored until the batch has processed at least 50 users. 
* **The Analogy:** You can't declare a coin "rigged" after flipping it only 5 times. You need a minimum statistical sample size.

---

## 8. Theme Diversity & Jitter

### `per_theme_cap = 2`
* **The Problem:** A user likes Action movies, so the AI fills all 10 recommendation slots with Action movies, creating a highly monotonous, boring feed.
* **The Solution & Math:** A strict intra-list diversity constraint. The selector loop tracks the primary theme of every selected item and rejects any candidate whose theme has already been selected 2 times.
* **The Analogy:** A balanced diet. Even if you love steak, a chef won't serve you 10 plates of steak. They will force you to have some vegetables and dessert.

### `discovery_jitter_threshold = 0.05`
* **The Problem:** Extremely cold users have such flat, uniform AI scores that the sorting algorithm essentially breaks, returning the exact same generic items in the exact same order for everyone.
* **The Solution & Math:** If a user's maximum personalization score is `< 0.05`, the system injects a tiny amount of mathematical noise (jitter) into their final scores.
* **The Analogy:** Shaking a box of sand. It breaks up artificial staleness and ensures that even generic recommendations have slight, organic variance between different new users.

---

## 9. Autonomous Semantic Clustering Engine
The NLP engine that autonomously groups items into themes without human tagging.

### `clustering_model = 'all-MiniLM-L6-v2'`
* **The Problem:** Human genre tagging is slow, expensive, prone to error, and impossible to scale.
* **The Solution:** A fast Sentence-Transformer NLP model that reads item text descriptions and turns them into 384-dimensional semantic math vectors.

### `k_min = 5` & `k_max = 50`
* **The Problem:** You don't know exactly how many genres exist in a new dataset. 
* **The Solution & Math:** The AI tests every possible grouping size between 5 and 50 clusters, scoring them mathematically (Silhouette Score) to find the absolute perfect number of categories.

### `secondary_cluster_threshold = 0.85`
* **The Problem:** Rigid tagging is inaccurate (e.g., is *Alien* Sci-Fi or Horror?).
* **The Solution & Math:** If an item's vector distance to a secondary cluster center is at least 85% as strong as its distance to its primary cluster, the system assigns both themes to the item.
* **The Analogy:** Soft-clustering. Allowing a book to be placed on both the "Sci-Fi" and "Horror" shelves simultaneously.

### `silhouette_sample_size = 2000`
* **The Problem:** Calculating the Silhouette Score across 100,000 items requires a massive $N^2$ distance matrix, which causes Out-Of-Memory (OOM) crashes on standard machines.
* **The Solution & Math:** The cluster evaluation algorithm randomly samples 2,000 items to calculate the quality score.
* **The Analogy:** Taking a poll of 2,000 voters to accurately predict a national election, rather than asking all 300 million citizens.
