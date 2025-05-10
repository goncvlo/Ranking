# Ranking

Ranking is a project that explores `learning to rank techniques` in the context of `artificial intelligence` to build a scalable, reliable and accurate recommender system.
The dataset used is `ml-100k`, from MovieLens, which consists of "(...) 100,000 ratings (1-5) from 943 users on 1682 movies (...)".

#### :popcorn: App
link > https://spiffy-dragon-2cdc30.netlify.app/

#### :test_tube: Work
Add some text here!

<p align="center">
  <img src="https://github.com/user-attachments/assets/37317043-fefb-4ecb-b791-3ef1641eea15" />
</p>
<p align="center"><em>Figure 1:</em> Distribution of ratings across items and Long tail effect</p>

#### :chart_with_upwards_trend: Baseline model

   - For each user, the system retrieved the top-10 most popular items within their favorite genre (based on global popularity).
   - Since users may not have seen all of these items, it was selected only users who had interacted with at least 5 of these top-10 items. The 5 items were again chosen by their popularity.
   - Using the ground truth and baseline recommendations, the evaluation metrics were computed.
   - Then, the trained ranker model was used to predict the ordering of items and compared its performance against the baseline.

*Note:*
This evaluation focuses only on how well the model ranks the items relative to each other (measured by ranking metrics like NDCG). It does not capture other aspects like diversity (novelty) or coverage (how repetitive or varied the recommendations are).

NDCG@5 results:
- Baseline: 91.7%
- Ranker: 94.8%

#### :rocket: Deployment

#### :hourglass_flowing_sand: Future Work
- [Candidate retrieval] Add content-based recommendations to mitigate cold-start problem
- [Implicit feedback conversion] Binarize ratings (e.g. rating>=4)
- [Scalability] Test ml-1m datset

#### :handshake: References
- [MovieLens Dataset](https://grouplens.org/datasets/movielens/100k/)
