# Ranking

Ranking is a project that explores `learning to rank techniques` in the context of `artificial intelligence` to build a scalable, reliable and accurate recommender system.
The dataset used is `ml-100k`, from MovieLens, which consists of "(...) 100,000 ratings (1-5) from 943 users on 1682 movies (...)".

#### :popcorn: App
Movie recommendation app that suggests 3 personalized movies for each user.
Easily accessible via a web API, deployed with FastAPI, Docker, and Google Cloud.


link > https://spiffy-dragon-2cdc30.netlify.app/

#### :test_tube: Work
Inspired by [modern industrial recommender systems](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf), this recommeder has a multi-stage architecture - **Candidate Generation** and **Ranking**.

- The 1st stage applies lightweight models to narrow down the vast space of possible movie recommendations to a manageable subset of candidates for each user. This step leverages negative sampling through collaborative filtering methods such as SVD and Co-Clustering to efficiently generate candidate sets. The goal is to reduce the search space while preserving as many relevant items as possible.
- In the 2nd stage, a predictive model ranks the candidate movies based on estimated relevance scores for each user. The ranking process is trained to optimize the Normalized Discounted Cumulative Gain (NDCG) metric, which emphasizes the placement of relevant items higher in the ranked list.

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

| Algorithm        | NDCG@5 |
|------------------|----------|
| Baseline | 0.917 |
| **XGBRanker** :trophy: | 0.954 |

#### :rocket: Deployment

#### :hourglass_flowing_sand: Future Work
- [Modeling] Add content-based candidates & try other rankers
- [Implicit feedback] Binarize ratings (e.g. rating>=4)
- [Scalability] Test ml-1m datset

#### :handshake: References
- [MovieLens Dataset](https://grouplens.org/datasets/movielens/100k/)
- [Multi-stage Architecture](https://medium.com/nvidia-merlin/recommender-systems-not-just-recommender-models-485c161c755e)
