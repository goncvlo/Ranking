# Ranking

[![Netlify Status](https://api.netlify.com/api/v1/badges/8c2a81b2-1b98-46ec-aefe-7e9038af5023/deploy-status)](https://app.netlify.com/sites/ranking-recsys-ui/deploys)
[![CI/CD Pipeline for Ranking Model App](https://github.com/6oncvlo/Ranking/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/6oncvlo/Ranking/actions/workflows/ci_cd.yml)

Ranking is a project that explores `learning to rank techniques` in the context of `artificial intelligence` to build a scalable and personalized recommender system.
The dataset used is `ml-100k`, from MovieLens, which consists of "(...) 100,000 ratings (1-5) from 943 users on 1682 movies (...)".

#### :popcorn: App
Movie recommendation app that suggests 3 personalized movies for each user.
Easily accessible via a web API, deployed with FastAPI, Docker, and Google Cloud.

**Visit https://ranking-recsys-ui.netlify.app/**


https://github.com/user-attachments/assets/6a8eccad-a60a-41ef-ae79-1d8babc2f60c



#### :test_tube: Work
Inspired by [modern industrial recommender systems](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf), this recommeder has a multi-stage architecture - **Candidate Generation** and **Ranking**.

- The 1st stage applies lightweight models to narrow down the vast space of possible movie recommendations to a manageable subset of candidates for each user. This step leverages negative sampling through collaborative filtering methods such as SVD and Co-Clustering to efficiently generate candidate sets.
- In the 2nd stage, a predictive model ranks the candidate movies based on estimated relevance scores for each user. The ranking process is trained to optimize the Normalized Discounted Cumulative Gain (NDCG) metric, which emphasizes the placement of relevant items higher in the ranked list.

<p align="center">
  <img src="https://github.com/user-attachments/assets/37317043-fefb-4ecb-b791-3ef1641eea15" />
</p>
<p align="center"><em>Figure 1:</em> Distribution of ratings across items and Long tail effect</p>

#### :chart_with_upwards_trend: Baseline model

**1. Candidate Generation**

To evaluate the 1st phase, a baseline is set to compare the candidates from each heuristic/model (e.g. hottest items, collaborative filtering, ...). The baseline is based on item popularity - number of times an item was rated. Here, the goal is to evaluate if all the relevant items were found and if the candidates are actually relevant.

   - For each user, it is retrieved the top-10 most popular items which weren't rated by the user.
   - Evaluate recall, precision and hit rate on validation (VS) and test (TS) sets.

The table bellow summarises the performance metrics for the different methods.

| Algorithm        | Recall@10 (%) VS | Precision@10 (%) VS |  HitRate@10 (%) VS | Recall@10 (%) TS |  Precision@10 (%) TS | HitRate@10 (%) TS |
|------------------|----------|----------|----------|----------|----------|----------|
| Baseline | 8.20 | 4.10 | 32.13 | 7.57 | 3.78 | 30.43 |
| KNNWithMeans | 1.35 | 0.67 | 6.46 | 1.44 | 0.72 | 6.78 |
| SVD | 0.44 | 0.22 | 2.22 | 0.27 | 0.13 | 1.37 |
| CoClustering | 0.36 | 0.18 | 1.59 | 0.25  | 0.12 | 1.27 |
| Hottest Items | 1.39 | 0.69 | 6.25 | 1.35 | 0.67 | 5.93 |
| CoVisited Weighted | 9.33 | 4.66 | 34.04 | 9.28 | 4.64 | 33.82 |
| CoVisited Directional | 12.53 | 6.26 | 43.37 | 10.71 | 5.35 | 38.70 |

Having now different methods to pick candidates from, a pool of candidates must be created for the ranking phase. For instance, if 10 candidates are picked from 5 different methods, it is obtained a total of 50 candidates for each user - which represents 3% of the total num. of items. In this case, the baseline is the top-50 most popular items which weren't rated by the user.

| Algorithm        | Recall@50 (%) VS | Precision@50 (%) VS | HitRate@50 (%) VS | Recall@50 (%) TS | Precision@50 (%) TS | HitRate@50 (%) TS |
|------------------|----------|----------|----------|----------|----------|----------|
| Baseline | 22.43 | 2.24 | 63.73 | 21.93 | 2.19 | 60.76 |
| Pool #1 |   |   |   |   |   |   |
| Pool #2 |   |   |   |   |   |   |

**2. Ranking**

This evaluation focuses only on how well the model ranks the items relative to each other. Similarly to 1st phase, the baseline choosen is popularity based - i.e., within the candidate set, rank by item popularity.

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
