from MovieLens import MovieLens
from surprise import KNNBasic
import heapq
from collections import defaultdict
from operator import itemgetter

# load the data
ml = MovieLens()
data = ml.loadMovieLensLatestSmall()

# NOTE: since we wont be measuring accuracy, there is no need for a train/ test split.
# build the test set
testData = data.build_full_trainset()

# construct the similarity options
simOptions = {
    'name': 'cosine',
    'user_based': False
}

# fit the model
model = KNNBasic(sim_options= simOptions)
model.fit(testData)
similarityMatrix = model.compute_similarities()

# get top N items the user rated
testUserId = '85'
N = 10

testUserInnerUID = testData.to_inner_uid(testUserId)
userRatings = testData.ur[testUserInnerUID]

# build an array of objects that contains the movie id, and the rating the user gave it
# [(650, 5.0), (20, 5.0), (27, 5.0), (4206, 5.0), (387, 5.0), (49, 5.0), (423, 5.0), (99, 5.0), (145, 5.0), (55, 5.0)]
knn = heapq.nlargest(N, userRatings, key= lambda x: x[1])

# TODO: experiment to only include rating higher than 4 stars
# knn = []
# for rating in userRatings:
#     if rating[1] > 4.0: knn.append(rating)

# get similar items to the ones the user has rated, wighed by rating
recommendationCanidates = defaultdict(float)
for itemId, rating in knn:
    similarityRow = similarityMatrix[itemId]

    for innerId, ratingScore in enumerate(similarityRow):
        weightedRating = ratingScore * (rating / 5.0)

        recommendationCanidates[innerId] += weightedRating

# build a dictionary of the items the user has already rated, ie. already viewed the content
viewed = {}
for itemId, rating in testData.ur[testUserInnerUID]:
    viewed[itemId] = 1

# get the top rated items for the similar users
recommendations = []
itemsAdded = 0
for itemId, ratingSum in sorted(recommendationCanidates.items(), key=itemgetter(1), reverse=True):
    # only recommend items the user has not yet seen
    if not itemId in viewed:
        # get the movie id in the form that the data set requires
        movieId = testData.to_raw_iid(itemId)

        movieName = ml.getMovieName(int(movieId))
        recommendations.append(movieName)

        itemsAdded += 1
        if itemsAdded > N: break

print(recommendations)