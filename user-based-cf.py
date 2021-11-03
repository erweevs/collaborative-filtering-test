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
    'user_based': True
}

# fit the model
model = KNNBasic(sim_options= simOptions)
model.fit(testData)
similarityMatrix = model.compute_similarities()

# get top N similar Users to the test user
testUserId = '85'
N = 10

testUserInnerUID = testData.to_inner_uid(testUserId)
similarityRow = similarityMatrix[testUserInnerUID]

# loop over the similar users, and find their ID and Similarity Score
similarUsers = []
for innerId, simScore in enumerate(similarityRow):
    # exclude the test user from its own similar users
    if innerId != testUserInnerUID:
        similarUsers.append((innerId, simScore))

# get the K Nearest Neighbors
# builds an array of objects that has the users inner id and the score, ie:
# [(10, 1.0), (11, 1.0), (13, 1.0), (24, 1.0), (36, 1.0), (44, 1.0), (45, 1.0), (51, 1.0), (53, 1.0), (61, 1.0)]
knn = heapq.nlargest(N, similarUsers, key= lambda x: x[1])

# TODO: experiment with only including users with a Cosine Similarity score of 0.95 or higher => play around with the score 
# to find best value
# knn = []
# for rating in similarUsers:
#     if rating[1] > 0.95: knn.append(rating)

# loop over the N most similar users to get the items they rated
recommendationCanidates = defaultdict(float)
for similarUser in knn:
    userInnerId = similarUser[0]
    userSimilarityScore = similarUser[1]

    # this will return an array of objects that contains the movie id, and the users rating
    userRatings = testData.ur[userInnerId]

    for userRating in userRatings:
        movieId = userRating[0]
        rating = userRating[1]

        # calculate the weighted Similarity
        ratingSum = (rating / 5) * userSimilarityScore

        recommendationCanidates[movieId] += ratingSum

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