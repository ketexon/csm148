import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.DataFrame({
    "food": [
        8,8,8,7,
        6,6,5,7,
        6,8,7
    ],
    "ambience": [
        8,9,8,9,8,7,5,9,8,7,9,
    ],
    "service": [
        8,8,7,7,7,5,5,6,7,7,7
    ],
    "rating": [
        8,8,8,7,
        7,5,5,7,
        6,8,7
    ],
})

var_food = df["food"].var()
var_rating = df["rating"].var()
r = df.corr()["food"]["rating"]

food = df["food"][:10]
rating = df["rating"][:10]
food_mean = food.mean()
rating_mean = rating.mean()
print(sum([(f - food_mean)*(r - rating_mean) for (f, r) in zip(food, rating)]))
print(sum([(f - food_mean)**2 for f in food]))
print(food.mean(), rating.mean())

sse = sum([(row["food"] - row["rating"])**2 for i,row in df[:10].iterrows()])
sst = sum([(rating_mean - row["rating"])**2 for i,row in df[:10].iterrows()])
r2 = (
    1 - sse/sst
)
print(sse)
print(sst)
print(r2)

# m = r*(var_rating)**0.5/(var_food)**0.5
# b = df["rating"].mean() - m * df["food"].mean()
# print(m)
# print(b)

# lr = LinearRegression()
# lr.fit(df[["food"]], df["rating"])
# print(lr.coef_)
# print(lr.intercept_)
