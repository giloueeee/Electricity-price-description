import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error

"""
Pour ce projet, nous utiliserons XGboost pour prédire le prix de l'électricité en New South Wales(Australie).

La base de donnée d'entrainement étant conséquente (+35 000 lignes), nous la diviserons simplement 
 en 2 sous-parties :
- une partie pour l'entrainement (80%)
- une partie pour la validation (20%)

Les données comportent des valeurs numériques et catégoriques.
Nous encoderons les valeurs catégoriques en utilisant l'encodage ordinal.

Nous chercherons la meilleure combinaison de paramètres pour le modèle en nous basant sur son score mse
et synthétiserons les résultats dans un tableau récapitulatif(heatmap).

"""


raw_train_set=pd.read_csv('data/train_set.csv', index_col='Id')
raw_test_set=pd.read_csv('data/test_set.csv', index_col='Id')

numerical_Features=['period', 'nswprice', 'nswdemand', 'transfer', 'vicprice', 'vicdemand']
raw_categorical_Features=['day', 'class']

ordinal_encoder=OrdinalEncoder()

train_encoded_categorical_Features=ordinal_encoder.fit_transform(raw_train_set[raw_categorical_Features])
test_encoded_categorical_Features=ordinal_encoder.transform(raw_test_set[raw_categorical_Features])

train_encoded_categorical_Features=pd.DataFrame(train_encoded_categorical_Features, columns=raw_categorical_Features, index=raw_train_set.index)
test_encoded_categorical_Features=pd.DataFrame(test_encoded_categorical_Features, columns=raw_categorical_Features, index=raw_test_set.index)
train_numerical_Features=raw_train_set[numerical_Features]
test_numerical_Features=raw_test_set[numerical_Features]

print(type(train_encoded_categorical_Features))


print(type(train_numerical_Features))

train_set=pd.concat([train_numerical_Features, train_encoded_categorical_Features], axis=1)
test_set=pd.concat([test_numerical_Features, test_encoded_categorical_Features], axis=1)
y=train_set.nswprice

print("train_set:", train_set)

X=train_set.drop('nswprice', axis=1)
test_set=test_numerical_Features.drop('nswprice', axis=1)

print(test_set)
print(X)

X=X.select_dtypes(exclude=['object'])
test_set_processed = test_set.select_dtypes(exclude=['object'])
test_set_processed = pd.DataFrame(test_set_processed, columns=test_set_processed.columns, index=test_set_processed.index)
test_set_processed = test_set_processed.dropna()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)


def get_score (trees, depths, X, y):
    model=XGBRegressor(n_estimators=trees, max_depth=depths)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    predictions = model.predict(X_val)
    score = mean_squared_error(predictions, y_val)
    MAE= mean_absolute_error(predictions, y_val)
    return score, MAE

scores = []
MAEs = []
list_depths = []
list_trees = []
depth_list = (3, 4, 5, 6, 7, 8, 9, 10)
n_estimatorslist = list(range(100, 1000, 50))

for depth in depth_list:
    for trees in n_estimatorslist:
        list_depths.append(depth)
        list_trees.append(trees)
        score, MAE = get_score(trees, depth, X, y)
        scores.append(score *10000)
        MAEs.append(MAE)
        print("Depth :", depth, """ /""", max(depth_list)) 
        print("Trees :", trees, """ /""", max(n_estimatorslist)) 
        print("Score :", scores[-1])
        print("MAE :", MAEs[-1])


min_score=min(scores)
position_min_score=scores.index(min_score)
print("Best score:", min_score,"*10⁴ with depth:", list_depths[position_min_score], "and trees:", list_trees[position_min_score], "and MAE:", MAEs[position_min_score])

best_depth=list_depths[position_min_score]
best_trees=list_trees[position_min_score]


print(f"Best depth: {best_depth} at position {position_min_score}")
print(f"Best trees: {best_trees} at position {position_min_score}")

model=XGBRegressor(n_estimators=best_trees, max_depth=best_depth, random_state=0)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
predictions=model.predict(test_set_processed)
predictions = pd.DataFrame(predictions, columns=['nswprice'], index=test_set_processed.index)

output=pd.DataFrame({"Id": test_set_processed.index, "NswPrice": predictions['nswprice']})
output.to_csv('output_test.csv', index=False)

heatmap_dataset=[depth_list], [n_estimatorslist], [scores]
reshaped_dataset=pd.DataFrame({
    'depths': list_depths,
    'trees': list_trees,
    'scores': scores
})

heatmap_dataframe = reshaped_dataset.pivot(index='trees', columns='depths', values='scores')

plt.figure(figsize=(10, 5))
plt.title("MSE*10⁴ evolution with depths and trees")
sns.heatmap(data=heatmap_dataframe, annot=True, fmt=".3f")
plt.xlabel('Max depths')
plt.ylabel('Number of Trees')
plt.show()
plt.savefig('heatmap.png')