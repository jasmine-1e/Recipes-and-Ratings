# Behind The Rating: An Analysis of Recipe Ratings and Reviews

**By**: Jasmine Le

---

## Introduction

As a person who bakes frequently, I spend a lot of time sifting through recipe ratings and reviews to look for the best recipes that suit my goals, even if they have lower ratings. 

Discovering what other factors go into how a recipe is rated is significant because **average rating is usually the first thing people see when they browse for recipes**, and is supposed to be a proxy for how "good" a recipe is, even when it might not be an accurate representation of the quality of the recipe.

I find this method to be somewhat flawed, as it can sometimes be more an evaluation of the preferences of the individuals using the recipe than the recipe itself. Low ratings from outside factors, such as individual choices or personal taste, can impact the average rating of an otherwise acceptable recipe not insignificantly. So, generally, **I would like to see how the average rating of recipes is affected by these factors**.

This indicates that ratings may not always be a good indicator of what the "best" recipe actually is. In practice, it means I'll often read all the reviews a recipe has and inspect the recipe itself to see if the rating it's incurred has been the result of a fair evaluation (as in if it's actually a good or bad recipe), or if it's been rated unfairly due to reasons not attributable to the recipe itself. Though I've been successful in choosing good recipes that do exactly what I want thus far, it is a heuristic process, and so I'd like to see if there's actually a way to do this utilizing data science tools.

**Central Question**: Is recipe healthiness (determined by nutritional content) a significant factor in determining average recipe rating?

As such, I'll be focusing on the relationship between health and recipe rating. Since this is a much more mainstream concern now, with lots of people tweaking recipes to make them healthier, I'd like to see how significant indicators of recipe health are towards determining rating.

### Dataset Overview

This analysis uses the **Recipes and Ratings dataset**, which contains recipes and user reviews from Food.com. The dataset consists of:

- **234,429 rows** (after merging recipes with their reviews)
- Key columns relevant to this analysis:
  - `name`: Recipe name
  - `id`: Unique recipe identifier
  - `minutes`: Preparation time
  - `nutrition`: List containing nutritional information [calories, total fat PDV, sugar PDV, sodium PDV, protein PDV, saturated fat PDV, carbohydrates PDV]
  - `n_steps`: Number of steps in the recipe
  - `n_ingredients`: Number of ingredients
  - `rating`: Individual user rating (1-5 stars)
  - `avg_rating`: Average rating across all reviews for the recipe
  - `review`: Text review from users

The nutritional values (except calories) are represented as Percentage of Daily Value (PDV), which standardizes the measurements relative to recommended daily intake.

---

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning Process

The following cleaning steps were performed to prepare the data for analysis:

1. **Merged datasets**: Combined the recipes dataset with the interactions (ratings/reviews) dataset using a left merge on recipe `id`

2. **Replaced zero ratings with NaN**: Ratings of 0 were considered invalid (the rating scale is 1-5) and replaced with NaN to avoid skewing the average

3. **Calculated average ratings**: Computed the average rating for each recipe across all its reviews

4. **Parsed nutrition information**: The nutrition column originally contained a string representation of a list. I converted this to an actual list and extracted individual nutritional components into separate columns:
   - `calories`: Total calories
   - `total_fat`: Total fat (PDV)
   - `sugar`: Sugar content (PDV)
   - `sodium`: Sodium content (PDV)
   - `protein`: Protein content (PDV)
   - `saturated_fat`: Saturated fat (PDV)
   - `carbohydrates`: Carbohydrates (PDV)

5. **Created health classification columns**: For hypothesis testing purposes, I created binary classifications (high/low) for each nutritional factor:
   - Used 20% PDV as the threshold for sugar, carbohydrates, and protein (based on FDA guidelines)
   - Used 400 calories as the threshold for calorie content (approximate single-serving benchmark)

6. **Handled missing values**: Examined patterns of missingness in the `rating` and `review` columns

### Cleaned Data (First 5 Rows)

| name                                 |     id |   minutes |   n_steps |   n_ingredients |   avg_rating |   calories |   sugar |   carbohydrates |   protein |
|:-------------------------------------|-------:|----------:|----------:|----------------:|-------------:|-----------:|--------:|----------------:|----------:|
| 1 brownies in the world    best ever | 333281 |        40 |        10 |               9 |            4 |      138.4 |      50 |               6 |         3 |
| 1 in canada chocolate chip cookies   | 453467 |        45 |        12 |              11 |            5 |      595.1 |     211 |              22 |        13 |
| 412 broccoli casserole               | 306168 |        40 |         6 |               9 |            5 |      194.8 |       6 |              20 |        22 |
| millionaire pound cake               | 286009 |       120 |         7 |               7 |            5 |      878.3 |     326 |              20 |        20 |
| 2000 meatloaf                        | 475785 |        90 |        17 |              13 |            5 |      267   |      12 |              29 |        29 |

### Univariate Analysis

To understand the distribution of key variables, I examined individual columns:

**Average Rating Distribution**

<iframe
  src="assets/rating-distribution.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The distribution of average ratings is heavily skewed toward higher values, with the majority of recipes receiving ratings between 4 and 5 stars. This suggests that users tend to rate recipes favorably overall, or that only successful recipes receive reviews.

**Sugar Content Distribution**

<iframe
  src="assets/sugar-distribution.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The sugar content (PDV) distribution is right-skewed with a long tail, indicating that while most recipes have moderate sugar content, there are many dessert recipes with very high sugar levels (200%+ PDV).

### Bivariate Analysis

**Relationship Between Carbohydrates and Average Rating**

<iframe
  src="assets/carbs-rating-scatter.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

This scatter plot shows the relationship between carbohydrate content and average rating. There doesn't appear to be a strong linear relationship, though we can observe that recipes with moderate carbohydrate content (20-60% PDV) seem to cluster around higher ratings.

**Calories vs. Rating by Recipe Category**

<iframe
  src="assets/calories-rating-box.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The box plots comparing high-calorie vs. low-calorie recipes show that both groups have similar median ratings around 4.5-5 stars, suggesting that calorie content alone may not significantly impact ratings.

### Interesting Aggregates

The following table shows the mean nutritional content and average rating grouped by the number of steps in a recipe:

| n_steps |   avg_rating |   calories |   sugar |   carbohydrates |   protein |
|--------:|-------------:|-----------:|--------:|----------------:|----------:|
|       1 |     4.65574  |    286.397 | 67.0537 |         10.2459 |   15.2541 |
|       2 |     4.67011  |    312.794 | 75.4767 |         11.1078 |   16.9155 |
|       3 |     4.68063  |    332.948 | 82.6144 |         11.8419 |   19.0774 |
|       4 |     4.67843  |    353.429 | 91.4219 |         12.9013 |   21.1992 |
|       5 |     4.67472  |    373.604 | 97.5086 |         13.6864 |   23.0709 |
|       6 |     4.67111  |    392.467 | 102.695 |         14.4342 |   24.7955 |
|       7 |     4.66876  |    407.856 | 106.575 |         15.0396 |   26.1729 |
|       8 |     4.66497  |    423.083 | 109.779 |         15.6175 |   27.4858 |

This table reveals that as recipes become more complex (more steps), they tend to have higher nutritional values across all categories, but the average rating remains relatively stable around 4.65-4.68 stars. This suggests that recipe complexity and nutritional content don't strongly correlate with ratings.

---

## Assessment of Missingness

### NMAR Analysis

I believe that the **`review` column is likely NMAR (Not Missing At Random)**. 

The missingness of reviews is likely related to the content of the review itself - that is, users who have nothing particular to say (neither strong praise nor complaints) may choose not to leave a review at all. This is a case where the missingness mechanism depends on the value that would have been recorded. Someone who found a recipe perfectly adequate but unremarkable might rate it but skip writing a review, whereas someone with strong feelings (very positive or very negative) would be more motivated to write detailed feedback.

To make this MAR instead of NMAR, we would need additional data such as:
- User engagement metrics (time spent on page, whether they saved/bookmarked the recipe)
- User profile information (frequency of reviewing, typical review length)
- Notification/prompt data (whether the user was asked to leave a review)

This additional information could explain the missingness through observable factors rather than the unobserved review content itself.

### Missingness Dependency

I analyzed the missingness of the `rating` column to determine whether it depends on other columns in the dataset.

**Column that rating missingness DOES depend on: `minutes` (preparation time)**

I performed a permutation test with:
- **Test statistic**: Absolute difference in mean preparation time between recipes with missing ratings vs. non-missing ratings
- **Null hypothesis**: The missingness of rating does not depend on preparation time; any observed difference is due to random chance
- **Alternative hypothesis**: The missingness of rating does depend on preparation time
- **Significance level**: 0.01
- **Number of permutations**: 10,000

<iframe
  src="assets/missingness-minutes.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

**Result**: p-value < 0.01 (observed difference was significantly larger than simulated differences)

**Interpretation**: We reject the null hypothesis and conclude that the missingness of ratings likely depends on preparation time. Recipes with extremely long or short preparation times may receive fewer ratings, possibly because users are less likely to attempt them or complete them.

**Column that rating missingness does NOT depend on: `n_ingredients` (number of ingredients)**

I performed a similar permutation test with:
- **Test statistic**: Absolute difference in mean number of ingredients
- **Same hypotheses structure as above**
- **Significance level**: 0.01
- **Number of permutations**: 10,000

<iframe
  src="assets/missingness-ingredients.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

**Result**: p-value = 0.34 (observed difference was well within the range of simulated differences)

**Interpretation**: We fail to reject the null hypothesis. The missingness of ratings does not appear to depend on the number of ingredients in a recipe. This suggests that recipe complexity in terms of ingredient count does not affect whether users choose to rate the recipe.

---

## Hypothesis Testing

**Research Question**: Does recipe healthiness significantly affect average rating?

Since "healthiness" can be defined in multiple ways depending on individual dietary goals and needs, I tested each nutritional factor independently rather than combining them into a single "healthiness score."

### Hypotheses

**Null Hypothesis**: Recipe healthiness (determined by sugar content, carbohydrate content, protein, and calorie count) is NOT the most significant factor in determining average recipe rating. Any observed differences in ratings between healthy and less healthy recipes are due to random chance.

**Alternative Hypothesis**: Recipe healthiness IS the most significant factor in determining average recipe rating. Recipes with healthier nutritional profiles have systematically different ratings than those with less healthy profiles.

### Methodology

For each nutritional factor (carbohydrates, calories, sugar, protein), I performed a separate permutation test:

- **Test Statistic**: Absolute difference in average rating between high-content and low-content recipes
  - For PDV measures (carbs, sugar, protein): High content = PDV ≥ 20%, Low content = PDV < 20%
  - For calories: High content ≥ 400 calories, Low content < 400 calories

- **Significance Level**: 0.01
- **Number of Permutations**: 10,000 for each test

### Results

| Nutritional Factor | P-value | Observed Difference | Conclusion |
|-------------------|---------|---------------------|------------|
| Carbohydrates     | 0.16176 | ~0.015 stars       | Fail to reject null |
| Calories          | 0.16305 | ~0.014 stars       | Fail to reject null |
| Sugar             | 0.21847 | ~0.012 stars       | Fail to reject null |
| Protein           | 0.63893 | ~0.005 stars       | Fail to reject null |

<iframe
  src="assets/hypothesis-carbs.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

### Interpretation and Discussion

**At a significance level of 0.01, I failed to reject the null hypothesis for all nutritional factors.** This means there is insufficient evidence to conclude that recipe healthiness (as measured by any single nutritional component) is a significant determinant of average rating.

However, several interesting patterns emerged:

1. **Carbohydrate content consistently showed the smallest p-values** across multiple test runs (ranging from 0.05 to 0.16), though still not statistically significant. This suggests that carbohydrate content may have a weak relationship with ratings that could be worth investigating further with a larger sample or different methodology.

2. **Protein had the largest p-values** (0.35+), indicating virtually no relationship between protein content and ratings.

3. **Sugar p-values remained stable** around 0.2+, showing no strong effect.

4. **Calorie measurements were inconsistent** because the dataset doesn't standardize for serving sizes. Some recipes provide nutritional information for entire batches (e.g., a whole cake) while others show single servings, making calorie count an unreliable metric without additional context.

**Methodological Considerations**:

- Testing factors separately allowed me to identify which specific aspects of "healthiness" might matter to users, but it also meant I couldn't detect synergistic effects if they exist
- The threshold values I chose (20% PDV, 400 calories) were based on FDA guidelines but may not align with how users actually think about "healthy" vs "unhealthy" recipes
- Running multiple permutation tests increases the chance of Type I errors; a Bonferroni correction would make the significance threshold even stricter (0.01/4 = 0.0025)

**Conclusion**: Based on this analysis, recipe healthiness does not appear to be a major factor in determining ratings. Users likely rate recipes based on taste, ease of preparation, and whether the recipe worked as expected, rather than nutritional content.

---

## Framing a Prediction Problem

**Prediction Problem**: Can we predict what types of issues or complaints users will have about a recipe based on recipe characteristics?

**Problem Type**: Multiclass classification

**Response Variable**: `issue` - the type of complaint/feedback in the review
- Why this variable? Understanding common issues can help recipe developers improve their recipes and help users identify potential problems before attempting a recipe. This is more actionable than simply predicting ratings, as it provides specific, interpretable feedback.

**Features Available at Time of Prediction**: 
At the time a recipe is published (before any reviews), we have access to:
- Recipe name
- Ingredients list
- Number of ingredients
- Preparation steps
- Number of steps
- Preparation time
- Nutritional information
- Recipe tags/categories

We would **NOT** use:
- Reviews (these come after)
- Ratings (these come after)
- User information from reviews

**Evaluation Metric**: **F1-Score (macro-averaged)**

**Why F1-Score?**
- The distribution of issue types is likely imbalanced (some complaints are more common than others)
- F1-score balances precision and recall, which is important because:
  - **Precision matters**: We don't want to falsely predict issues that don't exist (could discourage people from trying good recipes)
  - **Recall matters**: We want to catch most actual issues (so users are warned about potential problems)
- Macro-averaging treats all issue types equally, ensuring the model performs well on rare but important issues, not just common ones
- Unlike accuracy, F1-score accounts for class imbalance and gives a more realistic picture of model performance

**Alternative metrics considered**:
- Accuracy: Not suitable due to class imbalance - a model that predicts the majority class would have high accuracy but be useless
- Precision/Recall alone: Each tells only part of the story; F1 balances both
- Weighted F1: Would prioritize common issues over rare ones, but rare issues might be just as important to predict

---

## Baseline Model

### Model Description

My baseline model predicts recipe ratings (initially simplified to binary classification: high rating ≥ 4 vs. low rating < 4) using two features:

**Features**:
1. **`avg_rating`** (Quantitative): The average rating of the recipe across all reviews
2. **`n_steps`** (Quantitative): The number of preparation steps in the recipe

Both features are quantitative continuous variables, and I did not perform any encoding transformations in the baseline model - I used them in their raw form.

**Model Choice**: I initially used a simple classifier that predicts all recipes as "high rating" since the dataset is heavily skewed toward high ratings. This serves as a true baseline to compare against.

### Model Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

baseline_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])
```

### Performance

The baseline model achieved the following on the test set:
- **Accuracy**: ~85% (due to class imbalance favoring high ratings)
- **F1-Score (macro)**: ~0.42

However, examining the confusion matrix revealed that the model was essentially predicting the majority class (high ratings) for almost all instances, which explains the high accuracy but poor F1-score.

### Is This Model "Good"?

**No, this baseline model is not good** for several reasons:

1. **It doesn't meaningfully discriminate**: The model largely predicts the majority class, making it no better than a naive classifier
2. **Low F1-score**: A macro F1-score of 0.42 indicates poor performance on the minority class (low ratings)
3. **Features are not appropriate for the task**: Using `avg_rating` to predict issues is conceptually flawed because:
   - Ratings come AFTER reviews, not before
   - Using them creates data leakage
   - It doesn't help us understand recipe characteristics that lead to issues

4. **Wrong prediction target**: I realized that predicting ratings is less useful than predicting the types of issues users encounter

### Lessons Learned

This baseline model revealed the need to:
1. Reframe the prediction problem to focus on issue types rather than ratings
2. Use features that are available before reviews are written
3. Extract meaningful information from text fields (recipe names, ingredients, steps)
4. Address class imbalance more effectively

The next iteration (Final Model) addresses these issues by predicting issue types using recipe characteristics available at the time of publication.

---

## Final Model

### Problem Reframing

Based on the limitations of the baseline model, I shifted to predicting **types of issues/complaints in reviews** rather than ratings. This is more useful because:
- It provides actionable insights for recipe improvement
- It helps users anticipate potential problems
- It uses only information available before reviews are written

### Features Added

I engineered several new features to better capture recipe characteristics:

**1. Recipe Name Features (Nominal → TF-IDF + K-Means Clustering)**
- **Why**: Recipe names often indicate the type of dish (e.g., "chocolate cake", "chicken soup"), which can correlate with common issues
- **Transformation**: 
  - Applied TF-IDF vectorization to convert text to numerical features
  - Used K-Means clustering (k=5) to group similar recipe types
  - This reduces dimensionality while capturing semantic meaning
- **Data-generating process connection**: Different recipe categories (desserts vs. savory, simple vs. complex) have characteristic issues (e.g., baking issues for cakes, timing issues for meats)

**2. Ingredients Features (Nominal → TF-IDF + K-Means Clustering)**
- **Why**: Specific ingredients can predict common problems (e.g., eggs → texture issues, chocolate → burning/melting issues)
- **Transformation**: Same TF-IDF + K-Means approach as recipe names
- **Data-generating process connection**: Certain ingredients are technically challenging or have specific requirements that novice cooks might struggle with

**3. Number of Steps (Quantitative → Standardized)**
- **Why**: Recipe complexity affects likelihood of errors
- **Transformation**: StandardScaler to normalize the distribution
- **Data-generating process connection**: More steps → more opportunities for mistakes

**4. Nutritional Categories (Quantitative → One-Hot Encoded)**
- **Features**: High/low categories for calories, sugar, carbohydrates, protein
- **Why**: While not significant for ratings, nutritional content might correlate with recipe types that have characteristic issues
- **Transformation**: One-hot encoding of the binary high/low classifications
- **Data-generating process connection**: High-sugar recipes (desserts) might have different issues than high-protein recipes (meats)

### Model Architecture

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

preprocessor = ColumnTransformer(
    transformers=[
        ('numbers', StandardScaler(), ['n_steps']),
        ('ingredient', tfidf_kmeans_preprocessor, 'ingredients'),
        ('name', tfidf_kmeans_preprocessor, 'name'),
        ('calories', OneHotEncoder(), ['calories_category']),
        ('sugar', OneHotEncoder(), ['sugar_category']),
        ('carbs', OneHotEncoder(), ['carbohydrates_category']),
        ('protein', OneHotEncoder(), ['protein_category'])
    ],
    remainder='drop'
)

final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42
    ))
])
```

### Hyperparameter Tuning

I used **GridSearchCV** with 5-fold cross-validation to find optimal hyperparameters:

**Parameters Tuned**:
- `n_estimators`: [50, 100, 200] - Number of trees in the forest
- `max_depth`: [10, 20, 30, None] - Maximum depth of trees
- `min_samples_split`: [2, 5, 10] - Minimum samples required to split a node

**Why these parameters?**
- **n_estimators**: More trees generally improve performance but increase computation time
- **max_depth**: Controls model complexity and overfitting; deeper trees can capture complex patterns but may overfit
- **min_samples_split**: Higher values prevent overfitting by requiring more samples before splitting

**Best Parameters Found**:
- `n_estimators`: 100
- `max_depth`: 20
- `min_samples_split`: 5

**Selection Method**: 5-fold cross-validation with F1-score (macro) as the evaluation metric

### Performance Comparison

| Metric | Baseline Model | Final Model | Improvement |
|--------|---------------|-------------|-------------|
| Accuracy | ~0.85 | ~0.73 | -0.12 |
| F1-Score (macro) | ~0.42 | ~0.58 | +0.16 |
| Precision (macro) | ~0.45 | ~0.61 | +0.16 |
| Recall (macro) | ~0.40 | ~0.56 | +0.16 |

### Why This Is An Improvement

While accuracy decreased, **this is actually a sign of improvement** because:

1. **The baseline was gaming the metric** by predicting only the majority class, achieving high accuracy but providing no useful predictions for minority classes

2. **F1-score is the appropriate metric** for this imbalanced classification problem, and it improved significantly (+38%)

3. **The model now makes meaningful distinctions** between issue types rather than defaulting to one prediction

4. **Better balance across classes**: The macro-averaged metrics show that the model performs more consistently across all issue types, not just common ones

5. **Actionable predictions**: The model can now actually help identify potential issues before they occur

### Feature Engineering Impact

The improvements came primarily from:
1. **Text feature extraction** (TF-IDF + clustering) captured semantic information from recipe names and ingredients
2. **Nutritional categories** helped identify recipe types prone to specific issues
3. **Random Forest** naturally handles feature interactions and non-linear relationships

The combination of domain-relevant features and appropriate model selection led to a classifier that provides genuinely useful predictions for identifying potential recipe issues.

---

## Fairness Analysis

### Groups Defined

To assess whether the model performs fairly across different types of recipes, I defined two groups based on **average rating**:

- **Group X**: Recipes with lower ratings (average rating ≤ 3 stars)
- **Group Y**: Recipes with higher ratings (average rating > 3 stars)

**Why these groups?** 
- Recipes with lower ratings might have more diverse or severe issues
- There are far more highly-rated recipes than poorly-rated ones (class imbalance)
- We want to ensure the model doesn't just perform well on popular recipes while ignoring problematic ones

### Evaluation Metric

**F1-Score** (macro-averaged)

**Why F1-score instead of accuracy?**
- There is significant class imbalance: many more highly-rated recipes exist
- F1-score accounts for both precision (are predicted issues actually correct?) and recall (do we catch all actual issues?)
- Macro-averaging ensures we evaluate performance on all issue types equally, not just common ones

### Hypotheses

**Null Hypothesis**: The model is fair. Its F1-score for predicting issues in lower-rated recipes (Group X) and higher-rated recipes (Group Y) are roughly the same, and any differences are due to random chance.

**Alternative Hypothesis**: The model is unfair. Its F1-score for lower-rated recipes is significantly different from (likely lower than) its F1-score for higher-rated recipes.

### Test Setup

- **Test Statistic**: Absolute difference in F1-scores between Group X and Group Y
- **Significance Level**: α = 0.01
- **Method**: Permutation test with 1,000 iterations
  - Randomly shuffle group labels
  - Recalculate F1-scores for each group
  - Compute the difference
  - Compare to observed difference

### Results

<iframe
  src="assets/fairness-permutation.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

**Observed Difference in F1-Scores**: 0.23 (Group Y had F1 = 0.62, Group X had F1 = 0.39)

**P-value**: < 0.01

### Conclusion

**We reject the null hypothesis at the 0.01 significance level.** The model performs significantly better on highly-rated recipes than on lower-rated recipes.

**What does this mean?**

The model exhibits **performance disparity** - it is less effective at predicting issues for recipes that already have lower ratings. This is concerning because:

1. **Recipes with issues need the most help**: Lower-rated recipes are precisely the ones where predicting issues would be most valuable, yet the model performs worst on them

2. **Insufficient training data**: Lower-rated recipes are much rarer in the dataset, giving the model fewer examples to learn from

3. **Issue diversity**: Lower-rated recipes might have more varied or severe issues that are harder to predict from recipe characteristics alone

**Implications**:
- The model should be used with caution when analyzing recipes with lower ratings
- Additional data collection focused on problematic recipes could improve fairness
- Feature engineering specifically targeting common issues in lower-rated recipes might help
- Alternative approaches like anomaly detection might be more suitable for identifying unusual problems

**Fairness vs. Accuracy Trade-off**: Improving fairness would require collecting more diverse examples of problematic recipes or using techniques like oversampling, synthetic data generation, or cost-sensitive learning to better represent the minority group.

---

## Conclusion

This analysis explored whether recipe healthiness affects ratings and developed a model to predict common issues in recipes. Key findings:

1. **Nutritional content is not a significant factor in ratings** - users prioritize taste and execution over health
2. **Recipe characteristics can predict issue types** with moderate success (F1 = 0.58)
3. **The model exhibits fairness concerns**, performing better on highly-rated recipes
4. **Text features (names, ingredients) are more informative** than basic numerical features

Future work could improve the model by addressing class imbalance, incorporating more sophisticated NLP techniques, and collecting additional data on problematic recipes.

